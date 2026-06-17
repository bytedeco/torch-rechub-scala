package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.utils.DeviceSupport
import torchrec.TensorImplicits.RichTensor
import torchrec.utils.DeviceSupport.deviceOf

/**
 * Capsule network for multi-interest (MIND/Comirec).
 *
 * Parameters
 * ----------
 * embeddingDim : int
 *   Item embedding dimension.
 * seqLen : int
 *   Sequence length.
 * bilinearType : {0, 1, 2}, default 2
 *   0 for MIND, 2 for ComirecDR.
 * interestNum : int, default 4
 *   Number of interests.
 * routingTimes : int, default 3
 *   Routing iterations.
 * reluLayer : bool, default False
 *   Whether to apply ReLU after routing.
 *
 * Shape
 * -----
 * Input
 *   itemEb : ``(B, L, D)``
 *   mask : ``(B, L, 1)``
 * Output
 *   ``(B, interest_num, D)``
 */



class CapsuleNetwork(
                      embeddingDim: Int,
                      seqLen: Int,
                      bilinearType: Int = 2,
                      interestNum: Int = 4,
                      routingTimes: Int = 3,
                      reluLayer: Boolean = false,
                      device: String = DeviceSupport.backend
                    ) extends Module {
  private val h = embeddingDim
  private val s = seqLen
  private val k = interestNum
  private val dev = deviceOf(device)
  private val tensorOptsCPU = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
  private val tensorOptsDev = tensorOptsCPU.device(new DeviceOptional(dev))

  // 1. ReLU 仅在开启时创建，bias=False 对齐Python
  private val relu: Option[SequentialImpl] = if (reluLayer) {
    val seq = new SequentialImpl()
    val lin = new LinearImpl(embeddingDim, embeddingDim)
    seq.push_back("linear", lin)
    seq.push_back("relu", new ReLUImpl())
    register_module("relu", seq)
    Some(seq)
  } else None

  private var linear: Option[LinearImpl] = None
  private var w: Option[Tensor] = None

  bilinearType match {
    case 0 => // MIND: bias=False
      val l = new LinearImpl(embeddingDim, embeddingDim)
      register_module("linear", l)
      linear = Some(l)
    case 1 =>
      val l = new LinearImpl(embeddingDim, embeddingDim * interestNum)
      register_module("linear", l)
      linear = Some(l)
    case _ => // ComirecDR w参数，绑定device
      val wt = torch.rand(
        Array[Long](1L, seqLen.toLong, interestNum.toLong * embeddingDim.toLong, embeddingDim.toLong),
        tensorOptsDev
      )
      register_parameter("w", wt)
      w = Some(wt)
  }

  def forward(itemEb: Tensor, mask: Tensor): Tensor = {
    val batchSize = itemEb.size(0)
    require(itemEb.device().equals(dev), s"itemEb device mismatch, layer=$dev, input=${itemEb.device()}")

    // Step 1: Bilinear transform
    var itemEbHat = bilinearType match {
      case 0 =>
        val out = linear.get.forward(itemEb)
        out.repeat(Array(1L, 1L, interestNum.toLong) *)
      case 1 =>
        linear.get.forward(itemEb)
      case _ => // ComirecDR
        val u = itemEb.unsqueeze(2)
        val wt = w.get
        torch.sum(wt.narrow(1, 0, seqLen) * u, 3)
    }

    // Reshape & transpose to align with Python [B, K, L, D]
    val reshaped = itemEbHat.reshape(Array(batchSize, seqLen, interestNum, embeddingDim) *)
    val transposed = reshaped.transpose(1, 2).contiguous()
    itemEbHat = transposed.reshape(Array(batchSize, interestNum, seqLen, embeddingDim) *)

    val itemEbHatIter = itemEbHat.detach()

    // Initialize capsuleWeight: [B, K, L]
    var capsuleWeight = if (bilinearType > 0) {
      torch.zeros(
        Array[Long](batchSize.toLong, interestNum, seqLen),
        tensorOptsDev.requires_grad(new BoolOptional(false))
      )
    } else {
      torch.randn(
        Array[Long](batchSize.toLong, interestNum, seqLen),
        tensorOptsDev.requires_grad(new BoolOptional(false))
      )
    }

    var interestCapsule: Tensor = torch.empty()

    for (i <- 0 until routingTimes) {
      val attenMask = mask.unsqueeze(1).repeat(Array(1L, interestNum.toLong, 1L, 1L) *)
      val paddings = torch.zeros_like(attenMask)

      val capsuleSoftmaxWeight = torch.softmax(capsuleWeight, -1) // [B, K, L]

      // Safe mask expansion: [B, K, L] -> [B, K, L, 1]
      val capsuleSoftmaxWeight4d = capsuleSoftmaxWeight.unsqueeze(-1)
      val maskedWeight4d = torch.where(torch.eq(attenMask, new Scalar(0f)), paddings, capsuleSoftmaxWeight4d)

      // weight: [B, K, 1, L]
      val unsqueezedWeight = maskedWeight4d.squeeze(-1).unsqueeze(2)

      // Use detached tensor for first N-1 iterations, raw tensor for the last
      val targetItemEbHat = if (i < routingTimes - 1) itemEbHatIter else itemEbHat
      interestCapsule = torch.matmul(unsqueezedWeight, targetItemEbHat) // [B, K, 1, D]

      // CRITICAL FIX: Explicit unsqueeze bypasses JavaCPP keepdim=True mapping trap
      val rawNorm = torch.sum(torch.square(interestCapsule), -1) // [B, K, 1]
      val capNorm = rawNorm.unsqueeze(-1) // [B, K, 1, 1]

      val scalarFactor = capNorm.div(capNorm.add(new Scalar(1f)))
        .div(torch.sqrt(capNorm.add(new Scalar(1e-9f))))
      interestCapsule = scalarFactor.mul(interestCapsule)

      // Update weights only on intermediate routing steps
      if (i < routingTimes - 1) {
        val capT = interestCapsule.transpose(2, 3).contiguous() // [B, K, D, 1]
        val deltaWeight = torch.matmul(itemEbHatIter, capT) // [B, K, L, D] @ [B, K, D, 1] = [B, K, L, 1]

        // Squeeze the last dim to match capsuleWeight [B, K, L]
        capsuleWeight = capsuleWeight.add(deltaWeight.squeeze(-1))
      }
    }

    // Output reshape [B, K, 1, D] -> [B, K, D]
    val result = interestCapsule.reshape(Array(batchSize, interestNum, embeddingDim) *)
    if (reluLayer && relu.isDefined) relu.get.forward(result) else result
  }


}

object CapsuleNetwork {
  def apply(
             embeddingDim: Int,
             seqLen: Int,
             bilinearType: Int = 2,
             interestNum: Int = 4,
             routingTimes: Int = 3,
             reluLayer: Boolean = false,
             device: String = DeviceSupport.backend
           ): CapsuleNetwork = new CapsuleNetwork(embeddingDim, seqLen, bilinearType, interestNum, routingTimes, reluLayer, device)
}

//class CapsuleNetwork(
//  embeddingDim: Int,
//  seqLen: Int,
//  bilinearType: Int = 2,
//  interestNum: Int = 4,
//  routingTimes: Int = 3,
//  reluLayer: Boolean = false,
//  device: String = DeviceSupport.backend
//) extends Module {
//
//  private val h = embeddingDim
//  private val s = seqLen
//  private val k = interestNum
//
//  private var linear: Option[LinearImpl] = None
//  private var w: Option[Tensor] = None
//  private val relu = if (reluLayer) {
//    val r = new SequentialImpl()
//    r.push_back("linear", new LinearImpl(embeddingDim, embeddingDim))
//    r.push_back("relu", new ReLUImpl())
//    register_module("relu", r)
//    Some(r)
//  } else None
//
//  bilinearType match {
//    case 0 => // MIND
//      linear = Some {
//        val l = new LinearImpl(embeddingDim, embeddingDim)
//        register_module("linear", l)
//        l
//      }
//    case 1 =>
//      linear = Some {
//        val l = new LinearImpl(embeddingDim, embeddingDim * interestNum)
//        register_module("linear", l)
//        l
//      }
//    case _ =>
//      w = Some {
//        val wt = torch.rand(Array[Long](1L, seqLen.toLong, interestNum.toLong * embeddingDim.toLong, embeddingDim.toLong),
//          new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
//        register_parameter("w", wt)
//        wt
//      }
//  }
//
//  def forward(itemEb: Tensor, mask: Tensor): Tensor = {
//    val batchSize = itemEb.size(0)
//
//    val itemEbHat = bilinearType match {
//      case 0 =>
//        val out = linear.get.forward(itemEb)
//        out.repeat(1, 1, interestNum)
//      case 1 =>
//        linear.get.forward(itemEb)
//      case _ =>
//        val u = itemEb.unsqueeze(2)
//        val wt = w.get
//        torch.sum(wt.narrow(1, 0, seqLen) * u, 3)
//    }
//
//    val reshaped = itemEbHat.reshape(batchSize, seqLen, interestNum, embeddingDim)
//    val transposed = reshaped.transpose(1, 2).contiguous()
//    val finalReshaped = transposed.reshape(Array[Long](batchSize, interestNum, seqLen, embeddingDim))
//
//    val itemEbHatIter = if (true) finalReshaped.detach() else finalReshaped
//
//    var capsuleWeight = if (bilinearType > 0) {
//      torch.zeros(Array[Long](batchSize.toLong, interestNum, seqLen),
//        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
//    } else {
//      torch.randn(Array[Long](batchSize.toLong, interestNum, seqLen),
//        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
//    }.to(itemEb.device(), ScalarType.Float)
//
//    for (iter <- 0 until routingTimes) {
//      val attenMask = mask.unsqueeze(1).repeat(1, interestNum, 1)
//      val paddings = torch.zeros_like(attenMask)
//
//      val capsuleSoftmaxWeight = torch.softmax(capsuleWeight, -1)
//      val maskedWeight = torch.where(torch.eq(attenMask, new Scalar(0)), paddings, capsuleSoftmaxWeight)
//      val unsqueezedWeight = maskedWeight.unsqueeze(2)
//
//      if (iter < 2) {
//        val interestCapsule = torch.matmul(unsqueezedWeight, itemEbHatIter)
//        val capNorm = torch.sum(torch.square(interestCapsule), -1, 1L)
//        val scalarFactor = capNorm.div(capNorm.add(new Scalar(1.0f))).div(torch.sqrt(capNorm.add(new Scalar(1e-9f))))
//        val scaledCapsule = scalarFactor * interestCapsule
//
//        val deltaWeight = torch.matmul(itemEbHatIter.transpose(2, 3).contiguous(), scaledCapsule)
//        val reshapedDelta = deltaWeight.reshape(Array[Long](batchSize, interestNum, seqLen))
//        capsuleWeight = capsuleWeight.add(reshapedDelta)
//      } else {
//        val interestCapsule = torch.matmul(unsqueezedWeight, itemEbHatIter)
//        val capNorm = torch.sum(torch.square(interestCapsule), -1, 1L)
//        val scalarFactor = capNorm.div(capNorm.add(new Scalar(1.0f))).div(torch.sqrt(capNorm.add(new Scalar(1e-9f))))
//        val scaledCapsule = scalarFactor * interestCapsule
//      }
//    }
//
//    val result = itemEbHatIter.reshape(Array[Long](batchSize, interestNum, embeddingDim))
//
//    if (reluLayer && relu.isDefined) {
//      relu.get.forward(result)
//    } else {
//      result
//    }
//  }
//}
//
///**
// * CapsuleNetwork companion object with factory methods.
// */
//object CapsuleNetwork {
//  def apply(embeddingDim: Int, seqLen: Int, bilinearType: Int = 2, interestNum: Int = 4,
//            routingTimes: Int = 3, reluLayer: Boolean = false, device: String = DeviceSupport.backend): CapsuleNetwork = {
//    new CapsuleNetwork(embeddingDim, seqLen, bilinearType, interestNum, routingTimes, reluLayer, device)
//  }
//}
