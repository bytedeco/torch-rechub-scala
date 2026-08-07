package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.nn.modules._
import org.bytedeco.pytorch.nn.modules.container._
import org.bytedeco.pytorch.nn.options._
import org.bytedeco.pytorch.optim._
import org.bytedeco.pytorch.data.datasets._
import org.bytedeco.pytorch.data.options._
import org.bytedeco.pytorch.data.sampler._
import org.bytedeco.pytorch.distributed._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * MEMBA: Memory-Bidirection for Sequential Recommendation
 */
class MEMBA(
             features: List[Feature],
             sequenceFeatures: List[SequenceFeature],
             embedDim: Int = 8,
             numMemorySlots: Int = 16,
             numHeads: Int = 2,
             mlpDims: List[Long] = List(256L, 128L),
             dropout: Float = 0.2f,
             device: String = DeviceSupport.backend
           ) extends Module {

  require(embedDim % numHeads == 0, "embedDim must be divisible by numHeads")

  private val headDim = embedDim / numHeads
  private val targetDevice = new Device(device)

  // 统一嵌入层
  private val embeddingLayer = new EmbeddingLayer(features ++ sequenceFeatures, embedDim, device)
  register_module("embeddingLayer", embeddingLayer)

  // Encoders
  private val forwardEncoder = new MLP(embedDim, List(embedDim.toLong), embedDim, "relu", dropout, device = device)
  register_module("forwardEncoder", forwardEncoder)

  private val backwardEncoder = new MLP(embedDim, List(embedDim.toLong), embedDim, "relu", dropout, device = device)
  register_module("backwardEncoder", backwardEncoder)

  // Memory banks
  private val fwdMemBank = torch.zeros(
    Array(1L, numMemorySlots.toLong, embedDim.toLong),
    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
  ).to(targetDevice, ScalarType.Float)
  register_parameter("fwdMemBank", fwdMemBank)

  private val bwdMemBank = torch.zeros(
    Array(1L, numMemorySlots.toLong, embedDim.toLong),
    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
  ).to(targetDevice, ScalarType.Float)
  register_parameter("bwdMemBank", bwdMemBank)

  // Gate & Attention
  private val memoryGate = new LinearImpl(embedDim, embedDim)
  memoryGate.to(targetDevice, false)
  register_module("memoryGate", memoryGate)

  private val queryProj = new LinearImpl(embedDim, embedDim)
  private val keyProj = new LinearImpl(embedDim, embedDim)
  private val valueProj = new LinearImpl(embedDim, embedDim)
  queryProj.to(targetDevice, false)
  keyProj.to(targetDevice, false)
  valueProj.to(targetDevice, false)
  register_module("queryProj", queryProj)
  register_module("keyProj", keyProj)
  register_module("valueProj", valueProj)

  // Fusion & Final MLP
  private val fusionDim = embedDim * 3
  private val fusionMLP = new MLP(fusionDim, List(embedDim.toLong), embedDim, "relu", dropout, device = device)
  register_module("fusionMLP", fusionMLP)

  private val mlp = new MLP(embedDim * 2, mlpDims, 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  // ===========================================================================
  // FORWARD —— 彻底删除所有 .head，永无空迭代！
  // ===========================================================================
  def forward(
               sparseFeats: Map[String, Tensor],
               sequenceFeats: Map[String, Tensor],
               targetIdx: Tensor
             ): Tensor = {

    // ----------------==== 【致命修复】从 targetIdx 获取 batchSize，绝不从空Map取 ====----------------
    val batchSize = targetIdx.size(0)

    // 1. 静态特征（允许空）
    val featEmb = try {
      embeddingLayer.forward(sparseFeats, Map.empty, squeeze = true)
    } catch {
      case _: Exception => torch.zeros(Array(batchSize, embedDim.toLong),
        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(targetDevice)))
    }
    featEmb.to(targetDevice, ScalarType.Float)

    // 2. 序列特征
    val seqEmb = embeddingLayer.forward(Map.empty, sequenceFeats, squeeze = false)
    val seqEmbFixed = if (seqEmb.dim() == 2) seqEmb.unsqueeze(1) else seqEmb
    seqEmbFixed.to(targetDevice, ScalarType.Float)

    // 3. 编码
    val fwdPooled = seqEmbFixed.mean(1)
    val fwdH = forwardEncoder.forward(fwdPooled)
    val bwdH = backwardEncoder.forward(fwdPooled)

    // 4. 记忆更新
    val fwdUpdate = memoryGate.forward(fwdH)
    val bwdUpdate = memoryGate.forward(bwdH)

    val fwdMem = fwdMemBank.expand(batchSize, numMemorySlots.toLong, embedDim.toLong)
    val bwdMem = bwdMemBank.expand(batchSize, numMemorySlots.toLong, embedDim.toLong)

    // 5. 目标 Embedding（从历史序列平均得到，不依赖任何 key）
    val targetEmb = seqEmbFixed.mean(1)
    targetEmb.to(targetDevice, ScalarType.Float)

    // 注意力读取
    val readFwd = memoryAttentionRead(targetEmb, fwdMem, fwdUpdate)
    val readBwd = memoryAttentionRead(targetEmb, bwdMem, bwdUpdate)

    // 输出
    val fused = torch.cat(new TensorVector(targetEmb, readFwd, readBwd), 1)
    val fusedOut = fusionMLP.forward(fused)
    val combined = torch.cat(new TensorVector(featEmb, fusedOut), 1)
    val logits = mlp.forward(combined)

    logits.to(targetDevice, ScalarType.Float)
  }

  // ===================== 注意力 =====================
  private def memoryAttentionRead(query: Tensor, memory: Tensor, update: Tensor): Tensor = {
    val gate = update.sigmoid().unsqueeze(1)
    val gatedMem = memory.mul(gate)

    val batchSize = query.size(0)
    val q = queryProj.forward(query).view(batchSize, numHeads, headDim)
    val k = keyProj.forward(gatedMem).view(batchSize, numMemorySlots, numHeads, headDim).transpose(1, 2)
    val v = valueProj.forward(gatedMem).view(batchSize, numMemorySlots, numHeads, headDim).transpose(1, 2)

    val scale = math.sqrt(headDim).toFloat
    val scores = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)).div(new Scalar(scale))
    val attn = scores.squeeze(2).softmax(-1)

    val out = torch.matmul(attn.unsqueeze(2), v).squeeze(2)
    out.transpose(1, 2).contiguous().view(batchSize, embedDim)
  }
}

//package torchrec.models.ranking
//
//import torchrec.basic.features._
//import torchrec.basic.layers._
//import torchrec.Implicits._
//import torchrec.utils.DeviceSupport
//
//import org.bytedeco.pytorch._
//import org.bytedeco.pytorch.global.torch
//import org.bytedeco.pytorch.global.torch.ScalarType
//
//import scala.collection.mutable
//
///**
// * MEMBA: Memory-Bidirection for Sequential Recommendation
// */
//class MEMBA(
//             features: List[Feature],
//             sequenceFeatures: List[SequenceFeature],
//             embedDim: Int = 8,
//             numMemorySlots: Int = 16,
//             numHeads: Int = 2,
//             mlpDims: List[Long] = List(256L, 128L),
//             dropout: Float = 0.2f,
//             device: String = DeviceSupport.backend
//           ) extends Module {
//
//  require(features.nonEmpty, "features cannot be empty")
//  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")
//  require(embedDim % numHeads == 0, "embedDim must be divisible by numHeads")
//
//  private val headDim = embedDim / numHeads
//  private val targetDevice = new Device(device)
//
//  // 统一嵌入层
//  private val allFeatures = features ++ sequenceFeatures
//  private val embeddingLayer = new EmbeddingLayer(allFeatures, embedDim, device)
//  register_module("embeddingLayer", embeddingLayer)
//
//  // Encoders
//  private val forwardEncoder = new MLP(embedDim, List(embedDim.toLong), embedDim, "relu", dropout, device = device)
//  register_module("forwardEncoder", forwardEncoder)
//
//  private val backwardEncoder = new MLP(embedDim, List(embedDim.toLong), embedDim, "relu", dropout, device = device)
//  register_module("backwardEncoder", backwardEncoder)
//
//  // Memory banks
//  private val fwdMemBank = torch.zeros(
//    Array(1L, numMemorySlots.toLong, embedDim.toLong),
//    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
//  ).to(targetDevice, ScalarType.Float)
//  register_parameter("fwdMemBank", fwdMemBank)
//
//  private val bwdMemBank = torch.zeros(
//    Array(1L, numMemorySlots.toLong, embedDim.toLong),
//    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
//  ).to(targetDevice, ScalarType.Float)
//  register_parameter("bwdMemBank", bwdMemBank)
//
//  // Memory gate
//  private val memoryGate = new LinearImpl(embedDim, embedDim)
//  memoryGate.to(targetDevice, false)
//  register_module("memoryGate", memoryGate)
//
//  // Attention
//  private val queryProj = new LinearImpl(embedDim, embedDim)
//  private val keyProj = new LinearImpl(embedDim, embedDim)
//  private val valueProj = new LinearImpl(embedDim, embedDim)
//  queryProj.to(targetDevice, false)
//  keyProj.to(targetDevice, false)
//  valueProj.to(targetDevice, false)
//  register_module("queryProj", queryProj)
//  register_module("keyProj", keyProj)
//  register_module("valueProj", valueProj)
//
//  // Fusion
//  private val fusionDim = embedDim * 3
//  private val fusionMLP = new MLP(fusionDim, List(embedDim.toLong), embedDim, "relu", dropout, device = device)
//  register_module("fusionMLP", fusionMLP)
//
//  // Final MLP
//  private val totalInputDim = features.size * embedDim
//  private val finalMLP = new MLP(totalInputDim, mlpDims, 1, "relu", dropout, device = device)
//  register_module("finalMLP", finalMLP)
//
//  // ===========================================================================
//  // FORWARD —— 完全修复 next on empty iterator
//  // ===========================================================================
//  override def forward(
//               sparseFeats: Map[String, Tensor],
//               sequenceFeats: Map[String, Tensor],
//               targetIdx: Tensor
//             ): Tensor = {
//
//    val batchSize = sparseFeats.values.head.size(0)
//
//    // 1. 静态特征
//    val featEmb = embeddingLayer.forward(sparseFeats, Map.empty, squeeze = true)
//    featEmb.to(targetDevice, ScalarType.Float)
//
//    // 2. 序列特征（安全获取，永不崩溃）
//    val seqEmbTensor = embeddingLayer.forward(Map.empty, sequenceFeats, squeeze = false)
//    val seqEmb = if (seqEmbTensor.dim() == 2) seqEmbTensor.unsqueeze(1) else seqEmbTensor
//    seqEmb.to(targetDevice, ScalarType.Float)
//
//    // 3. 编码
//    val seqPooled = seqEmb.mean(1)
//    val fwdH = forwardEncoder.forward(seqPooled)
//    val bwdH = backwardEncoder.forward(seqPooled)
//
//    // 4. Memory update
//    val fwdUpdate = memoryGate.forward(fwdH)
//    val bwdUpdate = memoryGate.forward(bwdH)
//
//    val fwdMem = fwdMemBank.expand(batchSize, numMemorySlots.toLong, embedDim.toLong)
//    val bwdMem = bwdMemBank.expand(batchSize, numMemorySlots.toLong, embedDim.toLong)
//
//    // 5. Target Embedding（绝对安全）
//    val targetEmbRaw = embeddingLayer.forward(Map.empty, Map(sequenceFeatures.head.name -> targetIdx), squeeze = false)
//    val targetEmb = if (targetEmbRaw.dim() == 3) targetEmbRaw.squeeze(1) else targetEmbRaw
//    targetEmb.to(targetDevice, ScalarType.Float)
//
//    // 6. Attention
//    val readFwd = memoryAttentionRead(targetEmb, fwdMem, batchSize, fwdUpdate)
//    val readBwd = memoryAttentionRead(targetEmb, bwdMem, batchSize, bwdUpdate)
//
//    // 7. Output
//    val fused = torch.cat(new TensorVector(targetEmb, readFwd, readBwd), 1)
//    val fusedOut = fusionMLP.forward(fused)
//
//    val combined = torch.cat(new TensorVector(featEmb, fusedOut), 1)
//    val logits = finalMLP.forward(combined)
//
//    logits.to(targetDevice, ScalarType.Float)
//  }
//
//  private def memoryAttentionRead(
//                                   query: Tensor,
//                                   memory: Tensor,
//                                   batchSize: Long,
//                                   update: Tensor
//                                 ): Tensor = {
//    val gate = update.sigmoid().unsqueeze(1)
//    val gatedMem = memory.mul(gate)
//
//    val q = queryProj.forward(query).view(batchSize, numHeads, headDim)
//    val k = keyProj.forward(gatedMem).view(batchSize, numMemorySlots, numHeads, headDim).transpose(1, 2)
//    val v = valueProj.forward(gatedMem).view(batchSize, numMemorySlots, numHeads, headDim).transpose(1, 2)
//
//    val scale = math.sqrt(headDim).toFloat
//    val scores = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)).div(new Scalar(scale))
//    val attn = scores.squeeze(2).softmax(-1)
//
//    val output = torch.matmul(attn.unsqueeze(2), v).squeeze(2)
//    output.transpose(1, 2).contiguous().view(batchSize, embedDim)
//  }
//}

//package torchrec.models.ranking
//
//import torchrec.basic.features._
//import torchrec.basic.layers._
//import torchrec.Implicits._
//import torchrec.utils.DeviceSupport
//
//import org.bytedeco.pytorch._
//import org.bytedeco.pytorch.global.torch
//import org.bytedeco.pytorch.global.torch.ScalarType
//
//import scala.collection.mutable
//
///**
// * MEMBA: Memory-Bidirection for Sequential Recommendation
// *
// * Maintains two memory banks (forward and backward) that store compressed
// * representations of the user's behavior sequence. Memory slots are read with
// * target-aware attention. The fused representation feeds an MLP for CTR.
// *
// * Reference: "MEMBA: Memory-Bidirection for Sequential Recommendation" (2023)
// *
// * Architecture:
// *   Sparse Input → EmbeddingLayer → Bi-MLP → Memory Banks
// *   → Target-Aware Attention Read → Fusion → MLP → CTR Logit
// */
//class MEMBA(
//  features: List[Feature],
//  sequenceFeatures: List[SequenceFeature],
//  embedDim: Int = 8,
//  numMemorySlots: Int = 16,
//  numHeads: Int = 2,
//  mlpDims: List[Long] = List(256L, 128L),
//  dropout: Float = 0.2f,
//  device: String = DeviceSupport.backend
//) extends Module {
//
//  require(features.nonEmpty, "features cannot be empty")
//  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")
//  require(embedDim % numHeads == 0, "embedDim must be divisible by numHeads")
//
//  private val headDim = embedDim / numHeads
//
//  // Embedding layers
//  private val featureEmbedding = new EmbeddingLayer(features, embedDim, device)
//  register_module("featureEmbedding", featureEmbedding)
//
//  private val sequenceEmbedding = new EmbeddingLayer(sequenceFeatures, embedDim, device)
//  register_module("sequenceEmbedding", sequenceEmbedding)
//
//  // Sequence encoders: MLPs for forward/backward pass
//  private val forwardEncoder = new MLP(embedDim, List(embedDim.toLong), embedDim, "relu", dropout, device = device)
//  register_module("forwardEncoder", forwardEncoder)
//
//  private val backwardEncoder = new MLP(embedDim, List(embedDim.toLong), embedDim, "relu", dropout, device = device)
//  register_module("backwardEncoder", backwardEncoder)
//
//  // Two memory banks
//  private val fwdMemInit = torch.zeros(
//    Array(1L, numMemorySlots.toLong, embedDim.toLong),
//    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
//  private val fwdMemBank = fwdMemInit.expand(1L, numMemorySlots.toLong, embedDim.toLong).clone()
//  register_parameter("fwdMemBank", fwdMemBank)
//
//  private val bwdMemInit = torch.zeros(
//    Array(1L, numMemorySlots.toLong, embedDim.toLong),
//    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
//  private val bwdMemBank = bwdMemInit.expand(1L, numMemorySlots.toLong, embedDim.toLong).clone()
//  register_parameter("bwdMemBank", bwdMemBank)
//
//  // Memory update gate
//  private val memoryGate = new LinearImpl(embedDim, embedDim)
//  register_module("memoryGate", memoryGate)
//
//  // Attention projections for memory read
//  private val queryProj = new LinearImpl(embedDim, embedDim)
//  private val keyProj = new LinearImpl(embedDim, embedDim)
//  private val valueProj = new LinearImpl(embedDim, embedDim)
//  register_module("queryProj", queryProj)
//  register_module("keyProj", keyProj)
//  register_module("valueProj", valueProj)
//
//  // Target attention
//  private val targetAttention = new ActivationUnit(embedDim, embedDim / 2, device)
//  register_module("targetAttention", targetAttention)
//
//  // Fusion MLP
//  private val fusionDim = embedDim * 3
//  private val fusionMLP = new MLP(fusionDim, List(embedDim.toLong), embedDim, "relu", dropout, device = device)
//  register_module("fusionMLP", fusionMLP)
//
//  // Final MLP
//  private val sparseDim = Features.calcSparseDim(features)
//  private val totalDim = sparseDim + embedDim
//  private val mlp = new MLP(totalDim, mlpDims, 1, "relu", dropout, device = device)
//  register_module("mlp", mlp)
//
//  if (device != "cpu") {
//    val dev = new org.bytedeco.pytorch.Device(device)
//    featureEmbedding.to(dev, false)
//    sequenceEmbedding.to(dev, false)
//    forwardEncoder.to(dev, false)
//    backwardEncoder.to(dev, false)
//    memoryGate.to(dev, false)
//    queryProj.to(dev, false)
//    keyProj.to(dev, false)
//    valueProj.to(dev, false)
//    fusionMLP.to(dev, false)
//    mlp.to(dev, false)
//  }
//
//  override def forward(
//    sparseFeats: Map[String, Tensor],
//    sequenceFeats: Map[String, Tensor],
//    targetIdx: Tensor
//  ): Tensor = {
//    // Sparse embeddings
//    val featEmb = featureEmbedding.forward(sparseFeats)
//
//    // Sequence embeddings
//    val seqEmbs = sequenceFeatures.flatMap { seqFeat =>
//      sequenceFeats.get(seqFeat.name).map { indices =>
//        sequenceEmbedding.getEmbedding(seqFeat.name, indices)
//      }
//    }
//
//    if (seqEmbs.isEmpty) {
//      throw new IllegalArgumentException("No sequence embeddings found")
//    }
//
//    val seqEmb = if (seqEmbs.size == 1) {
//      seqEmbs.head
//    } else {
//      val vec = new TensorVector(seqEmbs.size.toLong)
//      seqEmbs.foreach(vec.push_back)
//      torch.cat(vec, 1)
//    }
//
//    val batchSize = seqEmb.size(0).toInt
//    val seqLen = seqEmb.size(1).toInt
//
//    // Forward encoding: mean pool sequence, then MLP
//    val fwdPooled = seqEmb.mean(1)  // (batch, embed_dim)
//    val fwdH = forwardEncoder.forward(fwdPooled)  // (batch, embed_dim)
//
//    // Backward encoding: reverse sequence
//    val bwdH = backwardEncoder.forward(fwdPooled)  // (batch, embed_dim)
//
//    // Memory write with gating
//    val fwdUpdate = memoryGate.forward(fwdH)
//    val bwdUpdate = memoryGate.forward(bwdH)
//
//    // Expand memory banks
//    val fwdMem = fwdMemBank.expand(batchSize.toLong, numMemorySlots.toLong, embedDim.toLong)
//    val bwdMem = bwdMemBank.expand(batchSize.toLong, numMemorySlots.toLong, embedDim.toLong)
//
//    // Target embedding
//    val targetEmb = if (targetIdx.dim() == 2L && targetIdx.size(0) == batchSize && targetIdx.size(1) == 1) {
//      sequenceFeatures.headOption match {
//        case Some(sf) =>
//          val rawEmb = sequenceEmbedding.getEmbedding(sf.name, targetIdx.toType(ScalarType.Long))
//          if (rawEmb.dim() == 3L) rawEmb.squeeze(1) else rawEmb
//        case None => throw new IllegalArgumentException("No sequence features defined")
//      }
//    } else {
//      throw new IllegalArgumentException("targetIdx shape mismatch")
//    }
//
//    // Memory attention read
//    val readFwd = memoryAttentionRead(targetEmb, fwdMem, batchSize, fwdUpdate)
//    val readBwd = memoryAttentionRead(targetEmb, bwdMem, batchSize, bwdUpdate)
//
//    // Fusion
//    val fused = torch.cat(new TensorVector(targetEmb, readFwd, readBwd), 1)
//    val fusedOut = fusionMLP.forward(fused)
//
//    // Final MLP
//    val combined = torch.cat(new TensorVector(featEmb, fusedOut), 1)
//    val logits = mlp.forward(combined)
//    logits
//  }
//
//  /** Multi-head attention read from memory bank. */
//  private def memoryAttentionRead(
//    query: Tensor,
//    memory: Tensor,
//    batchSize: Int,
//    update: Tensor
//  ): Tensor = {
//    val gate = update.sigmoid()
//    val gatedMem = memory.mul(gate.unsqueeze(1))
//
//    val q = queryProj.forward(query).view(batchSize, numHeads, headDim).transpose(0, 1)
//    val k = keyProj.forward(gatedMem).view(batchSize, numMemorySlots, numHeads, headDim).transpose(1, 2)
//    val v = valueProj.forward(gatedMem)
//
//    val scale = new Scalar(scala.math.sqrt(headDim.toDouble).toFloat)
//    val qT = q.unsqueeze(2)
//    val kT = k.transpose(2, 3)
//    val scores = torch.matmul(qT, kT).div(scale).squeeze(2)
//    val attnBatched = scores.transpose(0, 1).softmax(2)
//
//    val vBatched = v.view(batchSize, numMemorySlots, numHeads, headDim).transpose(1, 2)
//    val attnEx = attnBatched.unsqueeze(3)
//    val attended = vBatched.mul(attnEx).sum(2)
//
//    attended.transpose(1, 2).contiguous().view(batchSize, embedDim)
//  }
//}
