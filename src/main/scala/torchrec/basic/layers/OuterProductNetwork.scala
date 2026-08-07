package torchrec.basic.layers

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

import torchrec.utils.DeviceSupport

/**
 * Outer Product Network for computing cross-feature interactions.
 *
 * Computes the outer product between field pairs to capture cross-feature interactions.
 * Three kernel types are supported:
 *   - "mat": Matrix kernel — each pair has a learnable (embed_dim x embed_dim) matrix
 *   - "vec": Vector kernel — each pair has a learnable embed_dim vector
 *   - "num": Scalar kernel — each pair has a single learnable scalar weight
 *
 * Reference: "Product-based Neural Networks for User Response Prediction" (Song et al., 2016)
 *
 * @param numFields   Number of feature fields
 * @param embedDim    Embedding dimension
 * @param kernelType  One of "mat", "vec", "num"
 * @param device      Device to run on
 */
class OuterProductNetwork(
  numFields: Int,
  embedDim: Int,
  kernelType: String = "mat",
  device: String = DeviceSupport.backend
) extends Module {

  require(numFields >= 2, s"numFields must be >= 2, got $numFields")
  require(embedDim > 0, s"embedDim must be positive, got $embedDim")
  require(kernelType == "mat" || kernelType == "vec" || kernelType == "num",
    s"kernelType must be 'mat', 'vec', or 'num', got $kernelType")

  private val numPairs = (numFields * (numFields - 1)) / 2

  // Build pair index arrays for tensor indexing
  private val pairRowIndices: Array[Long] = {
    val rows = collection.mutable.ListBuffer[Long]()
    for (i <- 0 until numFields - 1) {
      for (j <- i + 1 until numFields) {
        rows += i
      }
    }
    rows.toArray
  }

  private val pairColIndices: Array[Long] = {
    val cols = collection.mutable.ListBuffer[Long]()
    for (i <- 0 until numFields - 1) {
      for (j <- i + 1 until numFields) {
        cols += j
      }
    }
    cols.toArray
  }

  // Kernel parameter initialization using torch.randn
  private val kernelInit: Tensor = kernelType match {
    case "mat" =>
      val k = torch.randn(Array(numPairs.toLong, embedDim.toLong, embedDim.toLong): _*).
        mul(new Scalar(math.sqrt(2.0 / (embedDim * 2)).toFloat))
      k
    case _ =>
      val k = torch.randn(Array(numPairs.toLong, embedDim.toLong): _*).
        mul(new Scalar(math.sqrt(2.0 / embedDim).toFloat))
      k
  }

  private val kernel = {
    val p = new Tensor()
    p.copy_(kernelInit)
    register_parameter("kernel", p)
    p
  }

  if (device != "cpu") {
    kernel.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
    ()
  }

  override def forward(embeddings: Tensor): Tensor = {
    // embeddings: (batch, num_fields, embed_dim)
    val batchSize = embeddings.size(0).toInt
    val dev = embeddings.device()

    // Build index tensors using torch.tensor
    val rowT = torch.tensor(
      pairRowIndices.map(_.toFloat),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long))
    ).to(dev, ScalarType.Long)
    val colT = torch.tensor(
      pairColIndices.map(_.toFloat),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long))
    ).to(dev, ScalarType.Long)

    // Gather p and q for each pair: (batch, num_pairs, embed_dim)
    val p = embeddings.index_select(1, rowT)
    val q = embeddings.index_select(1, colT)

    kernelType match {
      case "mat" =>
        // kernel: (num_pairs, embed_dim, embed_dim)
        // For each pair: p_pair @ kernel[p] @ q_pair^T
        val k = kernel.to(dev, ScalarType.Float)
        val result = torch.zeros(Array(batchSize.toLong, numPairs.toLong),
          new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(dev, ScalarType.Float)

        for (b <- 0 until batchSize) {
          for (pIdx <- 0 until numPairs) {
            val pVec = p.select(0, b.toLong).select(0, pIdx.toLong)
            val qVec = q.select(0, b.toLong).select(0, pIdx.toLong)
            val wMat = k.select(0, pIdx.toLong)
            val wq = torch.matmul(wMat, qVec)
            val dot = pVec.dot(wq)
            result.select(0, b.toLong).select(0, pIdx.toLong).copy_(dot)
          }
        }
        result

      case "vec" =>
        // kernel: (num_pairs, embed_dim)
        // (p * q) * kernel, then sum over embed_dim
        val k = kernel.to(dev, ScalarType.Float)
        val pq = p.mul(q)  // (batch, num_pairs, embed_dim)
        val kB = k.unsqueeze(0).expand(batchSize.toLong, numPairs.toLong, embedDim.toLong)
        pq.mul(kB).sum(2L)  // (batch, num_pairs)

      case "num" =>
        // kernel: (num_pairs,)
        // (p * q).sum(dim=embed) * kernel
        val k = kernel.to(dev, ScalarType.Float)
        val pq = p.mul(q).sum(2L)  // (batch, num_pairs)
        val kB = k.unsqueeze(0).expand(batchSize.toLong, numPairs.toLong)
        pq.mul(kB)  // (batch, num_pairs)
    }
  }
}
