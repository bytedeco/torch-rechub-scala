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
import torchrec.Implicits._

/**
 * Field-Aware Factorization Machine Layer
 *
 * Each field j has its own latent vector for each other field i.
 * This allows the model to learn different feature interactions per field pair.
 *
 * Reference: "Field-aware Factorization Machines for CTR Prediction" (Criteo, RecSys 2016)
 *
 * @param numFields   Number of feature fields
 * @param embedDim    Embedding dimension per field
 * @param device      Device to run on
 */
class FieldAwareFactorizationMachine(
  numFields: Int,
  embedDim: Int,
  device: String = DeviceSupport.backend
) extends Module {

  require(numFields >= 2, s"numFields must be >= 2, got $numFields")
  require(embedDim > 0, s"embedDim must be positive, got $embedDim")

  // Number of field pairs = C(numFields, 2) = numFields * (numFields - 1) / 2
  private val numPairs = (numFields * (numFields - 1)) / 2

  // Field-aware embeddings: (numFields, numFields - 1, embedDim)
  // ffee[j, i, :] = embedding of field i as seen by field j
  private val ffeeInit = {
    val t = torch.randn(Array(numFields.toLong, (numFields - 1).toLong, embedDim.toLong): _*).
      mul(new Scalar(math.sqrt(2.0 / embedDim).toFloat))
    t
  }

  private val ffee = {
    val p = new Tensor()
    p.copy_(ffeeInit)
    register_parameter("field_aware_embeddings", p)
    p
  }

  // Build pair index arrays
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

  if (device != "cpu") {
    ffee.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
  }

  override def forward(embeddings: Tensor): Tensor = {
    // embeddings: (batch, num_fields, embed_dim)
    val batchSize = embeddings.size(0).toInt
    val dev = embeddings.device()
    val ffeeDev = ffee.to(dev, ScalarType.Float)

    val result = torch.zeros(Array(batchSize.toLong, numPairs.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(dev, ScalarType.Float)

    for (pairIdx <- 0 until numPairs) {
      val i = pairRowIndices(pairIdx).toInt
      val j = pairColIndices(pairIdx).toInt

      // embeddings for fields i and j: (batch, embed_dim)
      val vi = embeddings.select(1, i)
      val vj = embeddings.select(1, j)

      // Field-aware vectors: ffee[j, i, :] and ffee[i, j, :]
      val vi_as_j = ffeeDev.select(0, j.toLong).select(0, i.toLong)
      val vj_as_i = ffeeDev.select(0, i.toLong).select(0, j.toLong)

      // interaction = (V_i dot V_j) * (V_i_as_j dot V_j_as_i)
      val direct = vi.mul(vj).sum(1L)
      val fieldAware = vi_as_j.mul(vj_as_i).sum()
      val interaction = direct.mul(new Scalar(fieldAware.item_float()))

      result.select(1, pairIdx.toLong).copy_(interaction)
    }

    // Sum over all pairs: (batch,)
    result.sum(1L)
  }
}
