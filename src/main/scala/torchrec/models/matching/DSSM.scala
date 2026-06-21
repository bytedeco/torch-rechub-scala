package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * Deep Structured Semantic Model (DSSM) - Two Tower Architecture
 * Reference: Microsoft
 */
class DSSM(
  userFeatures: List[Feature],
  itemFeatures: List[Feature],
  embedDim: Int = 8,
  towerDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  // User tower
  private val userEmbedding = new EmbeddingLayer(userFeatures, embedDim, device)
  register_module("userEmbedding", userEmbedding)

  // Calculate total input dim: sparse features + sequence features (after pooling, sequence dim = embedDim)
  private val userSparseDim = Features.calcSparseDim(userFeatures)
  private val userSeqDim = Features.calcSequenceDimFromFeatures(userFeatures, "mean")
  private val userTowerInputDim = userSparseDim + userSeqDim
  private val userTower = new MLP(userTowerInputDim, towerDims, embedDim, "relu", dropout, device = device)
  register_module("userTower", userTower)

  // Item tower
  private val itemEmbedding = new EmbeddingLayer(itemFeatures, embedDim, device)
  register_module("itemEmbedding", itemEmbedding)

  private val itemSparseDim = Features.calcSparseDim(itemFeatures)
  private val itemSeqDim = Features.calcSequenceDimFromFeatures(itemFeatures, "mean")
  private val itemTowerInputDim = itemSparseDim + itemSeqDim
  private val itemTower = new MLP(itemTowerInputDim, towerDims, embedDim, "relu", dropout, device = device)
  register_module("itemTower", itemTower)

  if (device != "cpu") {
    userTower.to(new org.bytedeco.pytorch.Device(device), false)
    itemTower.to(new org.bytedeco.pytorch.Device(device), false)
  }

  def forward(
    userFeats: Map[String, Tensor],
    itemFeats: Map[String, Tensor]
  ): Tensor = {
    val userEmb = userEmbedding.forward(userFeats)
    val itemEmb = itemEmbedding.forward(itemFeats)

    val userOut = userTower.forward(userEmb)
    val itemOut = itemTower.forward(itemEmb)

    // Cosine similarity
    // Compute L2 norm manually: sqrt(sum(x^2))
    val userNorm = userOut.pow(new Scalar(2)).sum(1).sqrt()
    val itemNorm = itemOut.pow(new Scalar(2)).sum(1).sqrt()
    val prodNorms = userNorm.mul(itemNorm)
    val cosSim = userOut.mul(itemOut).sum(1).div(prodNorms.add(new Scalar(1e-8f)))
    cosSim.unsqueeze(1)
  }

  def userTowerForward(userFeats: Map[String, Tensor]): Tensor = {
    // Extract sequence features from userFeats:
    // - 1D (batch,) -> sparse
    // - 2D (batch, 1) -> sparse
    // - 2D (batch, seqLen>1) -> sequence
    // - 3D (batch, 1, seqLen) -> sequence (from DataLoader stacking)
    val (sparseFeats, sequenceFeats) = userFeats.partition { case (_, tensor) =>
      tensor.dim() match {
        case 1L => true  // (batch,) -> sparse
        case 2L => tensor.size(1) == 1L  // (batch, 1) -> sparse, (batch, seqLen>1) -> sequence
        case 3L => tensor.size(1) != 1L  // (batch, seqLen, seqLen) -> sparse, (batch, 1, seqLen) -> sequence
        case _ => false
      }
    }
    // For 3D sequence tensors (batch, 1, seqLen), squeeze to 2D (batch, seqLen)
    val processedSeqFeats = sequenceFeats.map { case (k, v) =>
      k -> (if (v.dim() == 3L && v.size(1) == 1L) v.squeeze(1) else v)
    }
    val userEmb = userEmbedding.forward(sparseFeats.toMap, processedSeqFeats.toMap, squeeze = false)
    val squeezed = if (userEmb.dim() == 2L && userEmb.size(1) == 1L) userEmb.squeeze(1) else userEmb
    userTower.forward(squeezed)
  }

  def itemTowerForward(itemFeats: Map[String, Tensor]): Tensor = {
    // Extract sequence features from itemFeats:
    // - 1D (batch,) -> sparse
    // - 2D (batch, 1) -> sparse
    // - 2D (batch, seqLen>1) -> sequence
    // - 3D (batch, 1, seqLen) -> sequence (from DataLoader stacking)
    val (sparseFeats, sequenceFeats) = itemFeats.partition { case (_, tensor) =>
      tensor.dim() match {
        case 1L => true  // (batch,) -> sparse
        case 2L => tensor.size(1) == 1L  // (batch, 1) -> sparse, (batch, seqLen>1) -> sequence
        case 3L => tensor.size(1) != 1L  // (batch, seqLen, seqLen) -> sparse, (batch, 1, seqLen) -> sequence
        case _ => false
      }
    }
    // For 3D sequence tensors (batch, 1, seqLen), squeeze to 2D (batch, seqLen)
    val processedSeqFeats = sequenceFeats.map { case (k, v) =>
      k -> (if (v.dim() == 3L && v.size(1) == 1L) v.squeeze(1) else v)
    }
    val itemEmb = itemEmbedding.forward(sparseFeats.toMap, processedSeqFeats.toMap, squeeze = false)
    val squeezed = if (itemEmb.dim() == 2L && itemEmb.size(1) == 1L) itemEmb.squeeze(1) else itemEmb
    itemTower.forward(squeezed)
  }
}
