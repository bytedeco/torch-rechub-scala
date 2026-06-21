package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import torchrec.Implicits._
import torchrec.TensorImplicits._

/**
 * Comirec-DR: Dynamic Routing Multi-Interest Framework
 * Reference: RecSys 2020
 */
class ComirecDR(
  features: List[Feature],
  sequenceFeature: SequenceFeature,
  embedDim: Int = 8,
  numInterests: Int = 4,
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private val featureEmbedding = new EmbeddingLayer(features, embedDim, device)
  register_module("featureEmbedding", featureEmbedding)

  private val sequenceEmbedding = new EmbeddingLayer(List(sequenceFeature), embedDim, device)
  register_module("sequenceEmbedding", sequenceEmbedding)

  // Dynamic Routing
  private val dynamicRouter = new DynamicRouter(embedDim, numInterests, numRoutings = 3)
  register_module("dynamicRouter", dynamicRouter)

  // MLP
  private val featSparseDim = Features.calcSparseDim(features)
  // When combining per-interest feature vectors we repeat the sparse features
  // and concat with each interest vector, then flatten. The flattened input
  // dimension is numInterests * (featSparseDim + embedDim)
  private val totalInputDim = numInterests * (featSparseDim + embedDim)

  private val tower = new MLP(totalInputDim, mlpDims, embedDim, "relu", dropout, device = device)
  register_module("tower", tower)

  def forward(
    features: Map[String, Tensor],
    sequenceIndices: Tensor
  ): Tensor = {
    val featEmb = featureEmbedding.forward(features)
    // sequenceEmbedding is built for sequence features; use getSequenceEmbedding
    val seqEmb = sequenceEmbedding.getSequenceEmbedding(sequenceFeature.name, sequenceIndices)

    // Dynamic routing to get interests
    val interests = dynamicRouter.forward(seqEmb)

    // Combine - repeat featEmb along interest dimension then cat
    val featExpanded = featEmb.unsqueeze(1).repeat(1, numInterests, 1)
    // Ensure both tensors are on the same device before concatenation
    val dev = featExpanded.device()
    val interestsOnDev = if (interests.device().equals(dev)) interests else interests.to(dev, interests.dtype())
    val vec = new TensorVector()
    vec.push_back(featExpanded)
    vec.push_back(interestsOnDev)
    val featWithInterests = torch.cat(vec, 2L)
    val batchSize = featEmb.size(0)
    val flattened = featWithInterests.view(batchSize, -1)
    // Debug info to help if dimensions mismatch
    try {
      System.err.println(s"[DEBUG ComirecDR] featEmb.sizes=${featEmb.sizes().vec().get().mkString(",")}, interests.sizes=${interests.sizes().vec().get().mkString(",")}, featWithInterests.sizes=${featWithInterests.sizes().vec().get().mkString(",")}, flattened.sizes=${flattened.sizes().vec().get().mkString(",")}, totalInputDim=${totalInputDim}")
    } catch { case _: Throwable => () }

    tower.forward(flattened)
  }
}

/**
 * Dynamic Router for Multiple Interests
 */
class DynamicRouter(
  embedDim: Int,
  numInterests: Int,
  numRoutings: Int = 3
) extends Module {

  def forward(x: Tensor): Tensor = {
    // x: (batch, seq_len, embed_dim)
    val batchSize = x.size(0).toInt

    // Initialize interest capsules
    var capsules = torch.randn(batchSize, numInterests, embedDim).mul(new Scalar(0.1f)).to(x.device(), org.bytedeco.pytorch.global.torch.ScalarType.Float)

    var weightedItems: Tensor = capsules
    for (r <- 0 until numRoutings) {
      // Compute similarity between items and interest capsules
      val expandedCaps = capsules.unsqueeze(1) // (batch, 1, num_interests, embed_dim)
      val expandedItems = x.unsqueeze(2) // (batch, seq_len, 1, embed_dim)
      val similarities = expandedCaps.mul(expandedItems).sum(3) // (batch, seq_len, num_interests)

      // Routing weights (softmax)
      val weights = similarities.softmax(2)

      // Update capsules
      weightedItems = x.unsqueeze(2).mul(weights.unsqueeze(3)).sum(1) // (batch, num_interests, embed_dim)

      if (r < numRoutings - 1) {
        capsules = weightedItems
      }
    }

    weightedItems
  }
}
