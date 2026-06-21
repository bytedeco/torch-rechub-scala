package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * SINE - Self-supervised Interest Network
 * Reference: "SINE: Self-supervised Interest Network" (Zhang et al., 2021)
 */
class SINE(
  features: List[Feature],
  sequenceFeature: SequenceFeature,
  embedDim: Int = 8,
  numInterests: Int = 4,
  mlpDims: List[Long] = List(128L, 64L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private val embedding = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embedding)

  private val seqEmbedding = new EmbeddingLayer(List(sequenceFeature), embedDim, device)
  register_module("seqEmbedding", seqEmbedding)

  private val interestExtractor = new MLP(embedDim, List(numInterests * embedDim), embedDim, "relu", dropout, device = device)
  register_module("interestExtractor", interestExtractor)

  private val tower = new MLP(embedDim, mlpDims, embedDim, "relu", dropout, device = device)
  register_module("tower", tower)

  def forward(features: Map[String, Tensor], sequence: Tensor): Tensor = {
    val featEmb = embedding.forward(features)
    // sequence embedding must use getSequenceEmbedding
    val seqEmb = seqEmbedding.getSequenceEmbedding(sequenceFeature.name, sequence)
    val interests = interestExtractor.forward(seqEmb.mean(1))
    val combined = featEmb.add(interests)
    tower.forward(combined)
  }
}