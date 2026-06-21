package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import torchrec.Implicits._
import torchrec.TensorImplicits._

/**
 * Comirec-SA: Self-Attentive Multi-Interest Framework
 * Reference: RecSys 2020
 */
class ComirecSA(
  features: List[Feature],
  sequenceFeature: SequenceFeature,
  embedDim: Int = 8,
  numInterests: Int = 4,
  numHeads: Int = 2,
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private val featureEmbedding = new EmbeddingLayer(features, embedDim, device)
  register_module("featureEmbedding", featureEmbedding)

  private val sequenceEmbedding = new EmbeddingLayer(List(sequenceFeature), embedDim, device)
  register_module("sequenceEmbedding", sequenceEmbedding)

  // Multi-Interest Extractor with Self-Attention
  private val interestExtractor = new InterestExtractor(embedDim, numInterests, numHeads, dropout, device)
  register_module("interestExtractor", interestExtractor)

  // MLP
  private val featSparseDim = Features.calcSparseDim(features)
  // The flattened input after repeating sparse features per interest and
  // concatenating with each interest embedding is numInterests * (featSparseDim + embedDim)
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

    // Extract multiple interests
    val interests = interestExtractor.forward(seqEmb)
    // interests: (batch, num_interests, embed_dim)

    // Combine with features
    val featExpanded = featEmb.unsqueeze(1).repeat(1, numInterests, 1)
    // Align devices before concatenation to avoid device-less tensor errors
    val dev = featExpanded.device()
    val interestsOnDev = if (interests.device().equals(dev)) interests else interests.to(dev, interests.dtype())
    val vec = new TensorVector()
    vec.push_back(featExpanded)
    vec.push_back(interestsOnDev)
    val featWithInterests = torch.cat(vec, 2L)
    val batchSize = featEmb.size(0)
    val flattened = featWithInterests.view(batchSize, -1)
    try {
      System.err.println(s"[DEBUG ComirecSA] featEmb.sizes=${featEmb.sizes().vec().get().mkString(",")}, interests.sizes=${interests.sizes().vec().get().mkString(",")}, featWithInterests.sizes=${featWithInterests.sizes().vec().get().mkString(",")}, flattened.sizes=${flattened.sizes().vec().get().mkString(",")}, totalInputDim=${totalInputDim}")
    } catch { case _: Throwable => () }

    tower.forward(flattened)
  }
}

/**
 * Interest Extractor using Self-Attention
 */
class InterestExtractor(
  embedDim: Int,
  numInterests: Int,
  numHeads: Int,
  dropout: Float,
  device: String = DeviceSupport.backend
) extends Module {

  private val queryProj = new LinearImpl(embedDim, embedDim)
  private val keyProj = new LinearImpl(embedDim, embedDim)
  private val valueProj = new LinearImpl(embedDim, embedDim)

  private val interestQuery = new LinearImpl(embedDim, numInterests * embedDim)

  queryProj.to(new Device(device),false)
  keyProj.to(new Device(device),false)
  valueProj.to(new Device(device),false)
  interestQuery.to(new Device(device),false)

  register_module("queryProj", queryProj)
  register_module("keyProj", keyProj)
  register_module("valueProj", valueProj)
  register_module("interestQuery", interestQuery)

  private val dropoutLayer = new DropoutImpl(dropout)

  def forward(seqEmb: Tensor): Tensor = {
    // seqEmb: (batch, seq_len, embed_dim)
    val batchSize = seqEmb.size(0)
    val seqLen = seqEmb.size(1)

    // Self-attention
    val q = queryProj.forward(seqEmb)
    val k = keyProj.forward(seqEmb)
    val v = valueProj.forward(seqEmb)

    val scale = scala.math.sqrt(embedDim).toFloat
    val invScale = new Scalar(1.0f / scale)
    val scores = torch.matmul(q, k.transpose(1, 2)).mul(invScale)
    val attn = dropoutLayer.forward(scores.softmax(-1))
    val attended = torch.matmul(attn, v)

    // Generate interest queries
    val interestQ = interestQuery.forward(attended.mean(1)) // (batch, num_interests * embed_dim)
    val interestQReshaped = interestQ.view(batchSize, numInterests, embedDim)

    // Cross-attention between interest queries and attended sequence
    val interestEmb = torch.matmul(interestQReshaped, attended.transpose(1, 2)).softmax(-1)
    val interests = torch.matmul(interestEmb, attended) // (batch, num_interests, embed_dim)

    interests
  }
}
