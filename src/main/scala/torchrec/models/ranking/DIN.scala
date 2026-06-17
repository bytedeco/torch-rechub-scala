package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

//import torchrec.toTensorVector
import torchrec.Implicits._

/**
 * Deep Interest Network with Attention
 * Reference: Alibaba, KDD 2018
 */
class DIN(
  features: List[Feature],
  sequenceFeatures: List[SequenceFeature],
  embedDim: Int = 8,
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  attentionUnits: Int = 64,
  device: String = DeviceSupport.backend
) extends Module {

  // Embedding layers
  private val featureEmbedding = new EmbeddingLayer(features, embedDim, device)
  register_module("featureEmbedding", featureEmbedding)

  private val sequenceEmbedding = new EmbeddingLayer(sequenceFeatures, embedDim, device)
  register_module("sequenceEmbedding", sequenceEmbedding)

  // Attention network
  private val attentionNet = new AttentionNet(embedDim, attentionUnits, embedDim, device)
  register_module("attentionNet", attentionNet)

  // MLP
  private val totalDim = Features.calcSparseDim(features) +
                          sequenceFeatures.map(_.embedDim).sum
  private val mlp = new MLP(totalDim, mlpDims.map(_.toLong), 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  def forward(
    sparseFeats: Map[String, Tensor],
    sequenceFeats: Map[String, Tensor],
    targetIdx: Tensor  // (batch, 1) - the item to attend to
  ): Tensor = {
    // Get feature embeddings
    val featEmb = featureEmbedding.forward(sparseFeats)

    // Get sequence embeddings and apply attention
    val seqEmbs = sequenceFeatures.map { seqFeat =>
      val seqEmb = sequenceEmbedding.getEmbedding(seqFeat.name, sequenceFeats(seqFeat.name))
      // seqEmb: (batch, seq_len, embed_dim)

      // Expand target to same sequence length
      val targetEmb = sequenceEmbedding.getEmbedding(seqFeat.name, targetIdx.toType(ScalarType.Long))
      // targetEmb: (batch, 1, embed_dim)
      val targetExpanded = targetEmb.unsqueeze(1).repeat(1, seqEmb.size(1), 1)

      // Attention
      val attended = attentionNet.forward(seqEmb, targetExpanded)
      // attended: (batch, embed_dim)
      attended
    }

    val combined = if (seqEmbs.nonEmpty) {
      torch.cat((featEmb +: seqEmbs).toTensorVector,  1)
    } else {
      featEmb
    }

    val logits = mlp.forward(combined)
    logits
  }
}

/**
 * Attention network for DIN
 */
class AttentionNet(
  queryDim: Int,
  hiddenUnits: Int,
  outputDim: Int,
  device: String = DeviceSupport.backend
) extends Module {

  private val queryProj = new LinearImpl(queryDim * 2, hiddenUnits)
  private val keyProj = new LinearImpl(queryDim * 2, hiddenUnits)
  private val valueProj = new LinearImpl(queryDim, outputDim)
  private val attentionNet = new MLP(hiddenUnits, List(1L), 1L, "relu", 0f, false, false, false, device)

  register_module("queryProj", queryProj)
  register_module("keyProj", keyProj)
  register_module("valueProj", valueProj)
  register_module("attentionNet", attentionNet)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    queryProj.to(dev, false)
    keyProj.to(dev, false)
    valueProj.to(dev, false)
  }

  def forward(sequence: Tensor, target: Tensor): Tensor = {
    // sequence: (batch, seq_len, embed)
    // target: (batch, seq_len, embed)
    val batchSize = sequence.size(0)
    val seqLen = sequence.size(1)

    // Concat sequence and target along dim 2
    val combined = torch.cat(new TensorVector(sequence, target), 2) // (batch, seq_len, embed*2)

    // Project to hidden dim
    val query = queryProj.forward(combined)
    val key = keyProj.forward(combined)
    val values = valueProj.forward(sequence)

    // Dot product attention
    val scores = query.mul(key).sum(2).unsqueeze(2).div(new Scalar(scala.math.sqrt(query.size(2).toDouble).toFloat))
    val attnWeights = scores.softmax(1)

    // Weighted sum
    val attended = values.mul(attnWeights).sum(1)
    attended
  }
}

/**
 * Dice activation
 */
class DiceActivation extends Module {
  private var P = 0.0f
  private val eps = 1e-8f

  def forward(x: Tensor): Tensor = {
    val p = x.sigmoid()
    p.mul(x).add(p.neg().add(torch.ones_like(p)).mul(x))
  }
}
