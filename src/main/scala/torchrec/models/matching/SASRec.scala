package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import torchrec.Implicits._

/**
 * SASRec - Self-Attentive Sequential Recommender
 * Reference: "Self-Attentive Sequential Recommendation" (Kang & McAuley, 2018)
 *
 * Implements item embedding + learned positional embedding + multi-head
 * scaled dot-product self-attention blocks (with residual + feed-forward),
 * then pools the contextualized sequence representation to a single vector
 * used for a binary relevance score.
 */
class SASRec(
  sequenceFeatures: List[Feature],
  embedDim: Int = 8,
  numHeads: Int = 2,
  numLayers: Int = 2,
  ffnDim: Int = 128,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")
  require(embedDim % numHeads == 0, s"embedDim ($embedDim) must be divisible by numHeads ($numHeads)")

  private val targetDevice = new Device(device)

  // The (first) sequence feature drives the item vocabulary / sequence length.
  private val seqFeature: SequenceFeature = sequenceFeatures.head match {
    case sf: SequenceFeature => sf
    case other => throw new IllegalArgumentException(
      s"SASRec expects a SequenceFeature, got: ${other.getClass.getSimpleName}")
  }

  /** Name of the sequence feature consumed by [[forward]] (configurable, not hard-coded). */
  val seqFeatureName: String = seqFeature.name

  private val vocabSize: Long = seqFeature.vocabSize
  private val maxLen: Int = seqFeature.maxLen
  private val headDim: Int = embedDim / numHeads
  private val scale: Double = 1.0 / math.sqrt(headDim.toDouble)

  // Item embedding table (padding index 0).
  private val itemEmbedding = {
    val opts = new EmbeddingOptions(vocabSize, embedDim)
    opts.padding_idx().put(0L)
    val emb = new EmbeddingImpl(opts)
    emb.to(targetDevice, false)
    emb
  }
  register_module("item_embedding", itemEmbedding)

  // Learned positional embedding.
  private val positionEmbedding = {
    val emb = new EmbeddingImpl(new EmbeddingOptions(maxLen.toLong, embedDim))
    emb.to(targetDevice, false)
    emb
  }
  register_module("position_embedding", positionEmbedding)

  private def mkLinear(name: String, inDim: Int = embedDim, outDim: Int = embedDim): LinearImpl = {
    val lin = new LinearImpl(inDim.toLong, outDim.toLong)
    lin.to(targetDevice, false)
    register_module(name, lin)
    lin
  }

  // Per-layer Q/K/V/out projections and feed-forward networks.
  private val qProj = Array.tabulate(numLayers)(i => mkLinear(s"q_proj_$i"))
  private val kProj = Array.tabulate(numLayers)(i => mkLinear(s"k_proj_$i"))
  private val vProj = Array.tabulate(numLayers)(i => mkLinear(s"v_proj_$i"))
  private val oProj = Array.tabulate(numLayers)(i => mkLinear(s"o_proj_$i"))
  private val ffn1 = Array.tabulate(numLayers)(i => mkLinear(s"ffn1_$i", embedDim, ffnDim))
  private val ffn2 = Array.tabulate(numLayers)(i => mkLinear(s"ffn2_$i", ffnDim, embedDim))

  private val output = mkLinear("output", embedDim, 1)

  /**
   * @param sequence Long tensor of item ids, shape [batch, seqLen].
   * @return logits of shape [batch, 1].
   */
  def forward(sequence: Tensor): Tensor = {
    val seq = sequence.toType(ScalarType.Long).to(targetDevice, ScalarType.Long)
    val batch = seq.size(0)
    val len = seq.size(1)

    // Item embeddings: [batch, len, embedDim]
    val itemEmb = itemEmbedding.forward(seq)

    // Positional embeddings broadcast over batch.
    val posIds = torch.arange(new Scalar(len), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
      .to(targetDevice, ScalarType.Long)
    val posEmb = positionEmbedding.forward(posIds).unsqueeze(0L) // [1, len, embedDim]

    var hidden = itemEmb.add(posEmb)

    // Padding mask: positions equal to padding idx (0) are masked out. [batch, len]
    val padScalar = torch.full(Array(1L), new Scalar(0L)).to(targetDevice, ScalarType.Long)
    val keyMask = seq.ne(padScalar).toType(ScalarType.Float) // 1 for real tokens, 0 for pad

    var layer = 0
    while (layer < numLayers) {
      hidden = attentionBlock(hidden, keyMask, batch, len, layer)
      layer += 1
    }

    // Masked mean pooling over the sequence dimension -> [batch, embedDim]
    val maskExpanded = keyMask.unsqueeze(2L) // [batch, len, 1]
    val summed = hidden.mul(maskExpanded).sum(1L) // [batch, embedDim]
    val counts = keyMask.sum(1L).clamp_min(new Scalar(1.0)).unsqueeze(1L) // [batch, 1]
    val pooled = summed.div(counts)

    output.forward(pooled) // [batch, 1]
  }

  private def attentionBlock(input: Tensor, keyMask: Tensor, batch: Long, len: Long, layer: Int): Tensor = {
    // Linear projections: [batch, len, embedDim]
    val q = qProj(layer).forward(input)
    val k = kProj(layer).forward(input)
    val v = vProj(layer).forward(input)

    // Reshape to [batch, numHeads, len, headDim]
    val qh = reshapeHeads(q, batch, len)
    val kh = reshapeHeads(k, batch, len)
    val vh = reshapeHeads(v, batch, len)

    // Scores: [batch, numHeads, len, len]
    val scores = torch.matmul(qh, kh.transpose(-2L, -1L)).mul(new Scalar(scale))

    // Apply key padding mask: mask shape [batch, 1, 1, len]
    val maskBroadcast = keyMask.reshape(batch, 1L, 1L, len)
    val negInf = torch.full(Array(1L), new Scalar(-1e9)).to(input.device(), ScalarType.Float)
    val maskedScores = scores.mul(maskBroadcast).add(
      maskBroadcast.mul(new Scalar(-1.0)).add(new Scalar(1.0)).mul(negInf)
    )

    val attn = torch.softmax(maskedScores, -1L)
    val context = torch.matmul(attn, vh) // [batch, numHeads, len, headDim]

    // Merge heads back: [batch, len, embedDim]
    val merged = context.transpose(1L, 2L).contiguous().reshape(batch, len, embedDim.toLong)
    val attnOut = oProj(layer).forward(merged)

    // Residual + feed-forward with residual.
    val res1 = input.add(attnOut)
    val ff = ffn2(layer).forward(torch.relu(ffn1(layer).forward(res1)))
    res1.add(ff)
  }

  private def reshapeHeads(x: Tensor, batch: Long, len: Long): Tensor = {
    // [batch, len, embedDim] -> [batch, numHeads, len, headDim]
    x.reshape(batch, len, numHeads.toLong, headDim.toLong).transpose(1L, 2L)
  }
}


