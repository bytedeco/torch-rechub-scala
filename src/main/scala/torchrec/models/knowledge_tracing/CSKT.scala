package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.basic.layers.MLP
import torchrec.models.knowledge_tracing.layers.{ConeAttention, CosinePositionalEmbedding}
import torchrec.utils.DeviceSupport

/**
 * CSKT: Cone Shape Knowledge Tracing
 *
 * Reference: "CSKT: Cone Shape Knowledge Tracing" (Xu et al.)
 * A transformer-based KT with cone-shaped attention geometry.
 *
 * @param numConcepts  Number of unique concepts
 * @param embedDim      Model dimension
 * @param numHeads      Number of attention heads
 * @param numBlocks     Number of transformer blocks
 * @param r             Cone radius parameter
 * @param gamma         Cone scale parameter
 * @param dropout       Dropout rate
 * @param device        Device
 */
class CSKT(
  numConcepts: Long,
  embedDim: Int = 64,
  numHeads: Int = 8,
  numBlocks: Int = 2,
  r: Float = 1.0f,
  gamma: Float = 1.0f,
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)

  private val qEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim))
  private val qaEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2, embedDim))
  register_module("q_embed", qEmbed)
  register_module("qa_embed", qaEmbed)

  private val posEmbed = new CosinePositionalEmbedding(embedDim, 512, device)
  register_module("pos_embed", posEmbed)

  // Helper to create normalized_shape LongVector: [d]
  private def layerNormShape(d: Int) = { val v = new LongVector(1); v.put(0, d.toLong); v }

  // Cone attention layers
  private val coneLayers = (0 until numBlocks).map { i =>
    val coneAttn = new ConeAttention(embedDim, numHeads, r, gamma, dropout, device)
    val ffn1 = new LinearImpl(embedDim, embedDim * 4)
    val ffn2 = new LinearImpl(embedDim * 4, embedDim)
    val ln1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
    val ln2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
    val drop = new DropoutImpl(dropout)

    register_module(s"cone_attn_$i", coneAttn)
    register_module(s"ffn1_$i", ffn1)
    register_module(s"ffn2_$i", ffn2)
    register_module(s"ln1_$i", ln1)
    register_module(s"ln2_$i", ln2)

    (coneAttn, ffn1, ffn2, ln1, ln2, drop)
  }

  private val outLayer = new MLP(embedDim * 2, List(embedDim.toLong), 1, "relu", dropout, device = device)
  register_module("out_layer", outLayer)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    qEmbed.to(dev, false); qaEmbed.to(dev, false)
    outLayer.to(dev, false)
  }

  def forward(conceptIds: Tensor, responses: Tensor): Tensor = {
    val batchSize = conceptIds.size(0).toInt
    val seqLen = conceptIds.size(1).toInt

    // Interaction IDs: concept * 2 + response (0=incorrect, 1=correct for each concept)
    val interactionIds = conceptIds.mul(new Scalar(2.0)).add(responses).toType(ScalarType.Long)
    var qaEmb = qaEmbed.forward(interactionIds)
    val posEnc = posEmbed.forward(qaEmb)
    qaEmb = qaEmb.add(posEnc)

    val qEmb = qEmbed.forward(conceptIds)
    val posEncQ = posEmbed.forward(qEmb)
    val adjQEmb = qEmb.add(posEncQ)

    // Cone attention blocks
    var x = adjQEmb
    coneLayers.foreach { case (coneAttn, ffn1, ffn2, ln1, ln2, drop) =>
      val attended = coneAttn.forward(x, x, mask = 0)
      val withRes1 = x.add(drop.forward(attended))
      val normed1 = ln1.forward(withRes1)

      val ffnOut = drop.forward(ffn2.forward(torch.relu(ffn1.forward(normed1))))
      x = ln2.forward(normed1.add(ffnOut))
    }

    val concatQa = torch.cat(new TensorVector(x, adjQEmb), 2)
    val logits = outLayer.forward(concatQa)

    logits.sigmoid().squeeze(2)
  }

  def predict(conceptIds: Tensor, responses: Tensor): Tensor = forward(conceptIds, responses)
}
