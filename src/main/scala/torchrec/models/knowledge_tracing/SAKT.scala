package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.models.knowledge_tracing.layers.{CosinePositionalEmbedding, MultiHeadAttention}
import torchrec.utils.DeviceSupport

/**
 * SAKT: Self-Attentive Knowledge Tracing
 *
 * Reference: "SAKT: Self-Attentive Knowledge Tracing" (Choi et al., KDD 2020)
 *
 * Architecture:
 *   Exercise embedding + Interaction embedding + Positional encoding
 *   -> Multi-head self-attention blocks
 *   -> Prediction layer
 *
 * @param numConcepts  Number of unique concepts
 * @param embedDim     Embedding dimension
 * @param numHeads     Number of attention heads
 * @param numBlocks    Number of transformer blocks
 * @param dropout      Dropout rate
 * @param device       Device
 */
class SAKT(
  numConcepts: Long,
  embedDim: Int = 64,
  numHeads: Int = 8,
  numBlocks: Int = 2,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)

  // Interaction embedding: concept_id + num_concepts * response
  private val interactionEmb = new EmbeddingImpl(
    new EmbeddingOptions(numConcepts * 2, embedDim)
  )
  register_module("interaction_emb", interactionEmb)

  // Exercise embedding
  private val exerciseEmb = new EmbeddingImpl(
    new EmbeddingOptions(numConcepts + 1, embedDim)
  )
  register_module("exercise_emb", exerciseEmb)

  // Positional embedding
  private val posEmb = new CosinePositionalEmbedding(embedDim, 512, device)
  register_module("pos_emb", posEmb)

  // Helper to create normalized_shape LongVector: [embedDim]
  private def layerNormShape(d: Int) = { val v = new LongVector(1); v.put(0, d.toLong); v }

  // Transformer blocks
  private val blocks = (0 until numBlocks).map { i =>
    val attn = new MultiHeadAttention(embedDim, numHeads, dropout, device)
    val ffn1 = new LinearImpl(embedDim, embedDim * 4)
    val ffn2 = new LinearImpl(embedDim * 4, embedDim)
    val ln1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
    val ln2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
    val drop = new DropoutImpl(dropout)

    register_module(s"attn_$i", attn)
    register_module(s"ffn1_$i", ffn1)
    register_module(s"ffn2_$i", ffn2)
    register_module(s"ln1_$i", ln1)
    register_module(s"ln2_$i", ln2)

    (attn, ffn1, ffn2, ln1, ln2, drop)
  }

  // Prediction layer
  private val predLayer = new LinearImpl(embedDim, 1)
  register_module("pred", predLayer)

  private val dropoutLayer = new DropoutImpl(dropout)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    interactionEmb.to(dev, false); exerciseEmb.to(dev, false)
    predLayer.to(dev, false)
  }

  def forward(
    conceptIds: Tensor,     // (batch, seqLen) - current concept
    responses: Tensor,     // (batch, seqLen) - current response
    targetConcept: Tensor   // (batch, seqLen) - shifted concept (query)
  ): Tensor = {
    val batchSize = conceptIds.size(0).toInt
    val seqLen = conceptIds.size(1).toInt

    // Create interaction IDs: concept * 2 + response (0=incorrect, 1=correct for each concept)
    val interactionIds = conceptIds.mul(new Scalar(2.0))
      .add(responses)
      .toType(ScalarType.Long)

    // Interaction embedding
    var xEmb = interactionEmb.forward(interactionIds)

    // Positional encoding
    val posEnc = posEmb.forward(xEmb)
    xEmb = xEmb.add(posEnc)

    // Exercise embedding for query
    val qryEmb = exerciseEmb.forward(targetConcept)
    val qryPos = posEmb.forward(qryEmb)
    val qryEnc = qryEmb.add(qryPos)

    // Transformer blocks: query=qryEnc, key/value=xEmb
    var x = qryEnc
    blocks.foreach { case (attn, ffn1, ffn2, ln1, ln2, drop) =>
      // Cross-attention: x attends to xEmb
      val attended = attn.forward(x, xEmb, xEmb)
      val withRes1 = x.add(drop.forward(attended))
      val normed1 = ln1.forward(withRes1)

      // Self-attention on x
      val selfAttn = attn.forward(normed1, normed1, normed1)
      val withRes2 = normed1.add(drop.forward(selfAttn))
      val normed2 = ln2.forward(withRes2)

      // FFN
      val ffnOut = drop.forward(ffn2.forward(torch.relu(ffn1.forward(normed2))))
      x = normed2.add(ffnOut)
    }

    // Prediction
    val logits = predLayer.forward(dropoutLayer.forward(x))
    logits.sigmoid().squeeze(2)
  }

  def predict(conceptIds: Tensor, responses: Tensor, targetConcept: Tensor): Tensor =
    forward(conceptIds, responses, targetConcept)
}

/**
 * SAKT with unified query: uses the interaction embedding as query.
 */
class SAKTUnified(
  numConcepts: Long,
  embedDim: Int = 64,
  numHeads: Int = 8,
  numBlocks: Int = 2,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)

  private val interactionEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2, embedDim))
  private val posEmb = new CosinePositionalEmbedding(embedDim, 512, device)
  private val predLayer = new LinearImpl(embedDim, 1)
  private val dropoutLayer = new DropoutImpl(dropout)

  register_module("interaction_emb", interactionEmb)
  register_module("pos_emb", posEmb)
  register_module("pred", predLayer)

  private def layerNormShape(d: Int) = { val v = new LongVector(1); v.put(0, d.toLong); v }

  private val blocks = (0 until numBlocks).map { i =>
    val attn = new MultiHeadAttention(embedDim, numHeads, dropout, device)
    val ffn1 = new LinearImpl(embedDim, embedDim * 4)
    val ffn2 = new LinearImpl(embedDim * 4, embedDim)
    val ln1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
    val ln2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
    val drop = new DropoutImpl(dropout)

    register_module(s"attn_$i", attn)
    register_module(s"ffn1_$i", ffn1)
    register_module(s"ffn2_$i", ffn2)
    register_module(s"ln1_$i", ln1)
    register_module(s"ln2_$i", ln2)

    (attn, ffn1, ffn2, ln1, ln2, drop)
  }

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    interactionEmb.to(dev, false); predLayer.to(dev, false)
  }

  def forward(conceptIds: Tensor, responses: Tensor): Tensor = {
    val batchSize = conceptIds.size(0).toInt
    val seqLen = conceptIds.size(1).toInt

    val interactionIds = conceptIds.mul(new Scalar(2.0))
      .add(responses)
      .toType(ScalarType.Long)
    var x = interactionEmb.forward(interactionIds)
    val posEnc = posEmb.forward(x)
    x = x.add(posEnc)

    blocks.foreach { case (attn, ffn1, ffn2, ln1, ln2, drop) =>
      val attended = attn.forward(x, x, x)
      val withRes1 = x.add(drop.forward(attended))
      val normed1 = ln1.forward(withRes1)
      val ffnOut = drop.forward(ffn2.forward(torch.relu(ffn1.forward(normed1))))
      x = ln2.forward(normed1.add(ffnOut))
    }

    val logits = predLayer.forward(dropoutLayer.forward(x))
    logits.sigmoid().squeeze(2)
  }

  def predict(conceptIds: Tensor, responses: Tensor): Tensor = forward(conceptIds, responses)
}
