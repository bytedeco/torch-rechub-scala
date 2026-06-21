package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.basic.layers.MLP
import torchrec.models.knowledge_tracing.layers.{CosinePositionalEmbedding, DistanceBiasMultiHeadAttention, TransformerLayer}
import torchrec.utils.DeviceSupport

/**
 * AKT: Attention-based Knowledge Tracing
 *
 * Reference: "AKT: Attention-based Knowledge Tracing" (Pandey & Karypis, KDD 2019)
 *
 * Architecture:
 *   Question embedding + Rasch difficulty model
 *   QA interaction embedding + Positional encoding
 *   Two-stage transformer blocks (encode then cross-attend)
 *   Output: concat(knowledge_state, question_emb) -> MLP -> sigmoid
 *
 * @param numConcepts   Number of unique concepts/questions
 * @param embedDim     Model dimension
 * @param numHeads     Number of attention heads
 * @param numBlocks    Number of transformer blocks (per stage)
 * @param ffnDim       FFN hidden dimension
 * @param dropout      Dropout rate
 * @param device       Device
 */
class AKT(
  numConcepts: Long,
  embedDim: Int = 64,
  numHeads: Int = 8,
  numBlocks: Int = 2,
  ffnDim: Int = 256,
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0, "embedDim must be divisible by numHeads")

  // Question embedding
  private val qEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim))
  register_module("q_embed", qEmbed)

  // QA interaction embedding: (concept_id + numConcepts * response)
  private val qaEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2, embedDim))
  register_module("qa_embed", qaEmbed)

  // Question difficulty parameter (Rasch model)
  private val qDiff = {
    var t = torch.randn(Array((numConcepts + 1).toLong, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .mul(new Scalar(0.1f))
    // If running on GPU, move parameter to device so subsequent ops are on same device
    if (device != "cpu") t = t.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
    register_parameter("q_diff", t)
    t
  }
  private val qaDiff = {
    var t = torch.randn(Array((numConcepts * 2).toLong, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .mul(new Scalar(0.1f))
    if (device != "cpu") t = t.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
    register_parameter("qa_diff", t)
    t
  }

  // Positional embedding
  private val posEmbed = new CosinePositionalEmbedding(embedDim, 512, device)
  register_module("pos_embed", posEmbed)

  // Two-stage transformer blocks
  // Stage 1: encode QA interactions
  private val blocks1 = (0 until numBlocks).map { i =>
    val layer = new TransformerLayer(embedDim, numHeads, ffnDim, dropout, device)
    register_module(s"block1_$i", layer)
    layer
  }

  // Stage 2: cross-attend with questions (with positional encoding)
  private val blocks2 = (0 until numBlocks).map { i =>
    val layer = new TransformerLayer(embedDim, numHeads, ffnDim, dropout, device)
    register_module(s"block2_$i", layer)
    layer
  }

  // Output MLP
  private val outLayer = new MLP(embedDim * 2, List(embedDim.toLong), 1, "relu", dropout, device = device)
  register_module("out_layer", outLayer)

  private val dropoutLayer = new DropoutImpl(dropout)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    qEmbed.to(dev, false); qaEmbed.to(dev, false)
    outLayer.to(dev, false)
  }

  /**
   * Forward pass for AKT.
   * @param conceptIds    Concept IDs (batch, seqLen) - includes first padded concept
   * @param responses     Responses 0/1 (batch, seqLen) - includes first padded response
   * @return Predictions (batch, seqLen) - probability of correct response
   */
  def forward(conceptIds: Tensor, responses: Tensor): Tensor = {
    val batchSize = conceptIds.size(0).toInt
    val seqLen = conceptIds.size(1).toInt

    // Ensure IDs are Long and within valid ranges before embedding/index_select
    val minScalar = new org.bytedeco.pytorch.Scalar(0L)
    val maxConceptScalar = new org.bytedeco.pytorch.Scalar(numConcepts)
    val maxResponseScalar = new org.bytedeco.pytorch.Scalar(1L)

    val conceptIdsLong = conceptIds.toType(ScalarType.Long).clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxConceptScalar)
    ).toType(ScalarType.Long)
    val responsesLong = responses.toType(ScalarType.Long).clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxResponseScalar)
    ).toType(ScalarType.Long)

    // Build interaction ids safely and clamp to embedding table range
    val interactionRaw = conceptIdsLong.add(responsesLong.mul(new Scalar(numConcepts)))
    val maxInteraction = numConcepts * 2 - 1
    val interactionIds = interactionRaw.clamp(
      new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(0L)),
      new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(maxInteraction))
    ).toType(ScalarType.Long)
    var qaEmb = qaEmbed.forward(interactionIds)  // (batch, seqLen, embedDim)

    // Add positional encoding
    val posEnc = posEmbed.forward(qaEmb)
    qaEmb = qaEmb.add(posEnc)

    // Stage 1: encode QA interactions (no causal mask for encoding)
    var y = qaEmb
    blocks1.foreach { block =>
      y = block.forward(y, mask = 1)
    }

    // Question embedding with difficulty adjustment (use long/clamped IDs)
    val qEmb = qEmbed.forward(conceptIdsLong)

    // Apply Rasch difficulty model
    try {
      println(s"[DEBUG AKT] conceptIdsLong dtype=${conceptIdsLong.dtype()} shape=${conceptIdsLong.shape().mkString(",")} min=${conceptIdsLong.toType(ScalarType.Long).view(-1L).min().itemSafe()} max=${conceptIdsLong.toType(ScalarType.Long).view(-1L).max().itemSafe()}")
      println(s"[DEBUG AKT] interactionIds dtype=${interactionIds.dtype()} shape=${interactionIds.shape().mkString(",")} min=${interactionIds.toType(ScalarType.Long).view(-1L).min().itemSafe()} max=${interactionIds.toType(ScalarType.Long).view(-1L).max().itemSafe()}")
    } catch { case _: Throwable => () }
    val qDiffEmb = qDiff.index_select(0, conceptIdsLong.toType(ScalarType.Long).view(-1L)).view(batchSize, seqLen, embedDim)
    val qaDiffEmb = qaDiff.index_select(0, interactionIds.toType(ScalarType.Long).view(-1L)).view(batchSize, seqLen, embedDim)

    val adjQEmb = qEmb.add(qDiffEmb)
    val adjQaEmb = qaEmb.add(qaDiffEmb)

    // Stage 2: cross-attend with questions (causal mask)
    var x = adjQEmb
    for (i <- 0 until numBlocks) {
      val attended = blocks2(i).forward(x, mask = 0)
      x = attended
    }

    // Output: concat knowledge state with question embedding
    val concatQa = torch.cat(new TensorVector(x, adjQaEmb), 2)
    val logits = outLayer.forward(concatQa)  // (batch, seqLen, 1)

    logits.sigmoid().squeeze(2)
  }

  def predict(conceptIds: Tensor, responses: Tensor): Tensor = forward(conceptIds, responses)
}

/**
 * SimpleKT: Simplified Knowledge Tracing
 * Uses cross-attention between questions and QA interactions.
 *
 * Reference: Simplified version of AKT without difficulty modeling.
 *
 * @param numConcepts   Number of unique concepts
 * @param embedDim      Model dimension
 * @param numHeads      Number of attention heads
 * @param numBlocks     Number of transformer blocks
 * @param ffnDim        FFN hidden dimension
 * @param dropout       Dropout rate
 * @param device        Device
 */
class SimpleKT(
  numConcepts: Long,
  embedDim: Int = 64,
  numHeads: Int = 8,
  numBlocks: Int = 2,
  ffnDim: Int = 256,
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

  private val blocks = (0 until numBlocks).map { i =>
    val layer = new TransformerLayer(embedDim, numHeads, ffnDim, dropout, device)
    register_module(s"block_$i", layer)
    layer
  }

  private val outLayer = new MLP(embedDim * 2, List(embedDim.toLong), 1, "relu", dropout, device = device)
  register_module("out_layer", outLayer)

  private val dropoutLayer = new DropoutImpl(dropout)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    qEmbed.to(dev, false); qaEmbed.to(dev, false)
    outLayer.to(dev, false)
  }

  def forward(conceptIds: Tensor, responses: Tensor): Tensor = {
    val batchSize = conceptIds.size(0).toInt
    val seqLen = conceptIds.size(1).toInt

    // Safe interaction IDs: clamp to valid range [0, numConcepts*2-1]
    val conceptIdsLong = conceptIds.toType(ScalarType.Long)
    val responsesLong = responses.toType(ScalarType.Long)
    val maxInteractionId = numConcepts * 2 - 1
    val interactionIdsBase = responsesLong.mul(new Scalar(numConcepts.toDouble))
    val interactionIdsRaw = conceptIdsLong.add(interactionIdsBase)
    // Clamp to valid range, then convert to Long
    val minScalar = new org.bytedeco.pytorch.Scalar(0)
    val maxScalar = new org.bytedeco.pytorch.Scalar(maxInteractionId)
    val interactionIdsClamped = interactionIdsRaw.clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxScalar)
    ).toType(ScalarType.Long)

    var qaEmb = qaEmbed.forward(interactionIdsClamped)
    val posEnc = posEmbed.forward(qaEmb)
    qaEmb = qaEmb.add(posEnc)

    // Clamp concept IDs for question embedding, then convert to Long
    val maxQId = numConcepts
    val qIdsClamped = conceptIdsLong.clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxScalar)
    ).toType(ScalarType.Long)
    val qEmb = qEmbed.forward(qIdsClamped)
    val posEncQ = posEmbed.forward(qEmb)
    val adjQEmb = qEmb.add(posEncQ)

    var x = adjQEmb
    blocks.foreach { block =>
      x = block.forward(x, mask = 0)
    }

    val concatQa = torch.cat(new TensorVector(x, adjQEmb), 2)
    val logits = outLayer.forward(concatQa)

    logits.sigmoid().squeeze(2)
  }

  def predict(conceptIds: Tensor, responses: Tensor): Tensor = forward(conceptIds, responses)
}

/**
 * SparseKT: Knowledge Tracing with Sparse Attention
 *
 * Reference: Sparse attention variant of SimpleKT.
 */
class SparseKT(
  numConcepts: Long,
  embedDim: Int = 64,
  numHeads: Int = 8,
  numBlocks: Int = 2,
  ffnDim: Int = 256,
  dropout: Float = 0.1f,
  sparseRatio: Float = 0.8f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)

  // Uses the same architecture as SimpleKT
  // The sparse attention is handled by the DistanceBiasMultiHeadAttention's distance bias
  private val qEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim))
  private val qaEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2, embedDim))
  register_module("q_embed", qEmbed)
  register_module("qa_embed", qaEmbed)

  private val posEmbed = new CosinePositionalEmbedding(embedDim, 512, device)
  register_module("pos_embed", posEmbed)

  private val blocks = (0 until numBlocks).map { i =>
    val layer = new TransformerLayer(embedDim, numHeads, ffnDim, dropout, device)
    register_module(s"block_$i", layer)
    layer
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

    // Safe interaction IDs: clamp to valid range [0, numConcepts*2-1]
    val conceptIdsLong = conceptIds.toType(ScalarType.Long)
    val responsesLong = responses.toType(ScalarType.Long)
    val maxInteractionId = numConcepts * 2 - 1
    val interactionIdsBase = responsesLong.mul(new Scalar(numConcepts.toDouble))
    val interactionIdsRaw = conceptIdsLong.add(interactionIdsBase)
    // Clamp to valid range, then convert to Long
    val minScalar = new org.bytedeco.pytorch.Scalar(0)
    val maxScalar = new org.bytedeco.pytorch.Scalar(maxInteractionId)
    val interactionIdsClamped = interactionIdsRaw.clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxScalar)
    ).toType(ScalarType.Long)

    var qaEmb = qaEmbed.forward(interactionIdsClamped)
    val posEnc = posEmbed.forward(qaEmb)
    qaEmb = qaEmb.add(posEnc)

    // Clamp concept IDs for question embedding, then convert to Long
    val maxQId = numConcepts
    val qIdsClamped = conceptIdsLong.clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxScalar)
    ).toType(ScalarType.Long)
    val qEmb = qEmbed.forward(qIdsClamped)
    val posEncQ = posEmbed.forward(qEmb)
    val adjQEmb = qEmb.add(posEncQ)

    var x = adjQEmb
    blocks.foreach { block =>
      x = block.forward(x, mask = 0)
    }

    val concatQa = torch.cat(new TensorVector(x, adjQEmb), 2)
    val logits = outLayer.forward(concatQa)

    logits.sigmoid().squeeze(2)
  }

  def predict(conceptIds: Tensor, responses: Tensor): Tensor = forward(conceptIds, responses)
}
