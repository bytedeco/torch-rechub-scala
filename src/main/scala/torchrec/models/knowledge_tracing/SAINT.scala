package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.models.knowledge_tracing.layers.{LearnablePositionalEmbedding, SAINTDecoderBlock, SAINTEncoderBlock}
import torchrec.utils.DeviceSupport

/**
 * SAINT: Self-Attentive Interpretable Knowledge Tracing
 *
 * Reference: "SAINT: An Interpretable Self-Attentive Knowledge Tracing" (Shin et al., 2021)
 *
 * Architecture:
 *   Encoder: exercise + category + position -> Transformer encoder
 *   Decoder: response + position (with start token) -> Transformer decoder
 *   Output: sigmoid(decoder output)
 *
 * @param numExercises  Number of unique exercises/questions
 * @param numCategories  Number of categories/concepts
 * @param numResponses   Number of response types (typically 2)
 * @param embedDim      Embedding dimension
 * @param numHeads       Number of attention heads
 * @param numEncoderBlocks Number of encoder blocks
 * @param numDecoderBlocks Number of decoder blocks
 * @param ffnDim         FFN hidden dimension
 * @param dropout        Dropout rate
 * @param device         Device
 */
class SAINT(
  numExercises: Long,
  numCategories: Long,
  numResponses: Int = 2,
  embedDim: Int = 64,
  numHeads: Int = 8,
  numEncoderBlocks: Int = 3,
  numDecoderBlocks: Int = 3,
  ffnDim: Int = 256,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)

  // Exercise embedding
  private val exEmb = new EmbeddingImpl(new EmbeddingOptions(numExercises + 1, embedDim))
  register_module("ex_emb", exEmb)

  // Category/concept embedding
  private val catEmb = new EmbeddingImpl(new EmbeddingOptions(numCategories + 1, embedDim))
  register_module("cat_emb", catEmb)

  // Response embedding (0, 1, ... numResponses-1, start_token, pad_token)
  private val resEmb = new EmbeddingImpl(new EmbeddingOptions(numResponses + 2, embedDim))
  register_module("res_emb", resEmb)

  // Positional embedding
  private val posEmb = new LearnablePositionalEmbedding(512, embedDim, dropout, device)
  register_module("pos_emb", posEmb)

  // Encoder blocks
  private val encoderBlocks = (0 until numEncoderBlocks).map { i =>
    val block = new SAINTEncoderBlock(embedDim, numHeads, ffnDim, dropout, device)
    register_module(s"encoder_$i", block)
    block
  }

  // Decoder blocks
  private val decoderBlocks = (0 until numDecoderBlocks).map { i =>
    val block = new SAINTDecoderBlock(embedDim, numHeads, ffnDim, dropout, device)
    register_module(s"decoder_$i", block)
    block
  }

  // Output layer
  private val outLayer = new LinearImpl(embedDim, 1)
  register_module("out", outLayer)

  // Start token ID
  private val startTokenId = numResponses + 1  // After 0, 1 response values

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    exEmb.to(dev, false); catEmb.to(dev, false); resEmb.to(dev, false)
    outLayer.to(dev, false)
  }

  /**
   * Forward pass for SAINT.
   * @param exerciseIds  Exercise IDs (batch, seqLen)
   * @param categoryIds  Category/concept IDs (batch, seqLen)
   * @param responseIds  Response IDs 0/1 (batch, seqLen)
   * @return Predictions (batch, seqLen) - probability of correct response
   */
  def forward(exerciseIds: Tensor, categoryIds: Tensor, responseIds: Tensor): Tensor = {
    val batchSize = exerciseIds.size(0).toInt
    val seqLen = exerciseIds.size(1).toInt

    // Ensure IDs are Long and clamped to valid ranges to avoid index/select dtype or OOB issues
    val minScalar = new org.bytedeco.pytorch.Scalar(0)
    val maxExScalar = new org.bytedeco.pytorch.Scalar(numExercises.toDouble)
    val maxCatScalar = new org.bytedeco.pytorch.Scalar(numCategories.toDouble)
    val maxResScalar = new org.bytedeco.pytorch.Scalar(1)

    val exIds = exerciseIds.toType(ScalarType.Long).clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxExScalar)
    )
    val catIds = categoryIds.toType(ScalarType.Long).clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxCatScalar)
    )
    val resIds = responseIds.toType(ScalarType.Long).clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxResScalar)
    )

    // Debug prints for IDs
    // Ensure indices are on the correct device and of Long dtype
    val devObj = new org.bytedeco.pytorch.Device(device)
    var exIdsDev = exIds.to(devObj, ScalarType.Long)
    var catIdsDev = catIds.to(devObj, ScalarType.Long)
    var resIdsDev = resIds.to(devObj, ScalarType.Long)
    // Defensive clamp against actual embedding table sizes to avoid OOB
    try {
      val exNum = exEmb.weight().size(0)
      val catNum = catEmb.weight().size(0)
      val resNum = resEmb.weight().size(0)
      exIdsDev = exIdsDev.clamp(new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(0)), new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar((exNum - 1).toDouble)))
      catIdsDev = catIdsDev.clamp(new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(0)), new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar((catNum - 1).toDouble)))
      resIdsDev = resIdsDev.clamp(new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(0)), new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar((resNum - 1).toDouble)))
    } catch { case _: Throwable => () }
    try {
      println(s"[DEBUG SAINT] exIds dtype=${exIdsDev.dtype()} shape=${exIdsDev.shape().mkString(",")} min=${exIdsDev.view(-1L).min().itemSafe()} max=${exIdsDev.view(-1L).max().itemSafe()}")
      println(s"[DEBUG SAINT] catIds dtype=${catIdsDev.dtype()} shape=${catIdsDev.shape().mkString(",")} min=${catIdsDev.view(-1L).min().itemSafe()} max=${catIdsDev.view(-1L).max().itemSafe()}")
      println(s"[DEBUG SAINT] resIds dtype=${resIdsDev.dtype()} shape=${resIdsDev.shape().mkString(",")} min=${resIdsDev.view(-1L).min().itemSafe()} max=${resIdsDev.view(-1L).max().itemSafe()}")
    } catch { case _: Throwable => () }
    // Encoder: exercise + category + position
    // Defensive: ensure indices are Long dtype and contiguous right before embedding lookup
    val exIdsForEmb = exIdsDev.toType(ScalarType.Long).contiguous()
    val catIdsForEmb = catIdsDev.toType(ScalarType.Long).contiguous()
    val resIdsForEmb = resIdsDev.toType(ScalarType.Long).contiguous()
    val exEmbOut = exEmb.forward(exIdsForEmb)
    val catEmbOut = catEmb.forward(catIdsForEmb)
    val posEnc = posEmb.forward(seqLen.toLong)
    val posEx = posEnc.expand(batchSize.toLong, seqLen.toLong, embedDim.toLong)

    var enOut = exEmbOut.add(catEmbOut)
    encoderBlocks.foreach { block =>
      enOut = block.forward(enOut, catEmbOut, posEx)
    }

    // Decoder: prepend start token to responses
    val startTokens = torch.full(Array(batchSize.toLong, 1L), new Scalar(startTokenId.toDouble),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
    val startTokensDev = if (device != "cpu") startTokens.to(devObj, ScalarType.Long) else startTokens
    val paddedResponses = torch.cat(new TensorVector(startTokensDev, resIdsDev), 1)
    // Ensure indices are Long dtype on the correct device before embedding lookup
    // Ensure padded responses are Long dtype on correct device
    val paddedResponsesLong = paddedResponses.toType(ScalarType.Long).to(new org.bytedeco.pytorch.Device(device), ScalarType.Long).contiguous()

    val resEmbOut = resEmb.forward(paddedResponsesLong)
    val posDec = posEmb.forward((seqLen + 1).toLong)
    val posDecEx = posDec.expand(batchSize.toLong, (seqLen + 1).toLong, embedDim.toLong)

    var decOut = resEmbOut
    decoderBlocks.foreach { block =>
      decOut = block.forward(decOut, posDecEx, enOut)
    }

    // Predict from decoder output (skip the start token position)
    val decOutShifted = decOut.narrow(1, 1, seqLen)
    val logits = outLayer.forward(decOutShifted)

    logits.sigmoid().squeeze(2)
  }

  def predict(exerciseIds: Tensor, categoryIds: Tensor, responseIds: Tensor): Tensor =
    forward(exerciseIds, categoryIds, responseIds)
}
