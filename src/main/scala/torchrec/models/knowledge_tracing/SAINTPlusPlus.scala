package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.models.knowledge_tracing.layers.{LearnablePositionalEmbedding, SAINTDecoderBlock, SAINTEncoderBlock}
import torchrec.utils.DeviceSupport

/**
 * SAINT++: Self-Attentive Interpretable Knowledge Tracing (Enhanced)
 *
 * An enhanced version of SAINT with:
 * - Separate encoder for exercises and categories
 * - Enhanced decoder with more sophisticated cross-attention
 * - Deeper architecture with more transformer blocks
 *
 * Reference: "SAINT: An Interpretable Self-Attentive Knowledge Tracing" variants
 *
 * Architecture:
 *   Encoder: Exercise + Category + Position -> Transformer Encoder
 *   Decoder: Response + Position (with start token) -> Transformer Decoder (cross-attend to encoder)
 *   Output: sigmoid(decoder output)
 *
 * @param numExercises     Number of unique exercises/questions
 * @param numCategories     Number of categories/concepts
 * @param numResponses      Number of response types (typically 2)
 * @param embedDim         Embedding dimension
 * @param numHeads          Number of attention heads
 * @param numEncoderBlocks  Number of encoder blocks
 * @param numDecoderBlocks  Number of decoder blocks
 * @param ffnDim            FFN hidden dimension
 * @param dropout           Dropout rate
 * @param device            Device
 */
class SAINTPlusPlus(
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

  // Question embedding for decoder query
  private val questionEmb = new EmbeddingImpl(new EmbeddingOptions(numExercises * 2 + 1, embedDim))
  register_module("question_emb", questionEmb)

  // Positional embedding (learnable)
  private val posEmb = new LearnablePositionalEmbedding(512, embedDim, dropout, device)
  register_module("pos_emb", posEmb)

  // Encoder blocks (enhanced with self-attention on exercise+category)
  private val encoderBlocks = (0 until numEncoderBlocks).map { i =>
    val block = new SAINTEncoderBlockPlus(embedDim, numHeads, ffnDim, dropout, device)
    register_module(s"encoder_$i", block)
    block
  }

  // Decoder blocks (enhanced cross-attention)
  private val decoderBlocks = (0 until numDecoderBlocks).map { i =>
    val block = new SAINTDecoderBlockPlus(embedDim, numHeads, ffnDim, dropout, device)
    register_module(s"decoder_$i", block)
    block
  }

  // Output layer
  private val outLayer = new LinearImpl(embedDim, 1)
  register_module("out", outLayer)

  // Start token ID
  private val startTokenId = numResponses + 1

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    exEmb.to(dev, false); catEmb.to(dev, false); resEmb.to(dev, false)
    questionEmb.to(dev, false); outLayer.to(dev, false)
  }

  /**
   * Forward pass for SAINT++.
   * @param exerciseIds  Exercise IDs (batch, seqLen)
   * @param categoryIds  Category/concept IDs (batch, seqLen)
   * @param responseIds  Response IDs 0/1 (batch, seqLen)
   * @return Predictions (batch, seqLen) - probability of correct response
   */
  def forward(
    exerciseIds: Tensor,
    categoryIds: Tensor,
    responseIds: Tensor
  ): Tensor = {
    val batchSize = exerciseIds.size(0).toInt
    val seqLen = exerciseIds.size(1).toInt

    // Clamp IDs to valid range
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

    // Encoder: exercise + category + position
    // Defensive: ensure Long dtype and contiguous before embedding lookup
    val devObj = new org.bytedeco.pytorch.Device(device)
    var exIdsDev = exIds.to(devObj, ScalarType.Long)
    var catIdsDev = catIds.to(devObj, ScalarType.Long)
    val exIdsForEmb = exIdsDev.toType(ScalarType.Long).contiguous()
    val catIdsForEmb = catIdsDev.toType(ScalarType.Long).contiguous()
    val exEmbOut = exEmb.forward(exIdsForEmb)
    val catEmbOut = catEmb.forward(catIdsForEmb)
    val posEnc = posEmb.forward(seqLen.toLong)
    val posEx = posEnc.expand(batchSize.toLong, seqLen.toLong, embedDim.toLong)

    var enOut = exEmbOut.add(catEmbOut).add(posEx)
    encoderBlocks.foreach { block =>
      enOut = block.forward(enOut, catEmbOut, posEx)
    }

    // Decoder: prepend start token to responses
//    val devObj = new org.bytedeco.pytorch.Device(device)
    val startTokens = torch.full(Array(batchSize.toLong, 1L), new Scalar(startTokenId.toDouble),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
    val startTokensDev = if (device != "cpu") startTokens.to(devObj, ScalarType.Long) else startTokens
    var resIdsDev = resIds.to(devObj, ScalarType.Long)
    // Clamp against resEmb table size
    try { val resNum = resEmb.weight().size(0); resIdsDev = resIdsDev.clamp(new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(0)), new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar((resNum - 1).toDouble))) } catch { case _: Throwable => () }
    val paddedResponses = torch.cat(new TensorVector(startTokensDev, resIdsDev), 1)
    // Ensure indices are Long dtype on correct device before embedding lookup
    val paddedResponsesLong = paddedResponses.toType(ScalarType.Long).to(devObj, ScalarType.Long).contiguous()

    val resEmbOut = resEmb.forward(paddedResponsesLong)
    val posDec = posEmb.forward((seqLen + 1).toLong)
    val posDecEx = posDec.expand(batchSize.toLong, (seqLen + 1).toLong, embedDim.toLong)

    // Question embedding for decoder input
    // Map (exercise, response) -> combined id in range [0, 2*numExercises-1]
    // Build question ids from Long index tensors to avoid float arithmetic and rounding issues
    val exIdsLongForQuestion = exIds.toType(ScalarType.Long)
    val resIdsLongForQuestion = resIds.toType(ScalarType.Long)
    // For mapping, treat padding exercise id (which equals numExercises) specially.
    // Real exercises are in [0, numExercises-1]. We'll map pad -> last index in questionEmb (2*numExercises).
    val exIdsClampedForQuestion = try {
      // clamp to [0, numExercises-1] for real mapping
      exIdsLongForQuestion.clamp(new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(0)), new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar((numExercises - 1).toDouble)))
    } catch { case _: Throwable => exIdsLongForQuestion }

    // Build the combined id for real exercises
    val questionRawForReal = exIdsClampedForQuestion.mul(new Scalar(2)).add(resIdsLongForQuestion)

    // Now set pad positions (where original exIds == numExercises) to the reserved pad index (2 * numExercises)
    val padIndexForQuestion = (2 * numExercises).toLong
    val isPad = exIdsLongForQuestion.eq(new Scalar(numExercises.toDouble))
    // Create a tensor filled with padIndex
    val padIndexTensor = torch.full(questionRawForReal.shape(), new Scalar(padIndexForQuestion.toDouble), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long))).to(isPad.device(), ScalarType.Long)
    // Use torch.where to choose pad index for pad positions, else questionRawForReal
    val questionIdsLongUnclamped = torch.where(isPad, padIndexTensor, questionRawForReal)

    // Finally clamp against the actual questionEmb table size as an extra safety and cast to Long
    var questionIdsLong = try {
      val qNum = questionEmb.weight().size(0)
      questionIdsLongUnclamped.clamp(new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(0)), new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar((qNum - 1).toDouble))).toType(ScalarType.Long)
    } catch { case _: Throwable => questionIdsLongUnclamped.toType(ScalarType.Long) }
    val questionEmbOut = questionEmb.forward(questionIdsLong.contiguous())
    val posQ = posEmb.forward(seqLen.toLong)
    val questionEnc = questionEmbOut.add(posQ.expand(batchSize.toLong, seqLen.toLong, embedDim.toLong))

    // Prepend question embedding for decoder
    var zerosPad = torch.zeros(Array(batchSize.toLong, 1L, embedDim.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    if (device != "cpu") zerosPad = zerosPad.to(devObj, ScalarType.Float)
    val paddedQuestions = torch.cat(new TensorVector(zerosPad, questionEnc), 1)

    var decOut = resEmbOut.add(posDecEx)
    decoderBlocks.foreach { block =>
      decOut = block.forward(decOut, paddedQuestions, enOut)
    }

    // Predict from decoder output (skip the start token position)
    val decOutShifted = decOut.narrow(1, 1, seqLen)
    val logits = outLayer.forward(decOutShifted)

    logits.sigmoid().squeeze(2)
  }

  def predict(
    exerciseIds: Tensor,
    categoryIds: Tensor,
    responseIds: Tensor
  ): Tensor = forward(exerciseIds, categoryIds, responseIds)
}

/**
 * Enhanced SAINT Encoder Block with deeper architecture.
 */
class SAINTEncoderBlockPlus(
  embedDim: Int,
  numHeads: Int,
  ffnDim: Int = 256,
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  private val multiEn = new MultiHeadAttentionPlus(embedDim, numHeads, dropout, device)
  private val ffnEn1 = new LinearImpl(embedDim, ffnDim)
  private val ffnEn2 = new LinearImpl(ffnDim, embedDim)
  private def layerNormShape(d: Int) = { val v = new LongVector(1); v.put(0, d.toLong); v }
  private val ln1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
  private val ln2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
  private val dropoutLayer = new DropoutImpl(dropout)

  register_module("multi_en", multiEn)
  register_module("ffn_en1", ffnEn1)
  register_module("ffn_en2", ffnEn2)
  register_module("ln1", ln1)
  register_module("ln2", ln2)

  def forward(inEx: Tensor, inCat: Tensor, inPos: Tensor): Tensor = {
    val combined = inEx.add(inCat).add(inPos)
    val attended = multiEn.forward(combined, combined, combined)
    val withResidual1 = combined.add(dropoutLayer.forward(attended))
    val normed1 = ln1.forward(withResidual1)

    val ffnOut = dropoutLayer.forward(ffnEn2.forward(torch.relu(ffnEn1.forward(normed1))))
    ln2.forward(normed1.add(ffnOut))
  }
}

/**
 * Enhanced SAINT Decoder Block with deeper cross-attention.
 */
class SAINTDecoderBlockPlus(
  embedDim: Int,
  numHeads: Int,
  ffnDim: Int = 256,
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  // Cross attention: attend to encoder output
  private val multiDe1 = new MultiHeadAttentionPlus(embedDim, numHeads, dropout, device)
  // Self attention
  private val multiDe2 = new MultiHeadAttentionPlus(embedDim, numHeads, dropout, device)
  private val ffnDe1 = new LinearImpl(embedDim, ffnDim)
  private val ffnDe2 = new LinearImpl(ffnDim, embedDim)
  private def layerNormShape(d: Int) = { val v = new LongVector(1); v.put(0, d.toLong); v }
  private val ln1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
  private val ln2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
  private val ln3 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
  private val dropoutLayer = new DropoutImpl(dropout)

  register_module("multi_de1", multiDe1)
  register_module("multi_de2", multiDe2)
  register_module("ffn_de1", ffnDe1)
  register_module("ffn_de2", ffnDe2)
  register_module("ln1", ln1)
  register_module("ln2", ln2)
  register_module("ln3", ln3)

  def forward(inRes: Tensor, inQuestion: Tensor, enOut: Tensor): Tensor = {
    val combined = inRes

    // Cross attention on encoder output
    val crossAttn = multiDe1.forward(combined, enOut, enOut)
    val withResidual1 = combined.add(dropoutLayer.forward(crossAttn))
    val normed1 = ln1.forward(withResidual1)

    // Self attention
    val selfAttn = multiDe2.forward(normed1, normed1, normed1)
    val withResidual2 = normed1.add(dropoutLayer.forward(selfAttn))
    val normed2 = ln2.forward(withResidual2)

    // FFN
    val ffnOut = dropoutLayer.forward(ffnDe2.forward(torch.relu(ffnDe1.forward(normed2))))
    ln3.forward(normed2.add(ffnOut))
  }
}

/**
 * Enhanced Multi-head attention with residual connections.
 */
class MultiHeadAttentionPlus(
  embedDim: Int,
  numHeads: Int,
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)
  private val headDim = embedDim / numHeads

  private val qLinear = new LinearImpl(embedDim, embedDim)
  private val kLinear = new LinearImpl(embedDim, embedDim)
  private val vLinear = new LinearImpl(embedDim, embedDim)
  private val outLinear = new LinearImpl(embedDim, embedDim)
  private val dropoutLayer = new DropoutImpl(dropout)

  register_module("q_linear", qLinear)
  register_module("k_linear", kLinear)
  register_module("v_linear", vLinear)
  register_module("out_linear", outLinear)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    qLinear.to(dev, false); kLinear.to(dev, false)
    vLinear.to(dev, false); outLinear.to(dev, false)
  }

  def forward(q: Tensor, k: Tensor, v: Tensor, mask: Tensor = torch.empty()): Tensor = {
    val batchSize = q.size(0).toInt
    val seqLen = q.size(1).toInt
    val keySeqLen = k.size(1).toInt

    val qProj = qLinear.forward(q).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)
    val kProj = kLinear.forward(k).view(batchSize, keySeqLen, numHeads, headDim).transpose(1, 2)
    val vProj = vLinear.forward(v).view(batchSize, keySeqLen, numHeads, headDim).transpose(1, 2)

    val scale = new Scalar(scala.math.sqrt(headDim.toDouble).toFloat)
    var scores = torch.matmul(qProj, kProj.transpose(2, 3)).div(scale)

    if (mask != null && mask.numel() > 0) {
      scores = scores.add(mask)
    }

    val attnWeights = scores.softmax(-1)
    val attended = torch.matmul(dropoutLayer.forward(attnWeights), vProj)
    val reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim)
    outLinear.forward(reshaped)
  }
}
