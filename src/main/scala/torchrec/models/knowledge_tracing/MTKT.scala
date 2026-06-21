package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.basic.layers.MLP
import torchrec.models.knowledge_tracing.layers.{CosinePositionalEmbedding, TransformerLayer}
import torchrec.utils.DeviceSupport

/**
 * MTKT: Multi-Task Knowledge Tracing
 *
 * A knowledge tracing model that uses multi-task learning to jointly predict:
 * - Student responses (main task)
 * - Time-related features (auxiliary tasks)
 *
 * Uses a transformer-based architecture with separate encoders for:
 * - Question/skill embeddings
 * - Response embeddings
 * - Temporal aspect embeddings
 *
 * Architecture:
 *   Question Embedding + Response Embedding -> Transformer Encoder -> Gating -> MLP -> Prediction
 *
 * @param numConcepts   Number of unique concepts/questions
 * @param embedDim     Model dimension
 * @param numHeads     Number of attention heads
 * @param numBlocks    Number of transformer blocks
 * @param dropout      Dropout rate
 * @param device       Device
 */
class MTKT(
  numConcepts: Long,
  embedDim: Int = 64,
  numHeads: Int = 8,
  numBlocks: Int = 2,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)

  // Question/skill embedding
  private val qEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim))
  register_module("q_embed", qEmbed)

  // Response embedding
  private val rEmbed = new EmbeddingImpl(new EmbeddingOptions(2 + 1, embedDim))  // 0, 1, start_token
  register_module("r_embed", rEmbed)

  // Question-Response interaction embedding
  private val qaEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2, embedDim))
  register_module("qa_embed", qaEmbed)

  // Positional embedding
  private val posEmbed = new CosinePositionalEmbedding(embedDim, 512, device)
  register_module("pos_embed", posEmbed)

  // Question aspect transformer blocks
  private val questionBlocks = (0 until numBlocks).map { i =>
    val layer = new TransformerLayer(embedDim, numHeads, embedDim * 4, dropout, device)
    register_module(s"question_block_$i", layer)
    layer
  }

  // Response aspect transformer blocks
  private val responseBlocks = (0 until numBlocks).map { i =>
    val layer = new TransformerLayer(embedDim, numHeads, embedDim * 4, dropout, device)
    register_module(s"response_block_$i", layer)
    layer
  }

  // Gating mechanism for combining question and response aspects
  private val cWeight = new LinearImpl(embedDim, embedDim)
  register_module("c_weight", cWeight)
  private val tWeight = new LinearImpl(embedDim, embedDim)
  register_module("t_weight", tWeight)

  // Output MLP
  private val outMLP = new MLP(embedDim * 2, List(embedDim.toLong), 1, "relu", dropout, device = device)
  register_module("out_mlp", outMLP)

  // Dropout
  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropout", dropoutLayer)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    qEmbed.to(dev, false); rEmbed.to(dev, false); qaEmbed.to(dev, false)
    outMLP.to(dev, false)
  }

  /**
   * Forward pass for MTKT.
   * @param conceptIds  Concept IDs (batch, seqLen)
   * @param responses  Responses 0/1 (batch, seqLen)
   * @return Predictions (batch, seqLen) - probability of correct response
   */
  def forward(
    conceptIds: Tensor,
    responses: Tensor
  ): Tensor = {
    val batchSize = conceptIds.size(0).toInt
    val seqLen = conceptIds.size(1).toInt

    // Clamp IDs
    val minScalar = new org.bytedeco.pytorch.Scalar(0)
    val maxConceptScalar = new org.bytedeco.pytorch.Scalar(numConcepts.toDouble)
    val maxResponseScalar = new org.bytedeco.pytorch.Scalar(1)

    val cIdsLong = conceptIds.toType(ScalarType.Long).clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxConceptScalar)
    )
    val rLong = responses.toType(ScalarType.Long).clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxResponseScalar)
    )

    // Get embeddings (defensive: ensure Long dtype before embedding)
    val qEmb = qEmbed.forward(cIdsLong.toType(ScalarType.Long))  // (batch, seq, embedDim)
    val rEmb = rEmbed.forward(rLong.toType(ScalarType.Long))  // (batch, seq, embedDim)

    // QA interaction embedding
    val qaIds = cIdsLong.add(rLong.mul(new Scalar(numConcepts.toDouble)))
    val qaEmb = qaEmbed.forward(qaIds.toType(ScalarType.Long))  // (batch, seq, embedDim)

    // Add positional encoding
    val posEnc = posEmbed.forward(qEmb)
    val qWithPos = qEmb.add(posEnc)
    val rWithPos = rEmb.add(posEnc)

    // Question aspect encoding (self-attention on question embeddings)
    var qOut = qWithPos
    questionBlocks.foreach { block =>
      qOut = block.forward(qOut, mask = 1)
    }

    // Response aspect encoding (cross-attention with question)
    var rOut = rWithPos
    responseBlocks.foreach { block =>
      rOut = block.forward(rOut, mask = 0)
    }

    // Gating mechanism: combine question and response aspects
    val w = torch.sigmoid(cWeight.forward(qOut).add(tWeight.forward(rOut)))
    val dOutput = w.mul(qOut).add(w.mul(new Scalar(1.0)).neg().add(new Scalar(1.0)).mul(rOut))

    // Concatenate for output
    val concatQ = torch.cat(new TensorVector(dOutput, qEmb), 2)
    val logits = outMLP.forward(concatQ)  // (batch, seq, 1)

    logits.sigmoid().squeeze(2)
  }

  def predict(conceptIds: Tensor, responses: Tensor): Tensor = forward(conceptIds, responses)
}
