package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.basic.layers.MLP
import torchrec.models.knowledge_tracing.layers.{CosinePositionalEmbedding, TransformerLayer}
import torchrec.utils.DeviceSupport

/**
 * PromptKT: Prompt-based Knowledge Tracing
 *
 * A knowledge tracing model that uses prompt-based learning to:
 * - Incorporate knowledge prompts into embeddings
 * - Use dataset-specific prompts
 * - Apply prompt tuning for efficient fine-tuning
 *
 * Reference: "PromptKT: Prompt-based Knowledge Tracing"
 *
 * Architecture:
 *   Knowledge Prompts + Concept Embeddings -> Transformer Encoder -> MLP -> Prediction
 *
 * @param numConcepts   Number of unique concepts/questions
 * @param embedDim     Model dimension
 * @param numHeads     Number of attention heads
 * @param numBlocks    Number of transformer blocks
 * @param dropout       Dropout rate
 * @param device        Device
 */
class PromptKT(
  numConcepts: Long,
  embedDim: Int = 64,
  numHeads: Int = 8,
  numBlocks: Int = 2,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)

  // Dataset prompt embedding (learnable)
  private val datasetPrompt = {
    val p = torch.randn(Array(20l, embedDim),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .mul(new Scalar(0.01f))
    register_parameter("dataset_prompt", p)
    p
  }

  // Question embedding
  private val qEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim))
  register_module("q_embed", qEmbed)

  // Concept embedding
  private val cEmbed = {
    val p = torch.randn(Array(numConcepts + 1, embedDim),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .mul(new Scalar(0.01f))
    register_parameter("c_embed", p)
    p
  }

  // QA interaction embedding
  private val qaEmbed = new EmbeddingImpl(new EmbeddingOptions(2, embedDim))
  register_module("qa_embed", qaEmbed)

  // Prompt MLP for generating concept prompts
  private val promptMLP = new MLP(embedDim, List(embedDim.toLong, embedDim), embedDim, "relu", dropout, device = device)
  register_module("prompt_mlp", promptMLP)

  // Positional embedding
  private val posEmb = new CosinePositionalEmbedding(embedDim, 512, device)
  register_module("pos_emb", posEmb)

  // Transformer blocks for prompt-aware encoding
  private val blocks = (0 until numBlocks).map { i =>
    val block = new TransformerLayer(embedDim, numHeads, embedDim * 4, dropout, device)
    register_module(s"block_$i", block)
    block
  }

  // Output MLP
  private val outMLP = new MLP(embedDim * 2, List(embedDim.toLong, embedDim / 2), 1, "relu", dropout, device = device)
  register_module("out_mlp", outMLP)

  // Dropout
  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropout", dropoutLayer)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    qEmbed.to(dev, false); qaEmbed.to(dev, false)
    promptMLP.to(dev, false); outMLP.to(dev, false)
  }

  /**
   * Forward pass for PromptKT.
   * @param conceptIds  Concept IDs (batch, seqLen)
   * @param responses   Responses 0/1 (batch, seqLen)
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

    // Get question embeddings (defensive: ensure Long dtype)
    val qEmb = qEmbed.forward(cIdsLong.toType(ScalarType.Long))  // (batch, seq, embedDim)

    // Get concept embeddings
    val cEmb = cEmbed.index_select(0, cIdsLong.toType(ScalarType.Long).view(-1L)).view(batchSize, seqLen, embedDim)

    // Average concept embeddings for prompt generation
    val cMask = cIdsLong.ne(new Scalar(0)).unsqueeze(2).toType(ScalarType.Float)
    val cSum = cEmb.mul(cMask).sum(1)
    val cCount = cMask.sum(1).add(new Scalar(1e-8.toDouble))
    val cAvg = cSum.div(cCount)  // (batch, embedDim)

    // Generate prompts using MLP
    val prompts = promptMLP.forward(cAvg)  // (batch, embedDim)

    // Get QA embeddings (defensive: ensure Long dtype for indices)
    val qaEmb = qaEmbed.forward(rLong.toType(ScalarType.Long))  // (batch, seq, embedDim)

    // Combine question, concept, and prompt embeddings
    val combinedEmb = qEmb.add(cEmb).add(prompts.unsqueeze(1))  // (batch, seq, embedDim)

    // Add positional encoding
    val posEnc = posEmb.forward(combinedEmb)
    val embWithPos = combinedEmb.add(posEnc)

    // Add QA interaction information
    val embWithQA = embWithPos.add(qaEmb)

    // Process through transformer blocks
    var x = embWithQA
    blocks.foreach { block =>
      x = block.forward(x, mask = 0)
    }

    // Concatenate for output
    val concatQ = torch.cat(new TensorVector(x, qEmb), 2)
    val logits = outMLP.forward(concatQ)

    logits.sigmoid().squeeze(2)
  }

  def predict(conceptIds: Tensor, responses: Tensor): Tensor = forward(conceptIds, responses)
}
