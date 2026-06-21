package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.basic.layers.MLP
import torchrec.models.knowledge_tracing.layers.{CosinePositionalEmbedding, MultiHeadAttention, TransformerLayer}
import torchrec.utils.DeviceSupport

/**
 * DIMKT: Data-aware Inductive Moment Knowledge Tracing
 *
 * A knowledge tracing model that combines:
 * - Three concept views (exercise, knowledge component, skill) for comprehensive student modeling
 * - Data-aware prior distributions learned from concept statistics
 * - Inductive moment module capturing temporal learning dynamics via LSTM
 * - Cross-attention between concept views for multi-modal fusion
 *
 * Architecture:
 *   Three concept embeddings -> Cross-attention fusion -> LSTM (moment modeling)
 *   -> Output MLP -> Sigmoid
 *
 * @param numConcepts      Number of unique concepts/questions
 * @param embedDim        Model dimension
 * @param numHeads        Number of attention heads
 * @param numBlocks       Number of transformer blocks for cross-attention
 * @param hiddenDim       Hidden dimension for LSTM
 * @param dropout         Dropout rate
 * @param device          Device
 */
class DIMKT(
  numConcepts: Long,
  embedDim: Int = 64,
  numHeads: Int = 8,
  numBlocks: Int = 2,
  hiddenDim: Int = 64,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0, "embedDim must be divisible by numHeads")

  // Three concept embedding tables for multi-view concept representation
  // View 1: Exercise/concept embedding
  private val conceptEmb1 = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim))
  register_module("concept_emb1", conceptEmb1)

  // View 2: Knowledge component embedding
  private val conceptEmb2 = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim))
  register_module("concept_emb2", conceptEmb2)

  // View 3: Skill/mastery level embedding
  private val conceptEmb3 = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim))
  register_module("concept_emb3", conceptEmb3)

  // Response embedding
  private val responseEmb = new EmbeddingImpl(new EmbeddingOptions(2 + 1, embedDim))
  register_module("response_emb", responseEmb)

  // Data-aware module: learnable concept priors for data-aware attention
  // These capture concept difficulty/statistics from training data
  private val conceptPrior1 = {
    val t = torch.randn(Array((numConcepts + 1).toLong, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .mul(new Scalar(0.01f))
    register_parameter("concept_prior1", t)
    t
  }

  private val conceptPrior2 = {
    val t = torch.randn(Array((numConcepts + 1).toLong, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .mul(new Scalar(0.01f))
    register_parameter("concept_prior2", t)
    t
  }

  // Cross-attention modules for fusing the three concept views
  private val crossAttn12 = new MultiHeadAttention(embedDim, numHeads, dropout, device)
  register_module("cross_attn12", crossAttn12)

  private val crossAttn13 = new MultiHeadAttention(embedDim, numHeads, dropout, device)
  register_module("cross_attn13", crossAttn13)

  // Inductive moment module: LSTM to capture temporal learning dynamics
  private val lstmOptions = new LSTMOptions(embedDim, hiddenDim)
  lstmOptions.batch_first().put(true)
  lstmOptions.dropout().put(dropout.toDouble)
  private val momentLSTM = new LSTMImpl(lstmOptions)
  register_module("moment_lstm", momentLSTM)

  // Output projection from LSTM hidden state
  private val outputProj = new LinearImpl(hiddenDim, embedDim)
  register_module("output_proj", outputProj)

  // Output MLP
  private val outMLP = new MLP(embedDim, List(embedDim.toLong), 1, "relu", dropout, device = device)
  register_module("out_mlp", outMLP)

  // Dropout
  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropout", dropoutLayer)

  // Layer norm shapes
  private def layerNormShape(d: Int) = { val v = new LongVector(1); v.put(0, d.toLong); v }

  // Layer norms for residual connections
  private val ln1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
  private val ln2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(hiddenDim)))
  register_module("ln1", ln1)
  register_module("ln2", ln2)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    conceptEmb1.to(dev, false)
    conceptEmb2.to(dev, false)
    conceptEmb3.to(dev, false)
    responseEmb.to(dev, false)
    outputProj.to(dev, false)
    outMLP.to(dev, false)
  }

  /**
   * Forward pass for DIMKT.
   * @param conceptIds1  First concept IDs - exercise/concept view (batch, seqLen)
   * @param conceptIds2  Second concept IDs - knowledge component view (batch, seqLen)
   * @param conceptIds3  Third concept IDs - skill/mastery view (batch, seqLen)
   * @param responses    Responses 0/1 (batch, seqLen)
   * @return Predictions (batch, seqLen) - probability of correct response
   */
  def forward(
    conceptIds1: Tensor,
    conceptIds2: Tensor,
    conceptIds3: Tensor,
    responses: Tensor
  ): Tensor = {
    val batchSize = conceptIds1.size(0).toInt
    val seqLen = conceptIds1.size(1).toInt

    // Clamp concept IDs to valid range and convert to Long
    val minScalar = new org.bytedeco.pytorch.Scalar(0)
    val maxScalarConcept = new org.bytedeco.pytorch.Scalar(numConcepts.toDouble)
    val maxScalarResponse = new org.bytedeco.pytorch.Scalar(1)

    // Clamp first on original type, then convert to Long
    val c1Clamped = conceptIds1.clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxScalarConcept)
    ).toType(ScalarType.Long)
    val c2Clamped = conceptIds2.clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxScalarConcept)
    ).toType(ScalarType.Long)
    val c3Clamped = conceptIds3.clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxScalarConcept)
    ).toType(ScalarType.Long)
    val rClamped = responses.clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxScalarResponse)
    ).toType(ScalarType.Long)

    // Get embeddings for all three concept views
    val emb1 = conceptEmb1.forward(c1Clamped)  // (batch, seqLen, embedDim)
    val emb2 = conceptEmb2.forward(c2Clamped)  // (batch, seqLen, embedDim)
    val emb3 = conceptEmb3.forward(c3Clamped)  // (batch, seqLen, embedDim)

    // Get response embeddings
    val resEmb = responseEmb.forward(rClamped)  // (batch, seqLen, embedDim)

    // Data-aware fusion: Add learned concept priors
    val prior1 = conceptPrior1.index_select(0, c1Clamped.toType(ScalarType.Long).view(-1L)).view(batchSize, seqLen, embedDim)
    val prior2 = conceptPrior2.index_select(0, c2Clamped.toType(ScalarType.Long).view(-1L)).view(batchSize, seqLen, embedDim)

    val fused1 = emb1.add(prior1).add(resEmb)
    val fused2 = emb2.add(prior1).add(resEmb)
    val fused3 = emb3.add(prior2).add(resEmb)

    // Cross-attention between views
    // View 1 attends to View 2
    val attn12 = crossAttn12.forward(fused1, fused2, fused2)
    val combined12 = ln1.forward(fused1.add(dropoutLayer.forward(attn12)))

    // View 1 attends to View 3
    val attn13 = crossAttn13.forward(combined12, fused3, fused3)
    val combined = ln1.forward(combined12.add(dropoutLayer.forward(attn13)))

    // Add residual from emb3 for richer representation
    val enriched = combined.add(fused3)

    // Inductive moment module: LSTM processes temporal dynamics
    val lstmOut = momentLSTM.forward(enriched).get0()  // (batch, seqLen, hiddenDim)

    // Output projection with residual connection
    val projected = outputProj.forward(dropoutLayer.forward(lstmOut))
    val residualOut = ln2.forward(projected.add(lstmOut))

    // Output MLP
    val logits = outMLP.forward(residualOut)  // (batch, seqLen, 1)

    logits.sigmoid().squeeze(2)
  }

  /**
   * Predict next response.
   * Uses self as target concept for prediction.
   */
  def predict(
    conceptIds1: Tensor,
    conceptIds2: Tensor,
    conceptIds3: Tensor,
    responses: Tensor
  ): Tensor = forward(conceptIds1, conceptIds2, conceptIds3, responses)
}
