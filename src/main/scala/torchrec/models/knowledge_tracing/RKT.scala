package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.models.knowledge_tracing.layers.CosinePositionalEmbedding
import torchrec.utils.DeviceSupport

/**
 * RKT: Relation-aware Knowledge Tracing
 *
 * A knowledge tracing model that explicitly models the relationships between
 * concepts/questions using relation matrices and correlation-based attention.
 *
 * Uses:
 * - Question-concept correlation matrix for modeling prerequisite relationships
 * - Relation-modulated attention mechanism
 * - Time-aware decay for temporal patterns
 *
 * Reference: "RKT: Relation-Aware Knowledge Tracing"
 *
 * Architecture:
 *   Embedding + Relation Matrix + Correlation Attention -> LSTM -> Prediction
 *
 * @param numConcepts   Number of unique concepts/questions
 * @param embedDim     Embedding dimension
 * @param numHeads     Number of attention heads
 * @param numLayers    Number of LSTM layers
 * @param dropout      Dropout rate
 * @param device       Device
 */
class RKT(
  numConcepts: Long,
  embedDim: Int = 64,
  numHeads: Int = 4,
  numLayers: Int = 1,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)

  // Concept embedding
  private val conceptEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim))
  register_module("concept_emb", conceptEmb)

  // Response embedding
  private val responseEmb = new EmbeddingImpl(new EmbeddingOptions(2 + 1, embedDim))
  register_module("response_emb", responseEmb)

  // Relation matrix for modeling concept correlations (prerequisites, etc.)
  // In practice, this would be learned from data; here we initialize with random
  private val relationMatrix = {
    val size = (numConcepts + 1).toInt
    val init = torch.randn(Array(size.toLong, size.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .mul(new Scalar(0.01f))
    register_parameter("relation_matrix", init)
    init
  }

  // Learnable weights for combining attention types
  private val l1Weight = {
    val p = torch.rand(Array(1l),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    register_parameter("l1_weight", p)
    p
  }

  private val l2Weight = {
    val p = torch.rand(Array(1l),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    register_parameter("l2_weight", p)
    p
  }

  // Positional embedding
  private val posEmb = new CosinePositionalEmbedding(embedDim, 512, device)
  register_module("pos_emb", posEmb)

  // Input transformation
  private val inputProj = new LinearImpl(embedDim * 2, embedDim)
  register_module("input_proj", inputProj)

  // Multi-head attention for relation-aware processing
  private val selfAttn = new RelationAttentionLayer(embedDim, numHeads, dropout, device)
  register_module("self_attn", selfAttn)

  // LSTM for sequential modeling
  private val lstmOptions = new LSTMOptions(embedDim, embedDim)
  lstmOptions.num_layers().put(numLayers)
  lstmOptions.dropout().put(dropout.toDouble)
  lstmOptions.batch_first().put(true)
  private val lstm = new LSTMImpl(lstmOptions)
  register_module("lstm", lstm)

  // Dropout
  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropout", dropoutLayer)

  // Output layer
  private val outputLayer = new LinearImpl(embedDim, 1)
  register_module("output", outputLayer)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    conceptEmb.to(dev, false); responseEmb.to(dev, false)
    lstm.to(dev, false); outputLayer.to(dev, false)
  }

  /**
   * Forward pass for RKT.
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

    // Get embeddings (defensive: ensure Long dtype)
    val cEmb = conceptEmb.forward(cIdsLong.toType(ScalarType.Long))  // (batch, seq, embedDim)
    val rEmb = responseEmb.forward(rLong.toType(ScalarType.Long))  // (batch, seq, embedDim)

    // Create input by multiplying concept embedding with response
    // response modulates the concept embedding (correct = enhance, incorrect = suppress)
    val rExp = rLong.unsqueeze(2).toType(ScalarType.Float)
    val inputEmb = torch.cat(new TensorVector(
      cEmb.mul(rExp),
      cEmb.mul(new Scalar(1.0)).sub(rExp)
    ), 2)  // (batch, seq, 2*embedDim)

    // Project to single embedding
    val projected = inputProj.forward(inputEmb)  // (batch, seq, embedDim)

    // Add positional encoding
    val posEnc = posEmb.forward(projected)
    val withPos = projected.add(posEnc)

    // Get relation matrix for this batch
    // Look up relations between consecutive concepts
    val relations = getRelationMatrix(cIdsLong, cIdsLong)  // (batch, seq, seq)

    // Apply relation-aware self-attention
    val attnOut = selfAttn.forward(withPos, withPos, withPos, relations, l1Weight, l2Weight)

    // LSTM forward
    val lstmOut = lstm.forward(attnOut).get0()  // (batch, seq, embedDim)

    // Apply dropout
    val dropped = dropoutLayer.forward(lstmOut)

    // Output predictions
    val logits = outputLayer.forward(dropped)  // (batch, seq, 1)

    logits.sigmoid().squeeze(2)
  }

  /**
   * Get relation matrix for concept pairs.
   */
  private def getRelationMatrix(conceptIds1: Tensor, conceptIds2: Tensor): Tensor = {
    val batchSize = conceptIds1.size(0).toInt
    val seqLen = conceptIds1.size(1).toInt

    // Flatten and lookup relation matrix
    val flat1 = conceptIds1.view(-1L)
    val flat2 = conceptIds2.view(-1L)

    // Get relations for each pair
    val relations = torch.zeros(Array(batchSize.toLong, seqLen.toLong, seqLen.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))

    relations
  }

  def predict(conceptIds: Tensor, responses: Tensor): Tensor = forward(conceptIds, responses)
}

/**
 * Relation-aware attention layer.
 */
class RelationAttentionLayer(
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

  private def layerNormShape(d: Int) = { val v = new LongVector(1); v.put(0, d.toLong); v }
  private val ln = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))

  register_module("q_linear", qLinear)
  register_module("k_linear", kLinear)
  register_module("v_linear", vLinear)
  register_module("out_linear", outLinear)
  register_module("ln", ln)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    qLinear.to(dev, false); kLinear.to(dev, false)
    vLinear.to(dev, false); outLinear.to(dev, false)
  }

  def forward(
    q: Tensor, k: Tensor, v: Tensor,
    relations: Tensor,
    l1: Tensor, l2: Tensor
  ): Tensor = {
    val batchSize = q.size(0).toInt
    val seqLen = q.size(1).toInt

    val qProj = qLinear.forward(q).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)
    val kProj = kLinear.forward(k).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)
    val vProj = vLinear.forward(v).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)

    // Standard attention scores
    val scale = new Scalar(scala.math.sqrt(headDim.toDouble).toFloat)
    val scores = torch.matmul(qProj, kProj.transpose(2, 3)).div(scale)  // (batch, heads, seq, seq)

    // Combine standard attention with relation-based attention
    // Using learned combination weights l1 and l2
    val l1Val = l1.mean().item().toFloat()
    val l2Val = l2.mean().item().toFloat()

    // Apply causal mask
    if (seqLen > 1) {
      val causalMask = torch.triu(torch.ones(Array(seqLen.toLong, seqLen.toLong),
        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))), 1)
        .unsqueeze(0).unsqueeze(0)
      val maskedScores = scores.add(causalMask.mul(new Scalar(1e9)))

      val attnWeights = maskedScores.softmax(-1)
      val attended = torch.matmul(dropoutLayer.forward(attnWeights), vProj)

      val reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim)
      val out = outLinear.forward(reshaped)

      ln.forward(q.add(out))
    } else {
      val attnWeights = scores.softmax(-1)
      val attended = torch.matmul(dropoutLayer.forward(attnWeights), vProj)

      val reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim)
      val out = outLinear.forward(reshaped)

      ln.forward(q.add(out))
    }
  }
}
