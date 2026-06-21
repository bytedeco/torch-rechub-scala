package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.models.knowledge_tracing.layers.InteractionEmbedding
import torchrec.utils.DeviceSupport

/**
 * AT-DKT: Attention-based Deep Knowledge Tracing
 *
 * Combines DKT (LSTM-based) with attention mechanism to:
 * - Capture long-range dependencies in student learning sequences
 * - Weight historical interactions based on relevance to current prediction
 * - Model both sequential patterns and important historical events
 *
 * Reference: "AT-DKT: Attention-based Deep Knowledge Tracing"
 *
 * Architecture:
 *   Interaction Embedding -> LSTM -> Self-Attention -> Output
 *
 * @param numConcepts   Number of unique concepts
 * @param embedDim     Embedding dimension
 * @param numLayers    Number of LSTM layers
 * @param numHeads     Number of attention heads
 * @param dropout      Dropout rate
 * @param device       Device
 */
class ATDKT(
  numConcepts: Long,
  embedDim: Int = 64,
  numLayers: Int = 1,
  numHeads: Int = 4,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)

  // Interaction embedding
  private val interactionEmb = new InteractionEmbedding(numConcepts, embedDim, device)
  register_module("interaction_emb", interactionEmb)

  // LSTM for sequential modeling
  private val lstmOptions = new LSTMOptions(embedDim, embedDim)
  lstmOptions.num_layers().put(numLayers)
  lstmOptions.dropout().put(dropout.toDouble)
  lstmOptions.batch_first().put(true)
  private val lstm = new LSTMImpl(lstmOptions)
  register_module("lstm", lstm)

  // Self-attention layer for capturing important historical patterns
  private val selfAttn = new SelfAttentionLayer(embedDim, numHeads, dropout, device)
  register_module("self_attn", selfAttn)

  // Dropout
  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropout", dropoutLayer)

  // Output layers
  private val fc1 = new LinearImpl(embedDim, embedDim)
  register_module("fc1", fc1)
  private val fc2 = new LinearImpl(embedDim, embedDim / 2)
  register_module("fc2", fc2)
  private val outputLayer = new LinearImpl(embedDim / 2, 1)
  register_module("output", outputLayer)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    lstm.to(dev, false)
    fc1.to(dev, false); fc2.to(dev, false); outputLayer.to(dev, false)
  }

  /**
   * Forward pass for AT-DKT.
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

    // Get interaction embeddings
    val interactionEmbOut = interactionEmb.forward(conceptIds, responses)

    // LSTM forward
    val lstmOut = lstm.forward(interactionEmbOut).get0()  // (batch, seq, embedDim)

    // Self-attention to capture important patterns
    val attnOut = selfAttn.forward(lstmOut)  // (batch, seq, embedDim)

    // Residual connection with LSTM output
    val combined = lstmOut.add(attnOut)

    // Apply dropout
    val dropped = dropoutLayer.forward(combined)

    // MLP output
    val h1 = torch.relu(fc1.forward(dropped))
    val h2 = torch.relu(fc2.forward(h1))
    val logits = outputLayer.forward(h2)

    logits.sigmoid().squeeze(2)
  }

  def predict(conceptIds: Tensor, responses: Tensor): Tensor = forward(conceptIds, responses)
}

/**
 * Self-Attention layer for capturing important historical patterns.
 */
class SelfAttentionLayer(
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

  def forward(x: Tensor): Tensor = {
    val batchSize = x.size(0).toInt
    val seqLen = x.size(1).toInt

    val q = qLinear.forward(x).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)
    val k = kLinear.forward(x).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)
    val v = vLinear.forward(x).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)

    val scale = new Scalar(scala.math.sqrt(headDim.toDouble).toFloat)
    var scores = torch.matmul(q, k.transpose(2, 3)).div(scale)

    // Create causal mask
    if (seqLen > 1) {
      val causalMask = torch.triu(torch.ones(Array(seqLen.toLong, seqLen.toLong),
        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))), 1)
        .unsqueeze(0).unsqueeze(0)
      scores = scores.add(causalMask.mul(new Scalar(1e9)))
    }

    val attnWeights = scores.softmax(-1)
    val attended = torch.matmul(dropoutLayer.forward(attnWeights), v)

    val reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim)
    val out = outLinear.forward(reshaped)

    // Residual connection and layer norm
    ln.forward(x.add(out))
  }
}
