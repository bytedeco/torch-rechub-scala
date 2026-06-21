package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.models.knowledge_tracing.layers.InteractionEmbedding
import torchrec.utils.DeviceSupport

/**
 * DKT: Deep Knowledge Tracing
 * Uses LSTM to model student knowledge state over interaction sequences.
 *
 * Reference: "Deep Knowledge Tracing" (Piech et al., NeurIPS 2015)
 *
 * Architecture:
 *   Interaction Embedding (concept + response) -> LSTM -> Output Layer -> Sigmoid
 *
 * @param numConcepts  Number of unique concepts
 * @param embedDim    Embedding dimension (also hidden dim)
 * @param numLayers   Number of LSTM layers
 * @param dropout     Dropout rate
 * @param device      Device
 */
class DKT(
  numConcepts: Long,
  embedDim: Int = 64,
  numLayers: Int = 1,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private val interactionEmb = new InteractionEmbedding(numConcepts, embedDim, device)
  register_module("interaction_emb", interactionEmb)
  var opt = new LSTMOptions(embedDim, embedDim)
  opt.num_layers().put(numLayers.toInt)
  opt.dropout().put(dropout.toDouble)
  opt.batch_first().put(true)
  private val lstm = new LSTMImpl(opt)
  register_module("lstm", lstm)

  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropout", dropoutLayer)

  private val outputLayer = new LinearImpl(embedDim, numConcepts)
  register_module("output", outputLayer)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    lstm.to(dev, false)
    outputLayer.to(dev, false)
  }

  /**
   * Forward pass for DKT.
   * @param conceptIds  Concept IDs (batch, seqLen)
   * @param responses   Responses 0/1 (batch, seqLen)
   * @return Predictions (batch, seqLen, numConcepts) - probability of correct response for each concept
   */
  def forward(conceptIds: Tensor, responses: Tensor): Tensor = {
    val batchSize = conceptIds.size(0).toInt
    val seqLen = conceptIds.size(1).toInt

    // Get interaction embeddings
    val interactionEmbOut = interactionEmb.forward(conceptIds, responses)

    // LSTM forward
    val lstmOut = lstm.forward(interactionEmbOut).get0()

    // Apply dropout
    val dropped = dropoutLayer.forward(lstmOut)

    // Output: predict correctness for each concept
    val logits = outputLayer.forward(dropped)

    // Return shape: (batch, seqLen, numConcepts)
    logits
  }

  /**
   * Predict next response for a given concept.
   * @param conceptIds  Concept IDs (batch, seqLen)
   * @param responses   Responses 0/1 (batch, seqLen)
   * @return Predictions (batch, seqLen, numConcepts)
   */
  def predict(conceptIds: Tensor, responses: Tensor): Tensor = {
    forward(conceptIds, responses).sigmoid()
  }
}

/**
 * DKT+: Deep Knowledge Tracing with Regularization
 * Same as DKT but stores regularization parameters.
 *
 * Reference: Extended DKT with SLM loss
 */
class DKTPlus(
  numConcepts: Long,
  embedDim: Int = 64,
  numLayers: Int = 1,
  dropout: Float = 0.2f,
  lambdaR: Float = 0.0f,
  lambdaW1: Float = 0.0f,
  lambdaW2: Float = 0.0f,
  device: String = DeviceSupport.backend
) extends Module {

  private val interactionEmb = new InteractionEmbedding(numConcepts, embedDim, device)
  register_module("interaction_emb", interactionEmb)
  var opt = new LSTMOptions(embedDim, embedDim)
  opt.num_layers().put(numLayers.toInt)
  opt.dropout().put(dropout.toDouble)
  opt.batch_first().put(true)
  private val lstm = new LSTMImpl(opt)
  register_module("lstm", lstm)

  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropout", dropoutLayer)

  private val outputLayer = new LinearImpl(embedDim, numConcepts)
  register_module("output", outputLayer)

  // Regularization parameters (stored but not applied in forward)
  private val lambdaRParam = lambdaR
  private val lambdaW1Param = lambdaW1
  private val lambdaW2Param = lambdaW2

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    lstm.to(dev, false)
    outputLayer.to(dev, false)
  }

  def forward(conceptIds: Tensor, responses: Tensor): Tensor = {
    val interactionEmbOut = interactionEmb.forward(conceptIds, responses)
    val lstmOut = lstm.forward(interactionEmbOut).get0()
    val dropped = dropoutLayer.forward(lstmOut)
    outputLayer.forward(dropped)
  }

  def predict(conceptIds: Tensor, responses: Tensor): Tensor = {
    forward(conceptIds, responses).sigmoid()
  }
}
