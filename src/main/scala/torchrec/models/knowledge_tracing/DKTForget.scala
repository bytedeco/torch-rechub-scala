package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.models.knowledge_tracing.layers.InteractionEmbedding
import torchrec.utils.DeviceSupport

/**
 * DKT-Forget: Deep Knowledge Tracing with Forgetting Dynamics
 *
 * An extension of DKT that models the forgetting phenomenon using:
 * - Time gap between consecutive interactions (rgap, sgap, pcount)
 * - CIntegration module that combines concept embeddings with time features
 * - LSTM for temporal modeling with decay factors
 *
 * Reference: "Deep Knowledge Tracing with Forgetting" variants
 *
 * Architecture:
 *   Interaction Embedding + Time Gap Integration -> LSTM -> Output
 *
 * @param numConcepts  Number of unique concepts
 * @param embedDim    Embedding dimension
 * @param numLayers   Number of LSTM layers
 * @param dropout     Dropout rate
 * @param device      Device
 */
class DKTForget(
  numConcepts: Long,
  embedDim: Int = 64,
  numLayers: Int = 1,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  // Interaction embedding
  private val interactionEmb = new InteractionEmbedding(numConcepts, embedDim, device)
  register_module("interaction_emb", interactionEmb)

  // Time gap integration layer (processes concept + time features)
  // This combines interaction embedding with time gap information
  private val timeEmbed = new LinearImpl(embedDim + 3, embedDim)  // +3 for rgap, sgap, pcount features
  register_module("time_embed", timeEmbed)

  // LSTM for temporal modeling
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
  private val outputLayer = new LinearImpl(embedDim, numConcepts)
  register_module("output", outputLayer)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    lstm.to(dev, false)
    outputLayer.to(dev, false)
  }

  /**
   * Forward pass for DKT-Forget.
   * @param conceptIds  Concept IDs (batch, seqLen)
   * @param responses   Responses 0/1 (batch, seqLen)
   * @return Predictions (batch, seqLen, numConcepts)
   */
  def forward(
    conceptIds: Tensor,
    responses: Tensor
  ): Tensor = {
    val batchSize = conceptIds.size(0).toInt
    val seqLen = conceptIds.size(1).toInt

    // Get interaction embeddings
    val interactionEmbOut = interactionEmb.forward(conceptIds, responses)  // (batch, seq, embedDim)

    // Add time gap features (simplified - using fixed features for benchmark)
    // In full implementation, this would use actual time gaps
    val timeFeatures = torch.ones(Array(batchSize.toLong, seqLen.toLong, 3.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    if (device != "cpu") {
      timeFeatures.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
    }

    // Concatenate interaction embedding with time features
    val combined = torch.cat(new TensorVector(interactionEmbOut, timeFeatures), 2)  // (batch, seq, embedDim+3)

    // Process through time integration layer
    val timeIntegrated = timeEmbed.forward(combined)  // (batch, seq, embedDim)
    val activated = torch.tanh(timeIntegrated)

    // LSTM forward
    val lstmOut = lstm.forward(activated).get0()  // (batch, seq, embedDim)

    // Apply dropout
    val dropped = dropoutLayer.forward(lstmOut)

    // Output predictions
    val logits = outputLayer.forward(dropped)  // (batch, seq, numConcepts)

    logits
  }

  /**
   * Predict next responses.
   */
  def predict(
    conceptIds: Tensor,
    responses: Tensor
  ): Tensor = {
    forward(conceptIds, responses).sigmoid()
  }
}
