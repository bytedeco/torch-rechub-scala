package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.utils.DeviceSupport

import scala.collection.mutable

/**
 * GKT: Graph-based Knowledge Tracing (Simplified)
 * Uses LSTM instead of complex GNN for simplicity.
 *
 * @param numConcepts  Number of unique concepts
 * @param embedDim     Embedding/hidden dimension
 * @param hiddenDim    RNN hidden dimension
 * @param dropout      Dropout rate
 * @param graphType    Graph type (unused in simplified version)
 * @param device       Device
 */
class GKT(
  numConcepts: Long,
  embedDim: Int = 64,
  hiddenDim: Int = 64,
  dropout: Float = 0.5f,
  graphType: String = "dense",
  device: String = DeviceSupport.backend
) extends Module {

  // Interaction embedding
  private val interactionEmb = new EmbeddingImpl(
    new EmbeddingOptions(numConcepts * 2 + 1, embedDim)
  )
  register_module("interaction_emb", interactionEmb)

  // Concept embedding
  private val conceptEmb = new EmbeddingImpl(
    new EmbeddingOptions(numConcepts + 1, embedDim)
  )
  register_module("concept_emb", conceptEmb)

  // LSTM for temporal dynamics
  var lstmOpts = new LSTMOptions(embedDim, hiddenDim)
  lstmOpts.num_layers().put(1)
  lstmOpts.dropout().put(dropout)
  lstmOpts.batch_first().put(true)
  private val lstm = new LSTMImpl(lstmOpts)
  register_module("lstm", lstm)

  private val dropoutLayer = new DropoutImpl(dropout)

  // Prediction layer
  private val predictLayer = new LinearImpl(hiddenDim, 1)
  register_module("predict", predictLayer)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    this.to(dev, false)
  }

  /**
   * Forward pass for GKT.
   * @param conceptIds Concept IDs (batch, seqLen)
   * @param responses  Responses 0/1 (batch, seqLen)
   * @return Predictions (batch, seqLen-1)
   */
  def forward(conceptIds: Tensor, responses: Tensor): Tensor = {
    val batchSize = conceptIds.size(0).toInt
    val seqLen = conceptIds.size(1).toInt

    // Convert to Long and clamp
    val conceptIdsLong = conceptIds.toType(ScalarType.Long)
    val responsesLong = responses.toType(ScalarType.Long)
    val minScalar = new org.bytedeco.pytorch.Scalar(0)
    val maxConceptScalar = new org.bytedeco.pytorch.Scalar(numConcepts)
    val maxInteractionScalar = new org.bytedeco.pytorch.Scalar(numConcepts * 2)

    val conceptIdsClamped = conceptIdsLong.clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxConceptScalar)
    ).toType(ScalarType.Long)

    val interactionIdsRaw = conceptIdsClamped.add(responsesLong.mul(new Scalar(numConcepts.toDouble)))
    val interactionIds = interactionIdsRaw.clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxInteractionScalar)
    ).toType(ScalarType.Long)

    val xEmb = interactionEmb.forward(interactionIds)
    val cEmb = conceptEmb.forward(conceptIdsClamped)

    // Combine interaction and concept embeddings
    val combinedEmb = xEmb.add(cEmb)

    // LSTM forward
    val lstmOut = lstm.forward(combinedEmb).get0()

    val dropped = dropoutLayer.forward(lstmOut)

    // Predict
    val logits = predictLayer.forward(dropped)  // (batch, seqLen, 1)
    logits.sigmoid()  // Keep 3D for KTTrainer compatibility
  }

  def predict(conceptIds: Tensor, responses: Tensor): Tensor = forward(conceptIds, responses)
}
