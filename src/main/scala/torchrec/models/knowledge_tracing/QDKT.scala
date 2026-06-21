package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.models.knowledge_tracing.layers.InteractionEmbedding
import torchrec.utils.DeviceSupport

/**
 * QDKT: Question-specific Deep Knowledge Tracing
 *
 * A variant of DKT that makes predictions specifically for each question rather than
 * predicting general concept correctness. This allows for question-specific difficulty
 * modeling and more accurate predictions.
 *
 * Reference: "Question-specific Deep Knowledge Tracing"
 *
 * Architecture:
 *   Interaction Embedding (question-specific) -> LSTM -> Question-specific Output
 *
 * @param numQuestions   Number of unique questions
 * @param numConcepts   Number of concepts (for embedding)
 * @param embedDim      Embedding dimension
 * @param numLayers     Number of LSTM layers
 * @param dropout       Dropout rate
 * @param device        Device
 */
class QDKT(
  numQuestions: Long,
  numConcepts: Long,
  embedDim: Int = 64,
  numLayers: Int = 1,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  // Question interaction embedding: (question_id + num_questions * response)
  private val interactionEmb = new EmbeddingImpl(
    new EmbeddingOptions(numQuestions * 2, embedDim)
  )
  register_module("interaction_emb", interactionEmb)

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

  // Output layer - predicts per question
  private val outputLayer = new LinearImpl(embedDim, 1)
  register_module("output", outputLayer)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    interactionEmb.to(dev, false)
    lstm.to(dev, false)
    outputLayer.to(dev, false)
  }

  /**
   * Forward pass for QDKT.
   * @param questionIds  Question IDs (batch, seqLen)
   * @param conceptIds    Concept IDs (batch, seqLen) - used for concept-aware output selection
   * @param responses    Responses 0/1 (batch, seqLen)
   * @return Predictions (batch, seqLen) - probability of correct response for each question
   */
  def forward(
    questionIds: Tensor,
    conceptIds: Tensor,
    responses: Tensor
  ): Tensor = {
    val batchSize = questionIds.size(0).toInt
    val seqLen = questionIds.size(1).toInt

    // Clamp IDs to valid range
    val minScalar = new org.bytedeco.pytorch.Scalar(0)
    val maxQuestionScalar = new org.bytedeco.pytorch.Scalar((numQuestions * 2 - 1).toDouble)
    val maxConceptScalar = new org.bytedeco.pytorch.Scalar(numConcepts.toDouble)
    val maxResponseScalar = new org.bytedeco.pytorch.Scalar(1)

    val qIdsLong = questionIds.toType(ScalarType.Long).clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxQuestionScalar)
    )
    val rLong = responses.toType(ScalarType.Long).clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxResponseScalar)
    )

    // Create interaction IDs: question * 2 + response
    val interactionIds = qIdsLong.mul(new Scalar(2)).add(rLong)

    // Get interaction embeddings (defensive: ensure Long dtype)
    val emb = interactionEmb.forward(interactionIds.toType(ScalarType.Long))  // (batch, seq, embedDim)

    // LSTM forward
    val lstmOut = lstm.forward(emb).get0()  // (batch, seq, embedDim)

    // Apply dropout
    val dropped = dropoutLayer.forward(lstmOut)

    // Output predictions per question
    val logits = outputLayer.forward(dropped)  // (batch, seq, 1)

    logits.sigmoid().squeeze(2)
  }

  def predict(
    questionIds: Tensor,
    conceptIds: Tensor,
    responses: Tensor
  ): Tensor = forward(questionIds, conceptIds, responses)
}
