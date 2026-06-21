package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.basic.layers.MLP
import torchrec.utils.DeviceSupport

/**
 * IEKT: Individual Estimation Knowledge Tracing
 *
 * A knowledge tracing model that estimates individual student knowledge states
 * using:
 * - Cognitive matrix for modeling skill levels
 * - Acquisition sensitivity for modeling student-specific learning rates
 * - GRU-based state updates
 * - Policy gradient for cognitive action selection
 *
 * Reference: "IEKT: Individual Estimation Knowledge Tracing"
 *
 * Architecture:
 *   Concept Embedding -> Cognitive Policy -> GRU State Update -> Prediction
 *
 * @param numConcepts     Number of unique concepts/questions
 * @param embedDim       Embedding dimension
 * @param numCogLevels   Number of cognitive levels
 * @param numAcqLevels   Number of acquisition sensitivity levels
 * @param numLayers      Number of GRU layers
 * @param dropout        Dropout rate
 * @param gamma          Discount factor for RL
 * @param device         Device
 */
class IEKT(
  numConcepts: Long,
  embedDim: Int = 64,
  numCogLevels: Int = 10,
  numAcqLevels: Int = 10,
  numLayers: Int = 1,
  dropout: Float = 0.2f,
  gamma: Float = 0.93f,
  device: String = DeviceSupport.backend
) extends Module {

  // Concept embedding
  private val conceptEmb = {
    var p = torch.randn(Array(numConcepts + 1, embedDim),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .mul(new Scalar(0.01f))
    if (device != "cpu") p = p.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
    register_parameter("concept_emb", p)
    p
  }

  // Question embedding
  private val qEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim))
  register_module("q_embed", qEmbed)

  // Cognitive matrix (skill levels)
  private val cogMatrix = {
    var p = torch.randn(Array(numCogLevels.toLong, embedDim * 2),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    if (device != "cpu") p = p.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
    register_parameter("cog_matrix", p)
    p
  }

  // Acquisition sensitivity matrix
  private val acqMatrix = {
    var p = torch.randn(Array(numAcqLevels.toLong, embedDim * 2),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    if (device != "cpu") p = p.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
    register_parameter("acq_matrix", p)
    p
  }

  // Predictor MLP (input is hQConcat (2*embed) concatenated with rt.narrow(.,0,embed) => 3*embed)
  private val predictor = new MLP(embedDim * 3, List(embedDim.toLong), 1, "relu", dropout, device = device)
  register_module("predictor", predictor)

  // Cognitive selector MLP
  private val cogSelector = new MLP(embedDim * 4, List(embedDim.toLong), numCogLevels, "relu", dropout, device = device)
  register_module("cog_selector", cogSelector)

  // Acquisition selector MLP
  private val acqSelector = new MLP(embedDim * 4, List(embedDim.toLong), numAcqLevels, "relu", dropout, device = device)
  register_module("acq_selector", acqSelector)

  // GRU cell for state update (input: q + cogEmb + acqEmb => 5*embedDim)
  private val gruCell = new GRUCellImpl(embedDim * 5, embedDim)
  register_module("gru_cell", gruCell)

  // Dropout
  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropout", dropoutLayer)

  // Output layer
  private val outputLayer = new LinearImpl(embedDim, 1)
  register_module("output", outputLayer)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    qEmbed.to(dev, false)
    predictor.to(dev, false)
    cogSelector.to(dev, false)
    acqSelector.to(dev, false)
    gruCell.to(dev, false)
    outputLayer.to(dev, false)
  }

  /**
   * Forward pass for IEKT.
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

    // Get concept embeddings
    // Debug: ensure index tensor is Long and within expected range
    try {
      println(s"[DEBUG IEKT] cIdsLong dtype=${cIdsLong.dtype()} shape=${cIdsLong.shape().mkString(",")} min=${cIdsLong.toType(ScalarType.Long).view(-1L).min().itemSafe()} max=${cIdsLong.toType(ScalarType.Long).view(-1L).max().itemSafe()}")
    } catch { case _: Throwable => () }
    // Ensure indices are on the same device and have Long dtype before embedding/index_select
    val cIdsLongDev = cIdsLong.to(new org.bytedeco.pytorch.Device(device), ScalarType.Long)
    val cEmb = conceptEmb.index_select(0, cIdsLongDev.view(-1L)).view(batchSize, seqLen, embedDim)
    val qEmb = qEmbed.forward(cIdsLongDev)

    // Initialize hidden state
    var h = torch.zeros(Array(batchSize.toLong, embedDim),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    if (device != "cpu") {
      h = h.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
    }

    // RT tensor (knowledge state representation)
    var rt = torch.zeros(Array(batchSize.toLong, embedDim * 2),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    if (device != "cpu") {
      rt = rt.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
    }

    // Collect predictions
    val predictions = scala.collection.mutable.ListBuffer[Tensor]()

    for (i <- 0 until seqLen) {
      // Get concept at position i
      val q = qEmb.select(1, i)  // (batch, embedDim)

      // Concatenate hidden state and question for predictor input
      val hQConcat = torch.cat(new TensorVector(h, q), 1)  // (batch, 2*embedDim)

      // Concatenate with RT for full input
      val fullInput = torch.cat(new TensorVector(hQConcat, rt), 1)  // (batch, 4*embedDim)

      // Get cognitive action (simplified - using argmax instead of sampling)
      val cogLogits = cogSelector.forward(fullInput)  // (batch, numCogLevels)
      val cogAction = cogLogits.argmax(new LongOptional(1), true)  // (batch, 1)

      // Get acquisition action (simplified - using argmax instead of sampling)
      val acqLogits = acqSelector.forward(fullInput)  // (batch, numAcqLevels)
      val acqAction = acqLogits.argmax(new LongOptional(1), true)  // (batch, 1)

      // Look up cognitive and acquisition embeddings
       val cogActionDev = cogAction.to(new org.bytedeco.pytorch.Device(device), ScalarType.Long)
       val acqActionDev = acqAction.to(new org.bytedeco.pytorch.Device(device), ScalarType.Long)
       val cogEmb = cogMatrix.index_select(0, cogActionDev.view(-1L)).view(batchSize, embedDim * 2)
       val acqEmb = acqMatrix.index_select(0, acqActionDev.view(-1L)).view(batchSize, embedDim * 2)

      // Predict response probability
      val predInput = torch.cat(new TensorVector(hQConcat, rt.narrow(1, 0, embedDim)), 1)  // (batch, 3*embedDim)
      val predLogit = predictor.forward(predInput)  // (batch, 1)
      val prob = torch.sigmoid(predLogit)  // (batch, 1)
      predictions.append(prob.squeeze(1))

      // Get response at position i for state update
       val r = rLong.select(1, i).toType(ScalarType.Float).unsqueeze(1)  // (batch, 1)

       // Update RT based on response
       val correctMask = r  // (batch,1)
       val incorrectMask = torch.ones_like(r).sub(r) // (batch,1) = 1 - r

       // RT update: combine concept based on correctness
       // Expand question embedding to match RT half-size (2*embedDim)
       val qExpanded = torch.cat(new TensorVector(q, q), 1)  // (batch, 2*embedDim)
       val rtCorrect = qExpanded.mul(correctMask)  // Keep concept if correct
       val rtIncorrect = rt.narrow(1, 0, embedDim * 2).mul(incorrectMask)  // Keep old RT if incorrect
       rt = rtCorrect.add(rtIncorrect)

      // GRUCell input: combine question, cognitive, acquisition embeddings
      val gruInput = torch.cat(new TensorVector(q, cogEmb, acqEmb), 1)  // (batch, 4*embedDim)

      // Update hidden state
      h = gruCell.forward(gruInput, h)
    }

    // Stack predictions
    val predStack = torch.stack(new TensorVector(predictions.toSeq*), 1)  // (batch, seq)
    predStack
  }

  def predict(conceptIds: Tensor, responses: Tensor): Tensor = forward(conceptIds, responses)
}
