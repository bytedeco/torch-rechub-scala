package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.models.knowledge_tracing.layers.{EraseAddGate, InteractionEmbedding, KeyValueMemory}
import torchrec.utils.DeviceSupport

import scala.collection.mutable

/**
 * DKVMN: Dynamic Key-Value Memory Networks
 *
 * Reference: "Dynamic Key-Value Memory Networks for Knowledge Tracing" (Zhang et al., 2017)
 *
 * Architecture:
 *   - Key memory stores concept representations (static)
 *   - Value memory stores student knowledge state (dynamic, updated per interaction)
 *   - Write: erase-and-add gates update value memory
 *   - Read: content-based attention reads from value memory
 *
 * @param numConcepts  Number of unique concepts
 * @param numQuestions Number of unique questions
 * @param memDim       Memory slot dimension (also embedding dimension)
 * @param memSize      Number of memory slots
 * @param dropout      Dropout rate
 * @param device       Device
 */
class DKVMN(
  numConcepts: Long,
  numQuestions: Long,
  memDim: Int = 64,
  memSize: Int = 20,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  // Key embedding: maps concept IDs to key vectors
  // Clamp concept IDs to [0, numConcepts-1], add 1 extra for safety
  private val kEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, memDim))
  register_module("k_emb", kEmb)

  // Value embedding: maps (concept, response) to value vectors
  // interactionId = conceptId + numConcepts * response, max = (numConcepts-1) + numConcepts = numConcepts*2-1
  // Add 1 extra for safety
  private val vEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2 + 1, memDim))
  register_module("v_emb", vEmb)

  // Key memory matrix: (memSize, memDim)
  private val mkInit: Tensor = {
    val init = torch.randn(Array(memSize.toLong, memDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .mul(new Scalar(scala.math.sqrt(1.0 / memDim.toDouble).toFloat))
    register_parameter("mk", init)
    init
  }

  // Value memory: (memSize, memDim)
  private val mvInit: Tensor = {
    val init = torch.zeros(Array(memSize.toLong, memDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    register_parameter("mv", init)
    init
  }

  // Erase gate - outputs scalar per sample for memory erase
  private val eLayer = new LinearImpl(memDim, 1)
  register_module("e_layer", eLayer)

  // Add gate - outputs scalar per sample for memory add
  private val aLayer = new LinearImpl(memDim, 1)
  register_module("a_layer", aLayer)

  // Read function: combines memory content with key
  private val fLayer = new LinearImpl(memDim * 2, memDim)
  register_module("f_layer", fLayer)

  // Prediction layer
  private val pLayer = new LinearImpl(memDim, 1)
  register_module("p_layer", pLayer)

  private val dropoutLayer = new DropoutImpl(dropout)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    this.to(dev, false)
  }

  /**
   * Forward pass for DKVMN.
   * @param conceptIds Concept IDs (batch, seqLen)
   * @param responses  Responses 0/1 (batch, seqLen)
   * @return Predictions (batch, seqLen)
   */
  def forward(conceptIds: Tensor, responses: Tensor): Tensor = {
    val batchSize = conceptIds.size(0).toInt
    val seqLen = conceptIds.size(1).toInt

    // Convert to Long and clamp to valid range
    val conceptIdsLong = conceptIds.toType(ScalarType.Long)
    val responsesLong = responses.toType(ScalarType.Long)

    // Key embeddings for concepts - clamp to valid range, then convert to Long
    val maxConceptId = numConcepts
    val minScalar = new org.bytedeco.pytorch.Scalar(0)
    val maxScalar = new org.bytedeco.pytorch.Scalar(maxConceptId)
    val kIdsClamped = conceptIdsLong.clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxScalar)
    ).toType(ScalarType.Long)
    val k = kEmb.forward(kIdsClamped)  // (batch, seqLen, memDim)

    // Interaction IDs for value embeddings - clamp to valid range, then convert to Long
    val maxInteractionId = numConcepts * 2
    val maxInteractionScalar = new org.bytedeco.pytorch.Scalar(maxInteractionId)
    val interactionIdsRaw = conceptIdsLong.add(responsesLong.mul(new Scalar(numConcepts.toDouble)))
    val interactionIds = interactionIdsRaw.clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxInteractionScalar)
    ).toType(ScalarType.Long)
    val v = vEmb.forward(interactionIds)  // (batch, seqLen, memDim)

    // Initialize value memory for each sample - clone and detach to ensure independent tensor
    val mvBase = mvInit.clone().view(1L, memSize.toLong, memDim.toLong).expand(batchSize.toLong, -1L, -1L).clone()

    // Process sequence step by step
    val preds = mutable.ArrayBuffer[Tensor]()
    var prevMv = mvBase

    for (t <- 0 until seqLen) {
      val kt = k.select(1, t)  // (batch, memDim)
      val vt = v.select(1, t)  // (batch, memDim)

      // Compute attention weights: softmax(k_t @ M_k^T)
      // Detach to avoid gradient issues with expanded parameter
      val mkT = mkInit.clone().view(1L, memSize.toLong, memDim.toLong).expand(batchSize.toLong, -1L, -1L).detach()
      val scores = torch.bmm(kt.unsqueeze(1), mkT.transpose(1, 2)).squeeze(1)  // (batch, memSize)
      val attn = scores.softmax(1)  // (batch, memSize)

      // Erase - expand eraseGate to match attention dimensions
      val eraseGate = torch.sigmoid(eLayer.forward(kt)).squeeze(1)  // (batch,)
      val eraseGateEx = eraseGate.unsqueeze(1).unsqueeze(2)  // (batch, 1, 1)
      val attnEx = attn.unsqueeze(2)  // (batch, memSize, 1)
      val newMv = prevMv.mul(attnEx.mul(eraseGateEx).neg().add(new Scalar(1.0)))

      // Add - expand addGate to match attention dimensions
      val addGate = torch.tanh(aLayer.forward(kt)).squeeze(1)  // (batch,)
      val addGateEx = addGate.unsqueeze(1).unsqueeze(2)  // (batch, 1, 1)
      val addedMv = newMv.add(attnEx.mul(addGateEx))

      prevMv = addedMv

      // Read: combine attention-weighted memory with key
      val readMem = (attn.unsqueeze(2).mul(addedMv)).sum(1)  // (batch, memDim)
      val combined = torch.cat(new TensorVector(readMem, kt), 1)  // (batch, memDim * 2)
      val fused = torch.tanh(fLayer.forward(combined))

      // Predict
      val pred = pLayer.forward(dropoutLayer.forward(fused))  // (batch, 1)
      preds += pred.sigmoid().squeeze(1)
    }

    // Stack predictions along dim=1 to get (batch, seqLen)
    if (preds.isEmpty) {
      torch.zeros(batchSize, seqLen)
    } else {
      torch.stack(new TensorVector(preds.toSeq: _*), 1)
    }
  }

  /**
   * Predict next response probability.
   */
  def predict(conceptIds: Tensor, responses: Tensor): Tensor = {
    forward(conceptIds, responses)
  }
}

/**
 * DeepIRT: Deep Item Response Theory
 * Combines DKVMN memory architecture with IRT-based prediction.
 *
 * Reference: "Deep-IRT: Making Deep Knowledge Tracing Explainable"
 *
 * @param numConcepts  Number of unique concepts
 * @param numQuestions Number of unique questions
 * @param memDim       Memory dimension
 * @param memSize      Number of memory slots
 * @param dropout      Dropout rate
 * @param device       Device
 */
class DeepIRT(
  numConcepts: Long,
  numQuestions: Long,
  memDim: Int = 64,
  memSize: Int = 20,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private val kEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, memDim))
  private val vEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2 + 2, memDim))
  register_module("k_emb", kEmb)
  register_module("v_emb", vEmb)

  private val mkInit = {
    val init = torch.randn(Array(memSize.toLong, memDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .mul(new Scalar(scala.math.sqrt(1.0 / memDim.toDouble).toFloat))
    register_parameter("mk", init)
    init
  }

  private val mvInit = {
    val init = torch.zeros(Array(memSize.toLong, memDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    register_parameter("mv", init)
    init
  }

  private val eLayer = new LinearImpl(memDim, 1)
  private val aLayer = new LinearImpl(memDim, 1)
  private val fLayer = new LinearImpl(memDim * 2, memDim)
  register_module("e_layer", eLayer)
  register_module("a_layer", aLayer)
  register_module("f_layer", fLayer)

  // IRT parameters: ability and difficulty estimation
  private val diffLayer = new LinearImpl(memDim, 1)
  private val abilityLayer = new LinearImpl(memDim, 1)
  register_module("diff_layer", diffLayer)
  register_module("ability_layer", abilityLayer)

  private val dropoutLayer = new DropoutImpl(dropout)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    this.to(dev, false)
  }

  def forward(conceptIds: Tensor, responses: Tensor): Tensor = {
    val batchSize = conceptIds.size(0).toInt
    val seqLen = conceptIds.size(1).toInt

    val k = kEmb.forward(conceptIds)
    // Interaction IDs: concept * 2 + response (0=incorrect, 1=correct for each concept)
    val interactionIds = conceptIds.mul(new Scalar(2.0)).add(responses).toType(ScalarType.Long)
    val v = vEmb.forward(interactionIds)

    // Expand memory to batch size: (memSize, memDim) -> (batchSize, memSize, memDim)
    // Detach and clone to avoid gradient issues with expanded parameter
    val mkBatch = mkInit.clone().view(1L, memSize.toLong, memDim.toLong).expand(batchSize.toLong, -1L, -1L).detach()
    var mv = mvInit.clone().view(1L, memSize.toLong, memDim.toLong).expand(batchSize.toLong, -1L, -1L).clone()
    val preds = mutable.ArrayBuffer[Tensor]()

    for (t <- 0 until seqLen) {
      val kt = k.select(1, t)  // (batch, memDim)
      val vt = v.select(1, t)

      // Detach mv to avoid gradient issues through the loop
      val mvCurrent = if (t == 0) mv else mv.detach()

      // Compute attention: (batch, 1, memDim) @ (batch, memDim, memSize) -> (batch, 1, memSize) -> (batch, memSize)
      val scores = torch.bmm(kt.unsqueeze(1), mkBatch.transpose(1, 2)).squeeze(1)
      val attn = scores.softmax(1)

      // Erase: erase = 1 - attn * erase_gate
      val eraseGate = torch.sigmoid(eLayer.forward(kt)).squeeze(1)  // (batch,)
      val attnEx = attn.unsqueeze(2)  // (batch, memSize, 1)
      val eraseGateEx = eraseGate.unsqueeze(1).unsqueeze(2)  // (batch, 1, 1)
      val attnScaled = attnEx.mul(eraseGateEx)  // (batch, memSize, 1)
      val eraseFactor = attnScaled.neg().add(new Scalar(1.0)).contiguous()  // (batch, memSize, 1)
      val newMv = mvCurrent.mul(eraseFactor)  // (batch, memSize, memDim)

      // Add: add = attn * add_gate
      val addGate = torch.tanh(aLayer.forward(kt)).squeeze(1)  // (batch,)
      val addGateEx = addGate.unsqueeze(1).unsqueeze(2)  // (batch, 1, 1)
      val addScaled = attnEx.mul(addGateEx)  // (batch, memSize, 1)
      mv = newMv.add(addScaled)  // (batch, memSize, memDim)

      // Read: combine attention-weighted memory with key
      val readMem = (attn.unsqueeze(2).mul(mv)).sum(1)  // (batch, memDim)
      val combined = torch.cat(new TensorVector(readMem, kt), 1)
      val fused = torch.tanh(fLayer.forward(combined))

      // IRT: P(correct) = sigmoid(3 * ability - difficulty)
      val ability = torch.tanh(abilityLayer.forward(dropoutLayer.forward(fused)))
      val difficulty = torch.tanh(diffLayer.forward(dropoutLayer.forward(kt)))
      val pred = torch.sigmoid(ability.mul(new Scalar(3.0)).sub(difficulty))

      preds += pred.squeeze(1)
    }

    if (preds.isEmpty) {
      torch.zeros(batchSize, seqLen)
    } else {
      torch.stack(new TensorVector(preds.toSeq: _*), 1)
    }
  }

  def predict(conceptIds: Tensor, responses: Tensor): Tensor = forward(conceptIds, responses)
}
