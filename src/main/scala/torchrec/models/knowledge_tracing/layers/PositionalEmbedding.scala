package torchrec.models.knowledge_tracing.layers

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.utils.DeviceSupport

/**
 * Cosine positional embedding used by AKT, SimpleKT, SparseKT, SAKT.
 */
class CosinePositionalEmbedding(
  embedDim: Int,
  maxLen: Int = 512,
  device: String = DeviceSupport.backend
) extends Module {

  // Create position embeddings as a buffer (non-learnable)
  private val pe = {
    val pe = torch.zeros(Array(1L, maxLen.toLong, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    val position = torch.arange(new Scalar(0), new Scalar(maxLen.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .unsqueeze(1)
    val divTerm = torch.exp(torch.arange(new Scalar(0), new Scalar(embedDim.toLong), new Scalar(2L)).mul(new Scalar(-scala.math.log(10000.0))).div(new Scalar(embedDim.toDouble)))
    val peArr = new Array[Float](maxLen * embedDim)
    for (i <- 0 until maxLen) {
      for (j <- 0 until embedDim) {
        val idx = i * embedDim + j
        if (j % 2 == 0) {
          peArr(idx) = (scala.math.sin(position.select(1, 0).select(0, i).item_float() * divTerm.select(0, j / 2).item_float())).toFloat
        } else {
          peArr(idx) = (scala.math.cos(position.select(1, 0).select(0, i).item_float() * divTerm.select(0, (j - 1) / 2).item_float())).toFloat
        }
      }
    }
    pe.copy_(torch.tensor(peArr*).view(Array(1L, maxLen.toLong, embedDim.toLong)*))
    if (device != "cpu") {
      pe.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
    }
    register_buffer("pe", pe)
    pe
  }

  def forward(x: Tensor): Tensor = {
    val batchSize = x.size(0).toInt
    val seqLen = x.size(1).toInt
    // Handle case where seqLen > maxLen
    val actualLen = seqLen.min(maxLen)
    val result = pe.narrow(1, 0, actualLen).expand(Array(batchSize.toLong, actualLen.toLong, embedDim.toLong)*).clone()
    // If seqLen > maxLen, need to handle differently - for now just return truncated result
    result
  }
}

/**
 * Learnable positional embedding used by SAINT.
 */
class LearnablePositionalEmbedding(
  maxLen: Int,
  embedDim: Int,
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  private val embedding = new EmbeddingImpl(new EmbeddingOptions(maxLen.toLong, embedDim))
  register_module("embedding", embedding)

  private val dropoutLayer = new DropoutImpl(dropout)

  def forward(seqLen: Long): Tensor = {
    val dev = new org.bytedeco.pytorch.Device(device)
    val positions = torch.arange(new Scalar(0), new Scalar(seqLen), new TensorOptions().device(new DeviceOptional(dev)).dtype(new ScalarTypeOptional(ScalarType.Long)))
    dropoutLayer.forward(embedding.forward(positions))
  }
}

/**
 * Interaction embedding: combines concept ID and response into a single ID.
 * Used by DKT, DKVMN, DKTPlus.
 * interaction_id = conceptId + numConcepts * response
 */
class InteractionEmbedding(
  numConcepts: Long,
  embedDim: Int,
  device: String = DeviceSupport.backend
) extends Module {

  // vocab size = numConcepts * 2 + 1 to handle all interaction IDs
  // interaction_id = conceptId + numConcepts * response, max = (numConcepts-1) + numConcepts*1 = numConcepts*2-1
  // Add 1 extra for safety (padding)
  private val vocabSize = numConcepts * 2 + 1
  private val interaction_emb = new EmbeddingImpl(
    new EmbeddingOptions(vocabSize, embedDim)
  )
  register_module("interaction_emb", interaction_emb)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    interaction_emb.to(dev, false)
  }

  def forward(conceptIds: Tensor, responses: Tensor): Tensor = {
    // interaction_id = conceptId + numConcepts * response, clamp to valid range
    val conceptIdsLong = conceptIds.toType(ScalarType.Long)
    val responsesLong = responses.toType(ScalarType.Long)
    val maxId = vocabSize - 1
    val interactionIdsRaw = conceptIdsLong.add(responsesLong.mul(new Scalar(numConcepts.toDouble)))
    // Clamp to valid range, then convert to Long
    val minScalar = new org.bytedeco.pytorch.Scalar(0)
    val maxScalar = new org.bytedeco.pytorch.Scalar(maxId)
    val interactionIdsClamped = interactionIdsRaw.clamp(
      new org.bytedeco.pytorch.ScalarOptional(minScalar),
      new org.bytedeco.pytorch.ScalarOptional(maxScalar)
    ).toType(ScalarType.Long)
    interaction_emb.forward(interactionIdsClamped)
  }
}
