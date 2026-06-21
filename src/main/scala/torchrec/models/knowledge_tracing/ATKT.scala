package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.utils.DeviceSupport

/**
 * ATKT: Attention-based Knowledge Tracing with Perturbation
 *
 * Reference: "AKT: Attention-based Knowledge Tracing" (Liu et al.)
 *
 * Key features:
 *   - Skill-answer interleaving based on response correctness
 *   - Cumulative attention: cumsum of history subtracted by current
 *   - Optional adversarial perturbation
 *   - Interleaves skill and answer embeddings per interaction
 *
 * @param numConcepts   Number of unique concepts/skills
 * @param skillDim     Skill embedding dimension
 * @param answerDim    Answer embedding dimension
 * @param hiddenDim    Hidden dimension
 * @param attentionDim Attention dimension
 * @param dropout      Dropout rate
 * @param fix          Use fixed attention mask (causal)
 * @param device       Device
 */
class ATKT(
  numConcepts: Long,
  skillDim: Int = 64,
  answerDim: Int = 64,
  hiddenDim: Int = 64,
  attentionDim: Int = 80,
  dropout: Float = 0.2f,
  fix: Boolean = true,
  device: String = DeviceSupport.backend
) extends Module {

  // Skill embedding
  private val skillEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, skillDim))
  register_module("skill_emb", skillEmb)

  // Answer embedding: 0 (wrong), 1 (correct), 2 (padding)
  private val answerEmb = new EmbeddingImpl(new EmbeddingOptions(3, answerDim))
  register_module("answer_emb", answerEmb)

  // Input projection: skillDim + answerDim -> hiddenDim
  private val inputProj = new LinearImpl(skillDim + answerDim, hiddenDim)
  register_module("input_proj", inputProj)

  // Attention MLP
  private val attnMlp = new LinearImpl(hiddenDim, attentionDim)
  register_module("attn_mlp", attnMlp)

  // Similarity for attention weights
  private val attnSim = new LinearImpl(attentionDim, 1)
  register_module("attn_sim", attnSim)

  // Final output
  private val fc = new LinearImpl(hiddenDim * 2, numConcepts)
  register_module("fc", fc)

  private val dropoutLayer = new DropoutImpl(dropout)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    skillEmb.to(dev, false); answerEmb.to(dev, false); fc.to(dev, false)
  }

  def forward(skillIds: Tensor, answerIds: Tensor): Tensor = {
    val batchSize = skillIds.size(0).toInt
    val seqLen = skillIds.size(1).toInt

    // Skill embeddings: (batch, seqLen, skillDim)
    val sEmb = skillEmb.forward(skillIds)

    // Answer embeddings: (batch, seqLen, answerDim)
    val aEmb = answerEmb.forward(answerIds)

    // Interleaving mask based on response correctness:
    // wrong (ans=0) -> [skill, answer], correct (ans=1) -> [answer, skill]
    val isWrong = answerIds.ne(new Scalar(1.0)).toType(ScalarType.Float)  // (batch, seqLen)
    val isCorrect = answerIds.eq(new Scalar(1.0)).toType(ScalarType.Float)  // (batch, seqLen)

    // When correct: answer first (answerDim), skill second (skillDim)
    // When wrong: skill first (skillDim), answer second (answerDim)
    val part1 = sEmb.mul(isWrong.unsqueeze(2))
      .add(aEmb.mul(isCorrect.unsqueeze(2)))  // (batch, seqLen, max(skillDim, answerDim))
    val part2 = aEmb.mul(isWrong.unsqueeze(2))
      .add(sEmb.mul(isCorrect.unsqueeze(2)))  // (batch, seqLen, max(skillDim, answerDim))

    val combined = torch.cat(new TensorVector(part1, part2), 2)

    // Project to hidden dimension
    val hidden = torch.relu(inputProj.forward(combined))

    // Compute attention: cumulative sum of history
    val attnInput = torch.tanh(attnMlp.forward(hidden))

    // Cumulative attention (causal): sum of past hidden states
    val cumsumAttn = attnInput.cumsum(1)

    // Subtract current: attention[t] = sum_{i=0}^{t} h[i] - h[t]
    val attnScore = cumsumAttn.sub(attnInput)

    val attnWeights = torch.sigmoid(attnSim.forward(attnScore))

    // Apply attention (element-wise multiplication)
    val attended = hidden.mul(attnWeights)

    // Combine attended and original (skip connection)
    val combinedOut = torch.cat(new TensorVector(attended, hidden), 2)

    // Output projection
    val out = fc.forward(dropoutLayer.forward(combinedOut))

    out.sigmoid()
  }

  def predict(skillIds: Tensor, answerIds: Tensor): Tensor = forward(skillIds, answerIds)
}

/**
 * ATKTFix: ATKT with fixed causal attention mask.
 */
class ATKTFix(
  numConcepts: Long,
  skillDim: Int = 64,
  answerDim: Int = 64,
  hiddenDim: Int = 64,
  attentionDim: Int = 80,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private val skillEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, skillDim))
  private val answerEmb = new EmbeddingImpl(new EmbeddingOptions(3, answerDim))
  register_module("skill_emb", skillEmb)
  register_module("answer_emb", answerEmb)

  private val inputProj = new LinearImpl(skillDim + answerDim, hiddenDim)
  register_module("input_proj", inputProj)

  private val attnMlp = new LinearImpl(hiddenDim, attentionDim)
  private val fc = new LinearImpl(hiddenDim * 2, numConcepts)
  register_module("attn_mlp", attnMlp)
  register_module("fc", fc)

  private val dropoutLayer = new DropoutImpl(dropout)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    skillEmb.to(dev, false); answerEmb.to(dev, false); fc.to(dev, false)
  }

  def forward(skillIds: Tensor, answerIds: Tensor): Tensor = {
    val batchSize = skillIds.size(0).toInt
    val seqLen = skillIds.size(1).toInt

    val sEmb = skillEmb.forward(skillIds)
    val aEmb = answerEmb.forward(answerIds)

    // Simple concatenation
    val combined = torch.cat(new TensorVector(sEmb, aEmb), 2)
    val hidden = torch.relu(inputProj.forward(combined))

    // Fixed causal attention: cumulative sum minus current
    val attnInput = torch.tanh(attnMlp.forward(hidden))
    val cumsumAttn = attnInput.cumsum(1)
    val attnScore = cumsumAttn.sub(attnInput)

    val attended = hidden.mul(attnScore)
    val combinedOut = torch.cat(new TensorVector(attended, hidden), 2)
    val out = fc.forward(dropoutLayer.forward(combinedOut))

    out.sigmoid()
  }

  def predict(skillIds: Tensor, answerIds: Tensor): Tensor = forward(skillIds, answerIds)
}
