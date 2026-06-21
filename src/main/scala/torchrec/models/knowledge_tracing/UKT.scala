package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.basic.layers.MLP
import torchrec.models.knowledge_tracing.layers.CosinePositionalEmbedding
import torchrec.utils.DeviceSupport

/**
 * UKT: Uncertainty-aware Knowledge Tracing
 *
 * A knowledge tracing model that represents student knowledge states using
 * stochastic embeddings (mean and covariance) to capture uncertainty in learning.
 * Uses Wasserstein-based attention mechanism for knowledge state transitions.
 *
 * Reference: "Uncertainty-aware Knowledge Tracing"
 *
 * Architecture:
 *   Stochastic Embeddings (mean/cov) -> Transformer with Uncertainty Attention -> Prediction
 *
 * @param numConcepts   Number of unique concepts/questions
 * @param embedDim      Model dimension
 * @param numHeads      Number of attention heads
 * @param numBlocks     Number of transformer blocks
 * @param dropout       Dropout rate
 * @param device        Device
 */
class UKT(
  numConcepts: Long,
  embedDim: Int = 64,
  numHeads: Int = 8,
  numBlocks: Int = 2,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)

  // Mean embeddings for concepts
  private val meanQEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim))
  register_module("mean_q_embed", meanQEmbed)

  // Covariance embeddings for concepts
  private val covQEmbed = {
    val t = torch.randn(Array((numConcepts + 1).toLong, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .mul(new Scalar(0.01f))
    register_parameter("cov_q_embed", t)
    t
  }

  // Mean embeddings for QA interactions
  private val meanQAEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2, embedDim))
  register_module("mean_qa_embed", meanQAEmbed)

  // Covariance embeddings for QA interactions
  private val covQAEmbed = {
    val t = torch.randn(Array((numConcepts * 2).toLong, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .mul(new Scalar(0.01f))
    register_parameter("cov_qa_embed", t)
    t
  }

  // Question difficulty parameters
  private val qDiff = {
    val t = torch.randn(Array((numConcepts + 1).toLong, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .mul(new Scalar(0.01f))
    register_parameter("q_diff", t)
    t
  }

  // Positional embedding for mean
  private val posMeanEmbed = new CosinePositionalEmbedding(embedDim, 512, device)
  register_module("pos_mean_embed", posMeanEmbed)

  // Positional embedding for covariance
  private val posCovEmbed = new CosinePositionalEmbedding(embedDim, 512, device)
  register_module("pos_cov_embed", posCovEmbed)

  // Transformer blocks for uncertainty modeling
  private val blocks = (0 until numBlocks).map { i =>
    val block = new UncertaintyTransformerBlock(embedDim, numHeads, dropout, device)
    register_module(s"block_$i", block)
    block
  }

  // Output MLP
  private val outMLP = new MLP(embedDim * 4, List(embedDim * 2, embedDim), 1, "relu", dropout, device = device)
  register_module("out_mlp", outMLP)

  // Dropout
  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropout", dropoutLayer)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    meanQEmbed.to(dev, false); meanQAEmbed.to(dev, false)
    outMLP.to(dev, false)
  }

  /**
   * Forward pass for UKT.
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

    // Get mean embeddings
    // Ensure indices are on the model device and Long dtype
    val cIdsLongDev = cIdsLong.to(new org.bytedeco.pytorch.Device(device), ScalarType.Long)
    val rLongDev = rLong.to(new org.bytedeco.pytorch.Device(device), ScalarType.Long)

    val qMeanEmb = meanQEmbed.forward(cIdsLongDev)  // (batch, seq, embedDim)
    val qaIndex = cIdsLongDev.add(rLongDev.mul(new Scalar(numConcepts.toDouble))).toType(ScalarType.Long).to(new org.bytedeco.pytorch.Device(device), ScalarType.Long)
    val qaMeanEmb = meanQAEmbed.forward(qaIndex)

    // Get covariance embeddings
    val qCovEmb = covQEmbed.index_select(0, cIdsLongDev.view(-1L)).view(batchSize, seqLen, embedDim)
    val qaIndexFlat = qaIndex.view(-1L)
    val qaCovEmb = covQAEmbed.index_select(0, qaIndexFlat).view(batchSize, seqLen, embedDim)

    // Get question difficulty
    val qDiffEmb = qDiff.index_select(0, cIdsLongDev.view(-1L)).view(batchSize, seqLen, embedDim)

    // Add positional encoding
    val qMeanPos = posMeanEmbed.forward(qMeanEmb)
    val qCovPos = posCovEmbed.forward(qCovEmb)

    val qMeanWithPos = qMeanEmb.add(qMeanPos)
    val qCovWithPos = qCovEmb.add(qCovPos).add(new Scalar(1.0))  // ELU + 1 for positivity

    val qaMeanWithPos = qaMeanEmb.add(qMeanPos)
    val qaCovWithPos = qaCovEmb.add(qCovPos).add(new Scalar(1.0))

    // Apply difficulty adjustment
    val qMeanAdjusted = qMeanWithPos.add(qDiffEmb)
    val qCovAdjusted = qCovWithPos.add(qDiffEmb)

    // Process through transformer blocks
    var meanOut = qMeanAdjusted
    var covOut = qCovAdjusted
    blocks.foreach { block =>
      val result = block.forward(meanOut, covOut, qaMeanWithPos, qaCovWithPos)
      meanOut = result._1
      covOut = result._2
    }

    // Apply ELU activation to covariance for uncertainty measure
    val covActivated = torch.elu(covOut).add(new Scalar(1.0))

    // Concatenate features for prediction
    val concatFeatures = torch.cat(new TensorVector(meanOut, covActivated, qMeanEmb, qCovEmb), 2)
    val logits = outMLP.forward(concatFeatures)

    logits.sigmoid().squeeze(2)
  }

  def predict(conceptIds: Tensor, responses: Tensor): Tensor = forward(conceptIds, responses)
}

/**
 * Transformer block with uncertainty (mean and covariance) attention.
 */
class UncertaintyTransformerBlock(
  embedDim: Int,
  numHeads: Int,
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)
  private val headDim = embedDim / numHeads

  // Mean attention
  private val meanQ = new LinearImpl(embedDim, embedDim)
  private val meanK = new LinearImpl(embedDim, embedDim)
  private val meanV = new LinearImpl(embedDim, embedDim)
  register_module("mean_q", meanQ)
  register_module("mean_k", meanK)
  register_module("mean_v", meanV)

  // Covariance attention
  private val covQ = new LinearImpl(embedDim, embedDim)
  private val covK = new LinearImpl(embedDim, embedDim)
  private val covV = new LinearImpl(embedDim, embedDim)
  register_module("cov_q", covQ)
  register_module("cov_k", covK)
  register_module("cov_v", covV)

  // Output projections
  private val meanOutProj = new LinearImpl(embedDim, embedDim)
  private val covOutProj = new LinearImpl(embedDim, embedDim)
  register_module("mean_out_proj", meanOutProj)
  register_module("cov_out_proj", covOutProj)

  // Layer norms
  private def layerNormShape(d: Int) = { val v = new LongVector(1); v.put(0, d.toLong); v }
  private val ln1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
  private val ln2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
  register_module("ln1", ln1)
  register_module("ln2", ln2)

  // FFN
  private val ff1 = new LinearImpl(embedDim, embedDim * 4)
  private val ff2 = new LinearImpl(embedDim * 4, embedDim)
  register_module("ff1", ff1)
  register_module("ff2", ff2)

  private val dropoutLayer = new DropoutImpl(dropout)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    meanQ.to(dev, false); meanK.to(dev, false); meanV.to(dev, false)
    covQ.to(dev, false); covK.to(dev, false); covV.to(dev, false)
    meanOutProj.to(dev, false); covOutProj.to(dev, false)
    ff1.to(dev, false); ff2.to(dev, false)
  }

  def forward(
    qMean: Tensor, qCov: Tensor,
    vMean: Tensor, vCov: Tensor
  ): (Tensor, Tensor) = {
    val batchSize = qMean.size(0).toInt
    val seqLen = qMean.size(1).toInt

    // Project queries, keys, values
    val qM = meanQ.forward(qMean)
    val kM = meanK.forward(vMean)
    val vM = meanV.forward(vMean)

    // Scaled dot-product attention for mean
    val scale = new Scalar(scala.math.sqrt(headDim.toDouble).toFloat)
    var scores = torch.matmul(qM, kM.transpose(1, 2)).div(scale)

    // Apply causal mask
    if (seqLen > 1) {
      val causalMask = torch.triu(torch.ones(Array(seqLen.toLong, seqLen.toLong),
        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))), 1)
        .unsqueeze(0)
      scores = scores.add(causalMask.mul(new Scalar(1e9)))
    }

    val attnWeights = scores.softmax(-1)
    val meanAttn = torch.matmul(dropoutLayer.forward(attnWeights), vM)
    val meanProjected = meanOutProj.forward(meanAttn)

    // Residual connection and layer norm for mean
    val meanWithRes = qMean.add(meanProjected)
    val normedMean = ln1.forward(meanWithRes)

    // FFN for mean
    val ffOut = torch.relu(ff1.forward(normedMean))
    val ffDropped = dropoutLayer.forward(ffOut)
    val ffResult = ff2.forward(ffDropped)
    val meanFinal = ln2.forward(normedMean.add(ffResult))

    // For covariance, use simplified processing
    val covProjected = covOutProj.forward(qCov)
    val covWithRes = qCov.add(covProjected)

    (meanFinal, covWithRes)
  }
}
