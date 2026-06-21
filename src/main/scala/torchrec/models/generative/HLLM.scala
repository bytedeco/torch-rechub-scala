package torchrec.models.generative

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.basic.layers.{EmbeddingLayer, RelativeBucketedTimeAndPositionBias}
import torchrec.basic.features.Feature
import torchrec.utils.DeviceSupport

import scala.math

/**
 * HLLM: Hierarchical Large Language Model for Recommendation.
 *
 * This is a full implementation of the ByteDance HLLM model that uses pre-computed
 * item embeddings as input. The model applies stacked transformer blocks over
 * sequences of item embeddings, with optional time-difference embeddings.
 *
 * Architecture:
 *   - Item Embeddings: Pre-computed using LLM (offline, frozen), normalized once.
 *   - User LLM: Transformer blocks that model user sequences (trainable)
 *   - Scoring Head: Cosine similarity between user representation and item embeddings
 *
 * Reference:
 *   ByteDance HLLM: https://github.com/bytedance/HLLM
 *
 * @param itemEmbeddings Pre-computed item embeddings of shape (vocab_size, d_model),
 *                       already normalized.
 * @param vocabSize Vocabulary size including PAD.
 * @param dModel Hidden dimension. Default: 512.
 * @param nHeads Number of attention heads. Default: 8.
 * @param nLayers Number of transformer blocks. Default: 4.
 * @param maxSeqLen Maximum sequence length. Default: 256.
 * @param dropout Dropout rate. Default: 0.1.
 * @param useRelPosBias Whether to use relative position bias. Default: true.
 * @param useTimeEmbedding Whether to use time embeddings. Default: true.
 * @param numTimeBuckets Number of time buckets. Default: 2048.
 * @param timeBucketFn Time bucketization function ('sqrt' or 'log'). Default: 'sqrt'.
 * @param temperature Temperature for cos-sim logits. Default: 0.07.
 * @param device Device for computation.
 */
class HLLM(
  itemEmbeddings: Tensor,
  vocabSize: Long,
  dModel: Int = 512,
  nHeads: Int = 8,
  nLayers: Int = 4,
  maxSeqLen: Int = 256,
  dropout: Float = 0.1f,
  useRelPosBias: Boolean = true,
  useTimeEmbedding: Boolean = true,
  numTimeBuckets: Int = 2048,
  timeBucketFn: String = "sqrt",
  temperature: Float = 0.07f,
  device: String = DeviceSupport.backend
) extends Module {

  require(vocabSize > 0, "vocabSize must be positive")
  require(dModel > 0, "dModel must be positive")
  require(nHeads > 0, "nHeads must be positive")
  require(dModel % nHeads == 0, s"dModel ($dModel) must be divisible by nHeads ($nHeads)")
  require(timeBucketFn == "sqrt" || timeBucketFn == "log", s"timeBucketFn must be 'sqrt' or 'log'")
  require(temperature > 0, "temperature must be positive")

  private val targetDevice = new Device(device)
  private val headDim = dModel / nHeads
  private val l2NormEps = 1e-8f

  // Validate itemEmbeddings dimensions
  require(itemEmbeddings.size(0) == vocabSize,
    s"item_embeddings.shape[0]=${itemEmbeddings.size(0)} != vocab_size=$vocabSize")
  require(itemEmbeddings.size(1) == dModel,
    s"item_embeddings.shape[1]=${itemEmbeddings.size(1)} != d_model=$dModel")

  // ===========================================================================
  // Frozen item embeddings (buffer, not trainable) — normalized once
  // ===========================================================================
  private val normalizedItemEmbeddings: Tensor = {
    val emb = itemEmbeddings.clone()
    val opt = new NormalizeFuncOptions()
    opt.p().put(2)
    opt.dim().put(-1)
    opt.eps().put(l2NormEps)
    val normalized = torch.normalize(emb.toType(ScalarType.Float), opt)
    // Move to target device
    normalized.to(targetDevice, ScalarType.Float)
  }
  register_buffer("item_embeddings", normalizedItemEmbeddings)

  // ===========================================================================
  // Positional embedding
  // ===========================================================================
  private val positionEmbedding = {
    val opts = new EmbeddingOptions(maxSeqLen, dModel)
    val emb = new EmbeddingImpl(opts)
    emb.to(targetDevice, false)
    emb
  }
  register_module("positionEmbedding", positionEmbedding)

  // ===========================================================================
  // Time embedding (optional) — always defined, registered conditionally
  // ===========================================================================
  private val timeEmbedding: Option[EmbeddingImpl] = {
    if (useTimeEmbedding) {
      val opts = new EmbeddingOptions(numTimeBuckets + 1, dModel)
      opts.padding_idx().put(0L)
      val emb = new EmbeddingImpl(opts)
      emb.to(targetDevice, false)
      Some(emb)
    } else {
      None
    }
  }
  timeEmbedding.foreach(e => register_module("timeEmbedding", e))

  // ===========================================================================
  // Transformer blocks — each registered individually (mirroring nn.ModuleList)
  // ===========================================================================
  private val transformerBlocks: List[HLLMTransformerBlock] = (0 until nLayers).map { i =>
    val block = new HLLMTransformerBlock(dModel, nHeads, dropout, device)
    block.to(targetDevice, false)
    register_module(s"transformer_blocks_$i", block)
    block
  }.toList

  // ===========================================================================
  // ✅ 终极修复：彻底绕过 C++ 底层遍历，纯手动下发状态
  // ===========================================================================
  override def train(on: Boolean): Unit = {
    // ⚠️ 绝对不要调用 super.train(on) ⚠️

    // 手动下发给原生层
    positionEmbedding.train(on)
    timeEmbedding.foreach(_.train(on))
    dropoutLayer.train(on)

    // 手动下发给自定义的 Transformer 块
    transformerBlocks.foreach(_.train(on))

    // 如果 relPosBias 内部只有 Embedding (不受 train/eval 影响)
    // 我们可以安全地包一层 try-catch，防止它内部的 super.train 再次引发崩溃
    try {
      relPosBias.foreach(_.train(on))
    } catch {
      case _: Throwable => // 忽略
    }
  }


  // ===========================================================================
  // Relative position bias
  // ===========================================================================
  private val relPosBias: Option[RelativeBucketedTimeAndPositionBias] = {
    if (useRelPosBias) {
      val rab = new RelativeBucketedTimeAndPositionBias(
        nHeads = nHeads,
        maxSeqLen = maxSeqLen,
        numTimeBuckets = numTimeBuckets,
        timeBucketFn = timeBucketFn
      )
      rab.to(targetDevice, false)
      Some(rab)
    } else {
      None
    }
  }
  relPosBias.foreach(r => register_module("relPosBias", r))

  // ===========================================================================
  // Dropout
  // ===========================================================================
  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropoutLayer", dropoutLayer)
  // ===========================================================================
  // Initialize weights via constructor
  // ===========================================================================
  _initWeights()

  private def _initWeights(): Unit = {
    // Xavier-uniform for positional embedding weights
    val peWeight = positionEmbedding.weight()
    if (peWeight.dim() > 1) {
      torch.xavier_uniform_(peWeight)
    }

    // Initialize transformer block weights
    transformerBlocks.foreach(_.initWeights())
  }

  // ===========================================================================
  // Time bucketization
  // ===========================================================================
  private def _timeDiffToBucket(timeDiffs: Tensor): Tensor = {
    var buckets = timeDiffs.toType(ScalarType.Float)
    buckets = buckets.div(new Scalar(60.0f))
    buckets = torch.clamp(buckets, min = new ScalarOptional(new Scalar(1e-6f)))
    buckets = if (timeBucketFn == "sqrt") torch.sqrt(buckets) else torch.log(buckets)
    buckets.clamp(
      min = new ScalarOptional(new Scalar(0.0f)),
      max = new ScalarOptional(new Scalar((numTimeBuckets - 1).toFloat))
    ).toType(ScalarType.Long)
  }

  // ===========================================================================
  // Forward pass
  // ===========================================================================
  // ===========================================================================
  // Forward pass
  // ===========================================================================
  def forward(
               seqTokens: Tensor,
               timeDiffs: Option[Tensor] = None
             ): Tensor = {
    val batchSize = seqTokens.size(0).toInt
    val seqLen = seqTokens.size(1).toInt

    // 1. Look up item embeddings (修复：展平为 1D 向量后再进行 index_select，然后再 view 恢复形状)
    val flatTokens = seqTokens.contiguous().view(batchSize * seqLen).toType(ScalarType.Long)
    val flatItemEmb = normalizedItemEmbeddings.index_select(0, flatTokens) // [batchSize * seqLen, dModel]
    val itemEmb = flatItemEmb.view(batchSize, seqLen, dModel) // [batchSize, seqLen, dModel]

    // 2. Add positional embedding
    val positions = torch.arange(
      new Scalar(seqLen.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)).device(new DeviceOptional(seqTokens.device()))
    )
    val posEmb = positionEmbedding.forward(positions)
    var embeddings = itemEmb.add(posEmb.unsqueeze(0))

    // 3. Add time embedding if provided
    if (useTimeEmbedding) {
      val td = timeDiffs.getOrElse {
        torch.zeros(
          Array(batchSize.toLong, seqLen.toLong),
          new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)).device(new DeviceOptional(seqTokens.device()))
        )
      }
      val timeBuckets = _timeDiffToBucket(td)
      val timeEmbOpt = timeEmbedding
      if (timeEmbOpt.isDefined) {
        val timeEmb = timeEmbOpt.get.forward(timeBuckets)
        embeddings = embeddings.add(timeEmb)
      }
    }

    // 4. Dropout
    embeddings = dropoutLayer.forward(embeddings)

    // 5. Get relative position bias
    val relPosBiasTensor: Option[Tensor] = relPosBias.map(_.forward(None, seqLen))

    // 6. Pass through transformer blocks
    var x = embeddings
    for (block <- transformerBlocks) {
      x = block.forward(x, relPosBiasTensor)
    }

    // 7. Scoring head: cosine similarity with normalized item embeddings
    val xNormedOpt = new NormalizeFuncOptions()
    xNormedOpt.p().put(2)
    xNormedOpt.dim().put(-1)
    xNormedOpt.eps().put(l2NormEps)
    val xNormed = torch.normalize(x, xNormedOpt)
    val itemEmbTransposed = normalizedItemEmbeddings.t()
    val logits = torch.matmul(xNormed, itemEmbTransposed).div(new Scalar(temperature))

    logits
  }
  def forward2(
    seqTokens: Tensor,
    timeDiffs: Option[Tensor] = None
  ): Tensor = {
    val batchSize = seqTokens.size(0).toInt
    val seqLen = seqTokens.size(1).toInt

    // 1. Look up item embeddings
    val itemEmb = normalizedItemEmbeddings.index_select(0, seqTokens.toType(ScalarType.Long))

    // 2. Add positional embedding
    val positions = torch.arange(
      new Scalar(seqLen.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)).device(new DeviceOptional(seqTokens.device()))
    )
    val posEmb = positionEmbedding.forward(positions)
    var embeddings = itemEmb.add(posEmb.unsqueeze(0))

    // 3. Add time embedding if provided
    if (useTimeEmbedding) {
      val td = timeDiffs.getOrElse {
        torch.zeros(
          Array(batchSize.toLong, seqLen.toLong),
          new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)).device(new DeviceOptional(seqTokens.device()))
        )
      }
      val timeBuckets = _timeDiffToBucket(td)
      val timeEmbOpt = timeEmbedding
      if (timeEmbOpt.isDefined) {
        val timeEmb = timeEmbOpt.get.forward(timeBuckets)
        embeddings = embeddings.add(timeEmb)
      }
    }

    // 4. Dropout
    embeddings = dropoutLayer.forward(embeddings)

    // 5. Get relative position bias
    val relPosBiasTensor: Option[Tensor] = relPosBias.map(_.forward(None, seqLen))

    // 6. Pass through transformer blocks
    var x = embeddings
    for (block <- transformerBlocks) {
      x = block.forward(x, relPosBiasTensor)
    }

    // 7. Scoring head: cosine similarity with normalized item embeddings
    val xNormedOpt = new NormalizeFuncOptions()
    xNormedOpt.p().put(2)
    xNormedOpt.dim().put(-1)
    xNormedOpt.eps().put(l2NormEps)
    val xNormed = torch.normalize(x, xNormedOpt)
    val itemEmbTransposed = normalizedItemEmbeddings.t()
    val logits = torch.matmul(xNormed, itemEmbTransposed).div(new Scalar(temperature))

    logits
  }

  def getItemEmbedding(itemId: Tensor): Tensor = {
    normalizedItemEmbeddings.index_select(0, itemId.toType(ScalarType.Long))
  }
}

// =============================================================================
// HLLMTransformerBlock: Single transformer block with self-attention and FFN
// =============================================================================
class HLLMTransformerBlock(
  dModel: Int = 512,
  nHeads: Int = 8,
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  require(dModel % nHeads == 0, s"dModel ($dModel) must be divisible by nHeads ($nHeads)")

  private val targetDevice = new Device(device)
  private val headDim = dModel / nHeads
  private val scale = math.pow(headDim, -0.5).toFloat

  // Multi-head self-attention linear layers
  private val W_Q: LinearImpl = {
    val l = new LinearImpl(dModel, dModel)
    l.to(targetDevice, false)
    l
  }
  register_module("W_Q", W_Q)

  private val W_K: LinearImpl = {
    val l = new LinearImpl(dModel, dModel)
    l.to(targetDevice, false)
    l
  }
  register_module("W_K", W_K)

  private val W_V: LinearImpl = {
    val l = new LinearImpl(dModel, dModel)
    l.to(targetDevice, false)
    l
  }
  register_module("W_V", W_V)

  private val W_O: LinearImpl = {
    val l = new LinearImpl(dModel, dModel)
    l.to(targetDevice, false)
    l
  }
  register_module("W_O", W_O)

  // Feed-forward network
  private val ffnHidden = 4 * dModel
  // 1. 显式声明并持有强引用，防止 GC 回收！
  private val ffnLinear1 = new LinearImpl(dModel, ffnHidden)
  private val ffnRelu = new ReLUImpl()
  private val ffnDrop1 = new DropoutImpl(dropout)
  private val ffnLinear2 = new LinearImpl(ffnHidden, dModel)
  private val ffnDrop2 = new DropoutImpl(dropout)

  private val ffn: SequentialImpl = {
    val seq = new SequentialImpl()
    // 2. 使用命名 push_back，传入已有强引用的层
    seq.push_back("ffn_lin1", ffnLinear1)
    seq.push_back("ffn_relu", ffnRelu)
//    seq.push_back("ffn_drop1", ffnDrop1) //todo  here will make crash
    seq.push_back("ffn_lin2", ffnLinear2)
//    seq.push_back("ffn_drop2", ffnDrop2) //todo  here will make crash
    seq.to(targetDevice, false)
    seq
  }
  register_module("ffn", ffn)
//  private val ffn: SequentialImpl = {
//    val seq = new SequentialImpl()
//    seq.push_back(new LinearImpl(dModel, ffnHidden))
//    seq.push_back(new ReLUImpl())
//    seq.push_back(new DropoutImpl(dropout))
//    seq.push_back(new LinearImpl(ffnHidden, dModel))
//    seq.push_back(new DropoutImpl(dropout))
//    seq.to(targetDevice, false)
//    seq
//  }
//  register_module("ffn", ffn)

  // Layer normalization
  private val norm1: LayerNormImpl = {
    val vec = new LongVector(1)
    vec.put(0, dModel.toLong)
    val ln = new LayerNormImpl(vec)
    ln.to(targetDevice, false)
    ln
  }
  register_module("norm1", norm1)

  private val norm2: LayerNormImpl = {
    val vec = new LongVector(1)
    vec.put(0, dModel.toLong)
    val ln = new LayerNormImpl(vec)
    ln.to(targetDevice, false)
    ln
  }
  register_module("norm2", norm2)

  // Dropout on attention weights
//  private val attnDropout = new DropoutImpl(dropout)

  private val attnDropout = new DropoutImpl(dropout)
  register_module("attnDropout", attnDropout) // ✅ 补上注册，保证验证时关闭 Attention Dropout
  // Initialize block weights
  def initWeights(): Unit = {
    // 3. 直接使用 ffnLinear1 和 ffnLinear2，不再使用 ffn.get().asInstanceOf
    List(W_Q, W_K, W_V, W_O, ffnLinear1, ffnLinear2).foreach { linear =>
      val weight = linear.weight()
      if (weight.dim() > 1) torch.xavier_uniform_(weight)
      val bias = linear.bias()
      if (bias != null) torch.constant_(bias, new Scalar(0.0f))
    }
  }
  def initWeights2(): Unit = {
    List(W_Q, W_K, W_V, W_O).foreach { linear =>
      val weight = linear.weight()
      if (weight.dim() > 1) torch.xavier_uniform_(weight)
      val bias = linear.bias()
      if (bias != null) torch.constant_(bias, new Scalar(0.0f))
    }
    // FFN: SequentialImpl has get(i) to access modules by index
    val ffnLinear1 = ffn.get(0).asInstanceOf[LinearImpl]
    val ffnLinear2 = ffn.get(3).asInstanceOf[LinearImpl]
    List(ffnLinear1, ffnLinear2).foreach { linear =>
      val weight = linear.weight()
      if (weight.dim() > 1) torch.xavier_uniform_(weight)
      val bias = linear.bias()
      if (bias != null) torch.constant_(bias, new Scalar(0.0f))
    }
  }

  def forward(x: Tensor, relPosBias: Option[Tensor] = None): Tensor = {
    val batchSize = x.size(0).toInt
    val seqLen = x.size(1).toInt

    // Self-attention with residual connection
    val residual = x
    val normedX = norm1.forward(x)

    // Project to Q, K, V
    val Q = W_Q.forward(normedX)
    val K = W_K.forward(normedX)
    val V = W_V.forward(normedX)

    // Reshape for multi-head attention
    val QReshaped = Q.view(batchSize, seqLen, nHeads, headDim).transpose(1, 2)
    val KReshaped = K.view(batchSize, seqLen, nHeads, headDim).transpose(1, 2)
    val VReshaped = V.view(batchSize, seqLen, nHeads, headDim).transpose(1, 2)

    // Attention scores
    val scores = torch.matmul(QReshaped, KReshaped.transpose(-2, -1)).mul(new Scalar(scale))

    // Causal mask
    val causalMask = torch.tril(
      torch.ones(Array(seqLen.toLong, seqLen.toLong),
        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Bool))).to(x.device(), ScalarType.Bool)
    )
    val validMask = causalMask.unsqueeze(0).unsqueeze(0)

    // Apply causal mask
    val maskedScores = scores.masked_fill(
      torch.logical_not(validMask),
      new Scalar(Double.NegativeInfinity)
    )

    // Add relative position bias if provided
    val scoresWithBias = if (relPosBias.isDefined) maskedScores.add(relPosBias.get) else maskedScores

    // Softmax and dropout
    val attnWeights = attnDropout.forward(torch.silu(scoresWithBias).softmax(-1))

    // Attention output
    val attnOutput = torch.matmul(attnWeights, VReshaped)
    val attnOutputReshaped = attnOutput.transpose(1, 2).contiguous().view(batchSize, seqLen, dModel)
    val attnOutProj = W_O.forward(attnOutputReshaped)
    val attnOutFinal = residual.add(attnOutProj)

    // FFN with residual connection
    val residual2 = attnOutFinal
    val normed2 = norm2.forward(attnOutFinal)
    val ffnOut = ffn.forward(normed2)
    residual2.add(ffnOut)
  }
}

object HLLMTransformerBlock {
  def apply(
    dModel: Int = 512,
    nHeads: Int = 8,
    dropout: Float = 0.1f,
    device: String = DeviceSupport.backend
  ): HLLMTransformerBlock = {
    new HLLMTransformerBlock(dModel, nHeads, dropout, device)
  }
}

object HLLM {
  def apply(
    itemEmbeddings: Tensor,
    vocabSize: Long,
    dModel: Int = 512,
    nHeads: Int = 8,
    nLayers: Int = 4,
    maxSeqLen: Int = 256,
    dropout: Float = 0.1f,
    useRelPosBias: Boolean = true,
    useTimeEmbedding: Boolean = true,
    numTimeBuckets: Int = 2048,
    timeBucketFn: String = "sqrt",
    temperature: Float = 0.07f,
    device: String = DeviceSupport.backend
  ): HLLM = {
    new HLLM(
      itemEmbeddings, vocabSize, dModel, nHeads, nLayers, maxSeqLen,
      dropout, useRelPosBias, useTimeEmbedding, numTimeBuckets, timeBucketFn,
      temperature, device
    )
  }
}