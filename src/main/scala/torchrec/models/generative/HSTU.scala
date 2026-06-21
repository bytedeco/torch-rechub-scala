package torchrec.models.generative

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.basic.layers.HSTUBlock
import torchrec.utils.DeviceSupport

/**
 * HSTU: Hierarchical Sequential Transduction Units
 *
 * Autoregressive generative recommender that stacks HSTUBlock layers to capture
 * long-range dependencies and predict the next item.
 *
 * Reference: Meta, 2024 - "Generative Recommenders: Learning Identity and
 * Denoising from User-Item Interactions"
 *
 * This implementation mirrors the Python torch_rechub.basic.layers.HSTUModel class,
 * using the provided HSTUBlock and HSTULayer submodules.
 *
 * Parameters
 * ----------
 * vocabSize: Long - Vocabulary size (items incl. PAD=0)
 * dModel: Int - Hidden dimension (default: 512)
 * nHeads: Int - Number of attention heads (default: 8)
 * nLayers: Int - Number of stacked HSTU layers (default: 4)
 * dqk: Int - Query/key dim per head (default: 64)
 * dv: Int - Value/u dim per head (default: 64)
 * maxSeqLen: Int - Maximum sequence length (default: 256)
 * dropout: Float - Dropout rate (default: 0.1f)
 * useTimeEmbedding: Boolean - Use time-difference embeddings (default: true)
 * numTimeBuckets: Int - Number of time buckets (default: 128)
 * timeBucketFn: String - Bucketization function "sqrt" or "log" (default: "sqrt")
 * timeBucketDivisor: Float - Divisor for time bucketization (default: 1.0f)
 * timeBucketUnit: String - Time unit "minutes" or "seconds" (default: "minutes")
 * tieEmbeddings: Boolean - Tie output projection with token embedding (default: true)
 * scoreNorm: String - Score normalization "none" or "l2" (default: "none")
 * temperature: Float - Temperature for logits scaling (default: 1.0f)
 * useOutputBias: Boolean - Include bias in output logits (default: true)
 * scaleInputEmbedding: Boolean - Scale token embeddings by sqrt(d_model) (default: false)
 * l2NormEps: Float - Epsilon for L2 normalization (default: 1e-6f)
 *
 * Input Shape
 * -----------
 * tokens: (batch_size, seq_len)
 * timeDiffs: (batch_size, seq_len) - optional, time differences in seconds
 *
 * Output Shape
 * ------------
 * logits: (batch_size, seq_len, vocabSize)
 */
class HSTU(
  vocabSize: Long,
  dModel: Int = 512,
  nHeads: Int = 8,
  nLayers: Int = 4,
  dqk: Int = 64,
  dv: Int = 64,
  maxSeqLen: Int = 256,
  dropout: Float = 0.1f,
  useTimeEmbedding: Boolean = true,
  numTimeBuckets: Int = 128,
  timeBucketFn: String = "sqrt",
  timeBucketDivisor: Float = 1.0f,
  timeBucketUnit: String = "minutes",
  tieEmbeddings: Boolean = true,
  scoreNorm: String = "none",
  temperature: Float = 1.0f,
  useOutputBias: Boolean = true,
  scaleInputEmbedding: Boolean = false,
  l2NormEps: Float = 1e-6f,
  device: String = DeviceSupport.backend
) extends Module {

  require(vocabSize > 0, "vocabSize must be positive")
  require(dModel > 0, "dModel must be positive")
  require(nHeads > 0, "nHeads must be positive")
  require(dModel % nHeads == 0, s"dModel ($dModel) must be divisible by nHeads ($nHeads)")
  require(scoreNorm == "none" || scoreNorm == "l2", "scoreNorm must be 'none' or 'l2'")
  require(temperature > 0, "temperature must be positive")
  require(timeBucketFn == "sqrt" || timeBucketFn == "log", "timeBucketFn must be 'sqrt' or 'log'")
  require(timeBucketUnit == "minutes" || timeBucketUnit == "seconds", "timeBucketUnit must be 'minutes' or 'seconds'")

  // Token embeddings with padding_idx=0 (PAD token embedding stays zero)
  private val tokenEmbeddingOptions = {
    val opts = new EmbeddingOptions(vocabSize, dModel)
    opts.padding_idx().put(0L)
    opts
  }
  private val tokenEmbedding = new EmbeddingImpl(tokenEmbeddingOptions)
  tokenEmbedding.to(new Device(device), false)
  register_module("tokenEmbedding", tokenEmbedding)

  // Position embeddings
  private val positionEmbedding = {
    val opts = new EmbeddingOptions(maxSeqLen, dModel)
    new EmbeddingImpl(opts)
  }
  positionEmbedding.to(new Device(device), false)
  register_module("positionEmbedding", positionEmbedding)

  // Time difference embeddings (bucket 0 is the smallest legal time-diff bucket, not PAD)
  private val timeEmbedding = {
    val opts = new EmbeddingOptions(numTimeBuckets, dModel)
    val emb = new EmbeddingImpl(opts)
    emb.to(new Device(device), false)
    emb
  }
  if (useTimeEmbedding) {
    register_module("timeEmbedding", timeEmbedding)
  }

  // HSTU block using the provided submodule (stacks HSTULayer with residual connections)
  private val hstuBlock = HSTUBlock(
    dModel = dModel,
    nHeads = nHeads,
    nLayers = nLayers,
    dqk = dqk,
    dv = dv,
    dropout = dropout,
    maxSeqLen = maxSeqLen,
    numTimeBuckets = numTimeBuckets,
    timeBucketFn = timeBucketFn,
    timeBucketDivisor = timeBucketDivisor,
    timeBucketUnit = timeBucketUnit,
    device = device
  )
  register_module("hstuBlock", hstuBlock)

  // Output projection
  // If tieEmbeddings=true, we reuse tokenEmbedding.weight and only add a bias parameter
  // If tieEmbeddings=false, we use a separate Linear layer
  private val (outputProjection, outputBias) = {
    if (tieEmbeddings) {
      (None, None) // Will be registered separately if needed
    } else {
      val proj = new LinearImpl(dModel, vocabSize)
      proj.to(new Device(device), false)
      register_module("outputProj", proj)
      (Some(proj), None)
    }
  }

  // Output bias parameter (only used when tieEmbeddings=true)
//  private val outputBiasParam: Option[Tensor] = {
//    if (tieEmbeddings && useOutputBias) {
//      val biasTensor = new Tensor(vocabSize)
//      biasTensor.zero_()
//      biasTensor.to(new Device(device), biasTensor.dtype())
//      Some(biasTensor)
//    } else {
//      None
//    }
//  }

  // Output bias parameter (only used when tieEmbeddings=true)
  private val outputBiasParam: Option[Tensor] = {
    if (tieEmbeddings && useOutputBias) {
      // ✅ Use torch.zeros to properly allocate a 1D tensor
      val biasTensor = torch.zeros(
        Array(vocabSize),
        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
      ).to(new Device(device), ScalarType.Float)

      Some(biasTensor)
    } else {
      None
    }
  }
  if (outputBiasParam.isDefined) {
    register_parameter("outputBias", outputBiasParam.get)
  }

  // Dropout layer
  private val dropoutLayer = new DropoutImpl(dropout)

  // Initialize weights
  _initWeights()

  private def _initWeights(): Unit = {
    // Xavier-uniform for 2D weights (tokenEmbedding, positionEmbedding)
    val teWeight = tokenEmbedding.weight()
    if (teWeight.dim() > 1) {
      torch.xavier_uniform_(teWeight)
    }

    val peWeight = positionEmbedding.weight()
    if (peWeight.dim() > 1) {
      torch.xavier_uniform_(peWeight)
    }

    // Force padding_idx=0 row of tokenEmbedding back to zero
    // (xavier pass overwrites the zero row that nn.Embedding sets at construction time)
    val idxTensor = torch.tensor(Array(0L), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
//    teWeight.index_fill_(0, idxTensor, new Scalar(0f)) //todo do we need to do this after every epoch?

    // Initialize output bias if present
    if (tieEmbeddings && useOutputBias && outputBiasParam.isDefined) {
      torch.constant_(outputBiasParam.get, new Scalar(0.0f))
    }
  }

  /**
   * Map raw time differences (seconds) to bucket indices.
   *
   * Time deltas are optionally converted to minutes, passed through sqrt or log,
   * then divided by timeBucketDivisor and clipped to [0, numTimeBuckets-1].
   */
  private def _timeDiffToBucket(timeDiffs: Tensor): Tensor = {
    var buckets = timeDiffs.toType(ScalarType.Float)

    if (timeBucketUnit == "minutes") {
      buckets = buckets.div(new Scalar(60.0f))
    }

    buckets = torch.clamp(buckets, min = new ScalarOptional(new Scalar(1e-6f)))

    if (timeBucketFn == "sqrt") {
      buckets = torch.sqrt(buckets)
    } else if (timeBucketFn == "log") {
      buckets = torch.log(buckets)
    }

    buckets = (buckets.div(new Scalar(timeBucketDivisor.toFloat)))
      .clamp(min = new ScalarOptional(new Scalar(0.0f)) , max = new ScalarOptional(new Scalar((numTimeBuckets - 1).toFloat)))

    buckets.toType(ScalarType.Long)
  }

  /**
   * Forward pass.
   *
   * @param tokens Input token ids, shape (batch_size, seq_len). 0 is treated as PAD.
   * @param timeDiffs Optional time differences in seconds, shape (batch_size, seq_len).
   *                  If None and useTimeEmbedding=true, all-zero deltas are used.
   * @return Logits over the vocabulary, shape (batch_size, seq_len, vocabSize).
   */
  def forward(
    tokens: Tensor,
    timeDiffs: Tensor
  ): Tensor = {
    // Ensure inputs are on the same device as the model to avoid mixed-device ops
    val dev = new Device(device)
    val tokensOn = try { tokens.to(dev, ScalarType.Long) } catch { case _: Throwable => tokens }
    var timeDiffsOn: Option[Tensor] = None
    if (timeDiffs != null) {
      try { timeDiffsOn = Some(timeDiffs.to(dev, ScalarType.Float)) } catch { case _: Throwable => timeDiffsOn = Some(timeDiffs) }
    }

    // Ensure the module parameters and submodules are on the requested device
    try { this.to(dev, false) } catch { case _: Throwable => () }

    val batchSize = tokensOn.size(0).toInt
    val seqLen = tokensOn.size(1).toInt

    // Validate sequence length
    if (seqLen > maxSeqLen) {
      throw new IllegalArgumentException(
        s"Input seq_len ($seqLen) exceeds max_seq_len ($maxSeqLen). " +
        s"Either truncate the input or rebuild the model with a larger max_seq_len."
      )
    }

    // Create padding mask: true for valid tokens
    val paddingMask = torch.ne(tokensOn, new Scalar(0L))

    // Token embeddings (ensure indices are long and on model device)
    var embeddings = tokenEmbedding.forward(tokensOn.toType(ScalarType.Long))

    if (scaleInputEmbedding) {
      embeddings = embeddings.mul(new Scalar(math.sqrt(dModel).toFloat))
    }

    // Position embeddings: create positions [0, 1, 2, ..., seqLen-1]
    // Position indices must be created on the model device
    val positions = torch.arange(
      new Scalar(seqLen.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long))
    ).to(dev, ScalarType.Long)

    var posEmb = positionEmbedding.forward(positions.toType(ScalarType.Long))
    // Ensure position embeddings are on same device as token embeddings before adding
    try { posEmb = posEmb.to(embeddings.device(), posEmb.dtype()) } catch { case _: Throwable => () }
    embeddings = embeddings.add(posEmb.unsqueeze(0))

    // Time difference embeddings
    if (useTimeEmbedding) {
      val effectiveTimeDiffs = if (timeDiffsOn.isDefined) {
        timeDiffsOn.get
      } else {
        // allocate zeros on the model device
        torch.zeros(Array(batchSize.toLong, seqLen.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
          .to(dev, ScalarType.Long)
      }

      val timeBuckets = _timeDiffToBucket(effectiveTimeDiffs)
      var timeEmb = timeEmbedding.forward(timeBuckets)
      // Move timeEmb to embeddings device if necessary
      try { timeEmb = timeEmb.to(embeddings.device(), timeEmb.dtype()) } catch { case _: Throwable => () }
      embeddings = embeddings.add(timeEmb)
    }

    // Zero out padded positions to prevent position/time embeddings from leaking through PAD rows
    var maskExpand = paddingMask.unsqueeze(-1).to(embeddings.dtype())
    try { maskExpand = maskExpand.to(embeddings.device(), maskExpand.dtype()) } catch { case _: Throwable => () }
    embeddings = embeddings.mul(maskExpand)

    // Apply dropout
    embeddings = dropoutLayer.forward(embeddings)

    // HSTU block forward
    var hstuOutput = hstuBlock.forward(embeddings, Some(paddingMask),
      if (timeDiffsOn.isDefined) Some(timeDiffsOn.get) else None)

    // Zero out padded positions in output
    hstuOutput = hstuOutput.mul(paddingMask.unsqueeze(-1).to(hstuOutput.dtype()))

    // Get output weight and bias
    val (outputWeight, outputBiasOpt) = {
      if (tieEmbeddings) {
        (tokenEmbedding.weight(), outputBiasParam)
      } else {
        (outputProjection.get.weight(), outputBias)
      }
    }

    // Apply score normalization (L2 norm)
    var finalOutput = hstuOutput
    var finalWeight = outputWeight
    if (scoreNorm == "l2") {
      val opt = new NormalizeFuncOptions()
      opt.p().put(2) // p=2 for L2 norm
      opt.dim().put(-1) // Normalize across the last dimension
      opt.eps().put(l2NormEps)
      finalOutput = torch.normalize(finalOutput, opt)
      finalWeight = torch.normalize(finalWeight, opt)
    }

    // Compute logits: F.linear(hstu_output, output_weight, output_bias)
    val logits = torch.linear(finalOutput, finalWeight, outputBiasOpt.orNull)

    // Apply temperature scaling
    if (temperature != 1.0f) {
      logits.div(new Scalar(temperature))
    }

    logits
  }
}

/**
 * HSTU companion object with factory methods.
 */
object HSTU {
  def apply(
    vocabSize: Long,
    dModel: Int = 512,
    nHeads: Int = 8,
    nLayers: Int = 4,
    dqk: Int = 64,
    dv: Int = 64,
    maxSeqLen: Int = 256,
    dropout: Float = 0.1f,
    useTimeEmbedding: Boolean = true,
    numTimeBuckets: Int = 128,
    timeBucketFn: String = "sqrt",
    timeBucketDivisor: Float = 1.0f,
    timeBucketUnit: String = "minutes",
    tieEmbeddings: Boolean = true,
    scoreNorm: String = "none",
    temperature: Float = 1.0f,
    useOutputBias: Boolean = true,
    scaleInputEmbedding: Boolean = false,
    l2NormEps: Float = 1e-6f,
    device: String = DeviceSupport.backend
  ): HSTU = {
    new HSTU(
      vocabSize, dModel, nHeads, nLayers, dqk, dv, maxSeqLen, dropout,
      useTimeEmbedding, numTimeBuckets, timeBucketFn, timeBucketDivisor,
      timeBucketUnit, tieEmbeddings, scoreNorm, temperature, useOutputBias,
      scaleInputEmbedding, l2NormEps, device
    )
  }
}