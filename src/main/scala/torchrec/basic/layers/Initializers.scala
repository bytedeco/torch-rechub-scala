package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Base trait for embedding initializers.
 * All initializers produce an EmbeddingImpl with initialized weights.
 */
trait EmbeddingInitializer {
  def apply(vocabSize: Long, embedDim: Long, paddingIdx: Option[Long] = None): EmbeddingImpl

  protected def createEmbedding(vocabSize: Long, embedDim: Long, paddingIdx: Option[Long]): EmbeddingImpl = {
    val options = new EmbeddingOptions(vocabSize, embedDim)
    paddingIdx.foreach { idx =>
      options.padding_idx().put(idx)
    }
    new EmbeddingImpl(options)
  }

  protected def setPaddingZero2(embed: EmbeddingImpl, paddingIdx: Option[Long]): Unit = {
    paddingIdx.foreach { idx =>
      val weight = embed.weight()
      // 等价 weight[padding_idx, :] = 0
      val idxTensor = torch.tensor(Array(idx), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
      weight.index_fill_(0, idxTensor, new Scalar(0f))
    }
  }
  protected def setPaddingZero(embed: EmbeddingImpl, paddingIdx: Option[Long]): Unit = {
    paddingIdx.foreach { idx =>
      val weight = embed.weight()
      val embedDim = embed.options().embedding_dim().get()// weight.size(1)
      val zeroRow = torch.zeros(Array(embedDim), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
//      weight.index_copy_(0, torch.tensor(Array(idx), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long))), zeroRow) //todo need update
    }
  }
}

/**
 * Initializes embedding weights with a normal distribution.
 *
 * @param mean Mean of the normal distribution (default: 0.0)
 * @param std  Standard deviation of the normal distribution (default: 1.0)
 */
case class RandomNormal(mean: Float = 0.0f, std: Float = 1.0f) extends EmbeddingInitializer {
  override def apply(vocabSize: Long, embedDim: Long, paddingIdx: Option[Long] = None): EmbeddingImpl = {
    val embed = createEmbedding(vocabSize, embedDim, paddingIdx)
//    embed.weight().normal_(mean, std, new GeneratorOptional())
//    val weight = embed.weight().requires_grad_()
//    // 新建临时张量做原地初始化，规避叶子参数in-place报错
//    val temp = torch.empty_like(weight)
    torch.normal_(embed.weight(),mean, std)
//    weight.copy_(temp)
    setPaddingZero(embed, paddingIdx)
    embed
  }
}

/**
 * Initializes embedding weights with a uniform distribution.
 *
 * @param minval Lower bound of the uniform distribution (default: 0.0)
 * @param maxval Upper bound of the uniform distribution (default: 1.0)
 */
case class RandomUniform(minval: Float = 0.0f, maxval: Float = 1.0f) extends EmbeddingInitializer {
  override def apply(vocabSize: Long, embedDim: Long, paddingIdx: Option[Long] = None): EmbeddingImpl = {
    val embed = createEmbedding(vocabSize, embedDim, paddingIdx)
//    embed.weight().uniform_(minval, maxval, new GeneratorOptional())
//    val weight = embed.weight()
    // 新建临时张量做原地初始化，规避叶子参数in-place报错
//    val temp = torch.empty_like(weight)
    torch.uniform_(embed.weight(),minval, maxval)
//    weight.copy_(temp)
    setPaddingZero(embed, paddingIdx)
    embed
  }
}

/**
 * Initializes embedding weights using Xavier normal initialization.
 *
 * The weights are drawn from a normal distribution with std = gain * sqrt(2 / (fan_in + fan_out))
 *
 * Reference: Glorot, X. & Bengio, Y. (2010) "Understanding the difficulty of training deep feedforward neural networks"
 *
 * @param gain Scaling factor for the standard deviation (default: 1.0)
 */
case class XavierNormal(gain: Float = 1.0f) extends EmbeddingInitializer {
  override def apply(vocabSize: Long, embedDim: Long, paddingIdx: Option[Long] = None): EmbeddingImpl = {
    val embed = createEmbedding(vocabSize, embedDim, paddingIdx)
    val weight = embed.weight()
    val fanIn = vocabSize
    val fanOut = embedDim
    val std = gain * math.sqrt(2.0 / (fanIn + fanOut)).toFloat
    torch.normal_(embed.weight(),0.0f, std)
    setPaddingZero(embed, paddingIdx)
    embed
  }
}

/**
 * Initializes embedding weights using Xavier uniform initialization.
 *
 * The weights are drawn from a uniform distribution in the range
 * [-gain * sqrt(6 / (fan_in + fan_out)), gain * sqrt(6 / (fan_in + fan_out))]
 *
 * Reference: Glorot, X. & Bengio, Y. (2010) "Understanding the difficulty of training deep feedforward neural networks"
 *
 * @param gain Scaling factor for the uniform distribution (default: 1.0)
 */
case class XavierUniform(gain: Float = 1.0f) extends EmbeddingInitializer {
  override def apply(vocabSize: Long, embedDim: Long, paddingIdx: Option[Long] = None): EmbeddingImpl = {
    val embed = createEmbedding(vocabSize, embedDim, paddingIdx)
    val weight = embed.weight()
    val fanIn = vocabSize
    val fanOut = embedDim
    val bound = gain * math.sqrt(6.0 / (fanIn + fanOut)).toFloat
    torch.uniform_(embed.weight(),-bound, bound)
    setPaddingZero(embed, paddingIdx)
    embed
  }
}

/**
 * Creates an embedding layer from pretrained weights.
 *
 * @param embeddingWeights 2D array of shape (vocab_size, embed_dim) containing the pretrained weights
 * @param freeze If true, the embeddings will not be updated during training (default: true)
 */
case class Pretrained(embeddingWeights: Array[Array[Float]], freeze: Boolean = true) extends EmbeddingInitializer {
  private val weightTensor: Tensor = {
    val flatWeights = embeddingWeights.flatten
    torch.tensor(flatWeights*).view( Array(embeddingWeights.length.toLong, embeddingWeights(0).length.toLong)*)
  }

  override def apply(vocabSize: Long, embedDim: Long, paddingIdx: Option[Long] = None): EmbeddingImpl = {
    require(vocabSize == weightTensor.size(0), s"vocab_size mismatch: expected ${weightTensor.size(0)}, got $vocabSize")
    require(embedDim == weightTensor.size(1), s"embed_dim mismatch: expected ${weightTensor.size(1)}, got $embedDim")

    val embed = torch.embedding(
      torch.arange(new Scalar(vocabSize), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long))),
      weightTensor,
      padding_idx = if (paddingIdx.isDefined) paddingIdx.get else -1L,
      scale_grad_by_freq = false,
      sparse = false
    )

    val embedImpl = new EmbeddingImpl(new EmbeddingOptions(vocabSize, embedDim))
    embedImpl.weight().copy_(weightTensor)

    if (freeze) {
      embedImpl.weight().set_requires_grad(false)
    }

    paddingIdx.foreach { idx =>
      val zeroRow = torch.zeros(Array(embedDim.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      embedImpl.weight().index_copy_(0, torch.tensor(Array(idx), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long))), zeroRow)
    }

    embedImpl
  }
}

/**
 * Companion object for EmbeddingInitializer with factory methods.
 */
object EmbeddingInitializer {
  def randomNormal(mean: Float = 0.0f, std: Float = 1.0f): RandomNormal = RandomNormal(mean, std)
  def randomUniform(minval: Float = 0.0f, maxval: Float = 1.0f): RandomUniform = RandomUniform(minval, maxval)
  def xavierNormal(gain: Float = 1.0f): XavierNormal = XavierNormal(gain)
  def xavierUniform(gain: Float = 1.0f): XavierUniform = XavierUniform(gain)
  def pretrained(weights: Array[Array[Float]], freeze: Boolean = true): Pretrained = Pretrained(weights, freeze)
}