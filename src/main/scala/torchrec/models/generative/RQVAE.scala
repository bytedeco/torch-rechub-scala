package torchrec.models.generative

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.utils.DeviceSupport

import scala.collection.mutable

/**
 * VectorQuantizer: Single-stage vector quantization module.
 *
 * Quantizes input features using a learned codebook with optional
 * Sinkhorn-based soft assignment. Matches PyTorch's VectorQuantizer.
 *
 * @param n_e Number of embeddings (codebook size)
 * @param e_dim Dimensionality of each embedding vector
 * @param beta Weight for commitment loss
 * @param sk_epsilon Entropy regularization for Sinkhorn
 * @param sk_iters Number of Sinkhorn iterations
 * @param device Device for computation
 */
class VectorQuantizer(
  n_e: Int,
  e_dim: Int,
  beta: Float = 0.25f,
  sk_epsilon: Float = 0.003f,
  sk_iters: Int = 100,
  device: String = DeviceSupport.backend
) extends Module {

  private val targetDevice = new Device(device)

  // Codebook (embedding table) - using EmbeddingImpl like PyTorch's nn.Embedding
  private val embedding = new EmbeddingImpl(new EmbeddingOptions(n_e, e_dim))
  register_module("embedding", embedding)

  // Initialize embeddings uniformly (-1/n_e to 1/n_e)
  private val bound = 1.0f / n_e
  private val weightParam = embedding.weight()
  // Use torch.uniform_ directly on the tensor with Float parameters
  torch.uniform_(weightParam, (-bound).toFloat, bound.toFloat)

  private var initted = true

  /** Return the current codebook embeddings */
  def getCodebook(): Tensor = embedding.weight()

  /** Retrieve codebook entries corresponding to given indices */
  def getCodebookEntry(indices: Tensor, shape: Option[List[Long]] = None): Tensor = {
    var z_q = embedding.forward(indices.toType(ScalarType.Long))
    shape.foreach { s =>
      val dimsArr = s.toArray
      if (dimsArr.length > 1) {
        val reshaped = dimsArr.length match {
          case 2 => z_q.view(dimsArr(0), dimsArr(1))
          case 3 => z_q.view(dimsArr(0), dimsArr(1), dimsArr(2))
          case 4 => z_q.view(dimsArr(0), dimsArr(1), dimsArr(2), dimsArr(3))
          case _ => z_q
        }
        z_q = reshaped
      }
    }
    z_q
  }

  /** Initialize embeddings using K-Means clustering */
  def initEmb(data: Tensor): Unit = {
    // Simplified: initialize with zeros (full K-Means would require sklearn)
    embedding.weight().zero_()
    initted = true
  }

  /** Center and normalize distance values for constrained optimization */
  private def centerDistanceForConstraint(distances: Tensor): Tensor = {
    val max_distance = distances.max()
    val min_distance = distances.min()
    val middle = max_distance.add(min_distance).mul(new Scalar(0.5f))
    val amplitude = max_distance.sub(middle).add(new Scalar(1e-5f.toDouble))
    distances.sub(middle).div(amplitude)
  }

  /** Sinkhorn algorithm for soft assignment */
  private def sinkhornAlgorithm(distances: Tensor): Tensor = {
    var Q = torch.exp(distances.neg().div(new Scalar(sk_epsilon.toDouble)))

    val B = Q.size(0).toInt
    val K = Q.size(1).toInt

    // Normalize so Q sums to 1
    var sumQ = Q.sum(1L).sum(0L)
    Q = Q.div(sumQ)

    for (_ <- 0 until sk_iters) {
      // Normalize columns: weight per sample must be 1/B
      Q = Q.div(Q.sum(1L))
      Q = Q.div(new Scalar(B.toDouble))

      // Normalize rows: weight per prototype must be 1/K
      Q = Q.div(Q.sum(0L))
      Q = Q.div(new Scalar(K.toDouble))
    }

    Q.mul(new Scalar(B.toDouble))
    Q
  }

  /** Forward pass - apply vector quantization */
  def forward(x: Tensor, use_sk: Boolean = true): (Tensor, Tensor, Tensor) = {
    // Handle different input shapes
    val batchSize = x.size(0)
    val seqLen = if (x.dim() > 1) x.size(1) else 1L
    val flat = x.view(-1, e_dim.toLong)

    // Initialize embeddings if needed (during training)
    if (!initted && is_training()) {
      initEmb(flat)
    }

    // Calculate L2 distance: d = ||z||^2 + ||e||^2 - 2<z,e>
    // Use torch.pow with Scalar for exponent (like FM.scala)
    val twoScalar = new Scalar(2.0)
    val latentSq = torch.pow(flat, twoScalar).sum(1)
    val codebookSq = torch.pow(embedding.weight(), twoScalar).sum(1)
    val crossTerm = torch.matmul(flat, embedding.weight().t()).mul(new Scalar(-2.0))
    var d = latentSq.add(codebookSq).add(crossTerm)

    // Get indices
    val indices: Tensor = if (!use_sk || sk_epsilon <= 0) {
      d.argmin(new LongOptional(1L), false)
    } else {
      // Use Sinkhorn algorithm
      d = centerDistanceForConstraint(d.toType(ScalarType.Double))
      val Q = sinkhornAlgorithm(d.toType(ScalarType.Double))
      if (Q.isnan().any().item().toFloat() != 0.0f || Q.isinf().any().item().toFloat() != 0.0f) {
        println("Sinkhorn returns nan/inf, falling back to hard assignment")
        d.toType(ScalarType.Float).argmin(new LongOptional(1L), false)
      } else {
        Q.toType(ScalarType.Float).argmax(new LongOptional(1L), false)
      }
    }

    // Get quantized vectors using the embedding
    val x_q = embedding.forward(indices.toType(ScalarType.Long))

    // Reshape to original shape
    val x_q_reshaped = if (x.dim() > 1) {
      x_q.view(batchSize, seqLen, e_dim.toLong)
    } else {
      x_q
    }

    // Compute losses - compute MSE directly, then mean
    // Using pattern from Loss.scala: reduction via .mean() on result
    val commitmentLoss = torch.mse_loss(x_q_reshaped.detach(), x)
    val codebookLoss = torch.mse_loss(x_q_reshaped, x.detach())
    val loss = codebookLoss.add(commitmentLoss.mul(new Scalar(beta.toDouble)))

    // Straight-through estimator: preserve gradients
    // x_q_st = x + (x_q - x).detach()
    val x_q_st = x.add(x_q_reshaped.sub(x).detach())

    // Reshape indices
    val indicesFinal = indices.view(batchSize, seqLen)

    (x_q_st, loss.mean(), indicesFinal)
  }
}

/**
 * ResidualVectorQuantizer: Multi-stage residual vector quantization.
 *
 * Uses ModuleListImpl to store VectorQuantizer layers, matching PyTorch's nn.ModuleList.
 * Computes mean quantization loss across all stages.
 *
 * @param n_e_list List of embeddings per stage
 * @param e_dim Embedding dimension
 * @param beta Commitment loss weight
 * @param sk_epsilon_list Entropy regularization per stage
 * @param sk_iters Sinkhorn iterations
 * @param device Device
 */
class ResidualVectorQuantizer(
  n_e_list: List[Int],
  e_dim: Int,
  beta: Float = 0.25f,
  sk_epsilon_list: Option[List[Float]] = None,
  sk_iters: Int = 100,
  device: String = DeviceSupport.backend
) extends Module {

  private val targetDevice = new Device(device)
  private val num_quantizers = n_e_list.size
  private val sk_eps_list = sk_epsilon_list.getOrElse(List.fill(num_quantizers)(0.003f))

  // ModuleListImpl to store VectorQuantizer layers - matches PyTorch's nn.ModuleList
  private val vq_layers: ModuleListImpl = {
    val moduleList = new ModuleListImpl()
    for (i <- 0 until num_quantizers) {
      val vq = new VectorQuantizer(
        n_e_list(i),
        e_dim,
        beta,
        sk_eps_list(i),
        sk_iters,
        device
      )
      moduleList.push_back(vq)
      register_module(s"vq_$i", vq)
    }
    moduleList
  }

  /** Return stacked codebooks from all quantizers */
  def getCodebook(): Tensor = {
    val codebooks = (0 until num_quantizers).map { i =>
      vq_layers.get(i).asInstanceOf[VectorQuantizer].getCodebook()
    }
    torch.stack(new TensorVector(codebooks.toSeq: _*))
  }

  /** Forward pass - apply multi-stage residual quantization */
  def forward(x: Tensor, use_sk: Boolean = true): (Tensor, Tensor, Tensor) = {
    val all_losses = mutable.ListBuffer[Tensor]()
    val all_indices = mutable.ListBuffer[Tensor]()

    var x_q = torch.zeros_like(x)
    var residual = x

    for (i <- 0 until num_quantizers) {
      val vq = vq_layers.get(i).asInstanceOf[VectorQuantizer]
      val (x_res, loss, indices) = vq.forward(residual, use_sk)

      // Update residual: r = r - quantized
      residual = residual.sub(x_res)
      // Accumulate quantized output
      x_q = x_q.add(x_res)

      all_losses += loss
      all_indices += indices
    }

    val mean_losses = torch.stack(new TensorVector(all_losses.toSeq*)).mean()
    val stacked_indices = torch.stack(new TensorVector(all_indices.toSeq*), -1L)

    (x_q, mean_losses, stacked_indices)
  }
}

/**
 * RQVAE: Residual Quantized Variational Autoencoder
 *
 * Implements a VAE with multi-stage residual vector quantizer.
 * Uses SequentialImpl for encoder/decoder MLPs.
 *
 * @param in_dim Input feature dimension
 * @param num_emb_list Embeddings per quantization stage
 * @param e_dim Codebook entry dimension
 * @param encoder_dims Encoder hidden layers
 * @param decoder_dims Decoder hidden layers
 * @param dropout Dropout probability
 * @param quant_loss_weight Quantization loss weight
 * @param beta Commitment loss weight
 * @param sk_epsilon Sinkhorn epsilon
 * @param sk_iters Sinkhorn iterations
 * @param device Device
 */
class RQVAE(
  in_dim: Int,
  num_emb_list: List[Int],
  e_dim: Int = 64,
  encoder_dims: List[Long] = List(256L, 128L),
  decoder_dims: List[Long] = List(128L, 256L),
  dropout: Float = 0.0f,
  quant_loss_weight: Float = 1.0f,
  beta: Float = 0.25f,
  sk_epsilon: Float = 0.003f,
  sk_iters: Int = 100,
  device: String = DeviceSupport.backend
) extends Module {

  // Auxiliary constructor with embedDim, numCodebooks, codebookSize, latentDim parameters
  def this(
    embedDim: Int,
    numCodebooks: Int,
    codebookSize: Int,
    latentDim: Int,
    device: String
  ) = this(
    in_dim = embedDim,
    num_emb_list = List.fill(numCodebooks)(codebookSize),
    e_dim = latentDim,
    encoder_dims = List(256L, 128L),
    decoder_dims = List(128L, 256L),
    dropout = 0.0f,
    quant_loss_weight = 1.0f,
    beta = 0.25f,
    sk_epsilon = 0.003f,
    sk_iters = 100,
    device = device
  )

  private val targetDevice = new Device(device)
  private val num_quantizers = num_emb_list.size

  // Build encoder using SequentialImpl (like PyTorch's nn.Sequential)
  private val encoder: SequentialImpl = {
    val seq = new SequentialImpl()
    var prevDim = in_dim.toLong
    for (i <- 0 until encoder_dims.size) {
      val dim = encoder_dims(i)
      seq.push_back(new LinearImpl(prevDim, dim))
      seq.push_back(new ReLUImpl())
      if (dropout > 0) {
        seq.push_back(new DropoutImpl(dropout))
      }
      prevDim = dim
    }
    // Final projection to e_dim
    seq.push_back(new LinearImpl(prevDim, e_dim.toLong))
    seq
  }
  register_module("encoder", encoder)

  // Residual Vector Quantizer with ModuleListImpl
  private val sk_epsilon_list = List.fill(num_quantizers)(sk_epsilon)
  private val rq = new ResidualVectorQuantizer(
    num_emb_list,
    e_dim,
    beta,
    Some(sk_epsilon_list),
    sk_iters,
    device
  )
  register_module("rq", rq)

  // Build decoder using SequentialImpl (like PyTorch's nn.Sequential)
  private val decoder: SequentialImpl = {
    val seq = new SequentialImpl()
    var prevDim = e_dim.toLong
    for (i <- 0 until decoder_dims.size) {
      val dim = decoder_dims(i)
      seq.push_back(new LinearImpl(prevDim, dim))
      seq.push_back(new ReLUImpl())
      if (dropout > 0) {
        seq.push_back(new DropoutImpl(dropout))
      }
      prevDim = dim
    }
    // Final projection back to in_dim
    seq.push_back(new LinearImpl(prevDim, in_dim.toLong))
    seq
  }
  register_module("decoder", decoder)

  // Move all submodules to target device
  if (device != "cpu") {
    encoder.to(targetDevice, false)
    rq.to(targetDevice, false)
    decoder.to(targetDevice, false)
    this.to(targetDevice, false)
  }

  /** Forward: encode -> quantize -> decode */
  def forward(x: Tensor, use_sk: Boolean = true): (Tensor, Tensor, Tensor) = {
    val encoded = encoder.forward(x)
    val (x_q, rq_loss, indices) = rq.forward(encoded, use_sk)
    val decoded = decoder.forward(x_q)
    (decoded, rq_loss, indices)
  }

  /** Compute total loss combining reconstruction and quantization losses */
  def computeLoss(out: Tensor, quant_loss: Tensor, target: Tensor, lossType: String = "mse"): (Tensor, Tensor) = {
    val reconLoss = if (lossType == "mse") {
      torch.mse_loss(out, target).mean()
    } else if (lossType == "l1") {
      torch.l1_loss(out, target).mean()
    } else {
      throw new IllegalArgumentException(s"Unknown loss type: $lossType")
    }
    val totalLoss = reconLoss.add(quant_loss.mul(new Scalar(quant_loss_weight.toDouble)))
    (totalLoss, reconLoss)
  }

  /** Get quantization indices (inference mode) */
  def getIndices(x: Tensor, use_sk: Boolean = false): Tensor = {
    val encoded = encoder.forward(x)
    val (_, _, indices) = rq.forward(encoded, use_sk)
    indices
  }
}

/** RQVAE companion object with factory methods */
object RQVAE {
  /**
   * Create RQVAE with simplified parameters
   */
  def apply(
    embedDim: Int,
    numCodebooks: Int,
    codebookSize: Int,
    latentDim: Int,
    device: String
  ): RQVAE = {
    new RQVAE(
      in_dim = embedDim,
      num_emb_list = List.fill(numCodebooks)(codebookSize),
      e_dim = latentDim,
      device = device
    )
  }

  /**
   * Create RQVAE with full parameters
   */
  def apply(
    in_dim: Int,
    num_emb_list: List[Int],
    e_dim: Int,
    encoder_dims: List[Long],
    decoder_dims: List[Long],
    dropout: Float,
    quant_loss_weight: Float,
    beta: Float,
    sk_epsilon: Float,
    sk_iters: Int,
    device: String
  ): RQVAE = {
    new RQVAE(
      in_dim,
      num_emb_list,
      e_dim,
      encoder_dims,
      decoder_dims,
      dropout,
      quant_loss_weight,
      beta,
      sk_epsilon,
      sk_iters,
      device
    )
  }
}
