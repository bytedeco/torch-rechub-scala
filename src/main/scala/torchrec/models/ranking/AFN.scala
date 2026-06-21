package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Adaptive Factorization Network (AFN)
 *
 * Key idea: Use Logarithmic Neural Network (LNN) blocks to learn adaptive
 * feature interactions. LNN transforms input features logarithmically before
 * passing through a linear combination, enabling it to learn arbitrary
 * polynomial interactions of a specified order.
 *
 * Architecture:
 *   Sparse Input → EmbeddingLayer → LNN → MLP → Output
 *
 * Reference: "Adaptive Factorization Network" (AAAI 2020)
 *
 * @param features   List of sparse features
 * @param embedDim   Embedding dimension
 * @param lnnDim     LNN hidden dimension (controls interaction order)
 * @param mlpDims    Hidden layer dimensions for the MLP
 * @param dropout    Dropout rate
 * @param device     Device to run on
 */
class AFN(
  features: List[Feature],
  embedDim: Int = 8,
  lnnDim: Int = 8,
  mlpDims: List[Long] = List(256L, 128L, 64L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")

  private val numFields = features.collect { case f: SparseFeature => 1 }.size
  require(numFields >= 2, "AFN requires at least 2 sparse features")

  // Embedding layer
  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  // LNN: Logarithmic Neural Network for adaptive factorization
  private val lnnOutputDim = lnnDim * embedDim

  // LNN weight: (lnn_dim, num_fields)
  private val lnnWeight = {
    val opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
    val w = torch.randn(Array(lnnDim.toLong, numFields.toLong), opts).
      mul(new Scalar(math.sqrt(2.0 / numFields).toFloat))
    register_parameter("lnn_weight", w)
    w
  }

  // LNN bias: (lnn_dim, embed_dim)
  private val lnnBias = {
    val opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
    val b = torch.zeros(Array(lnnDim.toLong, embedDim.toLong), opts)
    register_parameter("lnn_bias", b)
    b
  }

  // Dropout layer for LNN output
  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropout_lnn", dropoutLayer)

  // MLP on LNN output
  private val mlp = new MLP(lnnOutputDim, mlpDims, 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  if (device != "cpu") {
    lnnWeight.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
    lnnBias.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
    mlp.to(new org.bytedeco.pytorch.Device(device), false)
    ()
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Get embeddings: (batch, num_fields, embed_dim) using forward3D
    val embeddings = embeddingLayer.forward3D(sparseFeats)
    val batchSize = embeddings.size(0).toInt

    // LNN: log(1 + |x|) transformation
    val absEmbeddings = embeddings.abs()
    val logEmbeddings = torch.log1p(absEmbeddings)

    val w = lnnWeight.to(embeddings.device(), ScalarType.Float)
    val b = lnnBias.to(embeddings.device(), ScalarType.Float)

    // LNN computation: W @ log(x)
    // logEmbeddings: (batch, num_fields, embed_dim) = (B, F, E)
    // w: (lnn_dim, num_fields) = (L, F)
    // Transpose logEmbeddings to (batch, embed_dim, num_fields) for matmul
    val logEmbeddingsT = logEmbeddings.transpose(1, 2)  // (B, E, F)
    val wT = w.t()  // (num_fields, lnn_dim) = (F, L)
    // preAct: (batch, embed_dim, lnn_dim) = (B, E, L)
    // Use matmul to allow broadcasting the shared weight across batch dimension
    val preAct = torch.matmul(logEmbeddingsT, wT)
    // preActT: (batch, lnn_dim, embed_dim) = (B, L, E)
    val preActT = preAct.transpose(1, 2)

    // Add bias
    val bBcast = b.unsqueeze(0).expand(batchSize.toLong, lnnDim.toLong, embedDim.toLong)
    var lnnOut = preActT.add(bBcast)

    // expm1: exp(x) - 1
    lnnOut = torch.expm1(lnnOut)

    // ReLU activation
    lnnOut = lnnOut.relu()

    // Dropout
    lnnOut = dropoutLayer.forward(lnnOut)

    // Flatten: (batch, lnn_dim * embed_dim)
    // Use reshape instead of view to avoid non-contiguous tensor errors after transpose
    val lnnFlat = lnnOut.reshape(batchSize, lnnOutputDim.toLong)

    // MLP
    mlp.forward(lnnFlat).squeeze(1)
  }
}
