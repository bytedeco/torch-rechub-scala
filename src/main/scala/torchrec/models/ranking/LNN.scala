package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Logarithmic Neural Network (LNN) for recommendation
 *
 * Key idea: Use logarithmic transformation of input features to enable
 * learning of arbitrary polynomial interactions in a linear layer.
 *
 * The LNN transformation:
 *   y = exp(W * log(|x| + eps) + b) - 1
 *
 * This enables the model to learn higher-order interactions through
 * a shallow network because the logarithm converts multiplicative
 * interactions into additive ones.
 *
 * Reference: "Liquid Neural Networks for Recommendation Systems" (inspired by ODE-based approaches)
 *
 * @param numFields   Number of input fields (features)
 * @param embedDim    Embedding dimension
 * @param lnnDim      LNN hidden dimension (controls interaction order/capacity)
 * @param mlpDims     Hidden layer dimensions for the MLP after LNN
 * @param dropout     Dropout rate
 * @param device      Device to run on
 */
class LNN(
  numFields: Int,
  embedDim: Int,
  lnnDim: Int = 8,
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(numFields >= 2, s"numFields must be >= 2, got $numFields")
  require(embedDim > 0, s"embedDim must be positive, got $embedDim")
  require(lnnDim >= 1, s"lnnDim must be >= 1, got $lnnDim")

  private val lnnOutputDim = lnnDim * embedDim

  // LNN weight: (lnn_dim, num_fields)
  private val lnnWeight = {
    val w = torch.randn(Array(lnnDim.toLong, numFields.toLong): _*).
      mul(new Scalar(math.sqrt(2.0 / numFields).toFloat))
    val p = new Tensor()
    p.copy_(w)
    register_parameter("lnn_weight", p)
    p
  }

  // LNN bias: (lnn_dim, embed_dim)
  private val lnnBias = {
    val opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
    val b = torch.zeros(Array(lnnDim.toLong, embedDim.toLong), opts)
    val p = new Tensor()
    p.copy_(b)
    register_parameter("lnn_bias", p)
    p
  }

  if (device != "cpu") {
    lnnWeight.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
    lnnBias.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
    ()
  }

  def forward(x: Tensor): Tensor = {
    // x: (batch, num_fields, embed_dim) = (B, F, E)
    val batchSize = x.size(0).toInt

    // Logarithmic transformation: log(1 + |x|)
    val absX = x.abs()
    val logX = torch.log1p(absX)

    val w = lnnWeight.to(x.device(), ScalarType.Float)
    // w: (lnn_dim, num_fields) = (L, F)

    // Transpose logX to (batch, embed_dim, num_fields) for matmul
    val logXT = logX.transpose(1, 2)  // (B, E, F)
    // wT: (num_fields, lnn_dim) = (F, L)
    val wT = w.t()
    // preAct: (batch, embed_dim, lnn_dim) = (B, E, L)
    val preAct = torch.bmm(logXT, wT)
    // preActT: (batch, lnn_dim, embed_dim) = (B, L, E)
    val preActT = preAct.transpose(1, 2)

    // Add bias: b.unsqueeze(0) gives (1, L, E), broadcasts to (B, L, E)
    val b = lnnBias.to(x.device(), ScalarType.Float)
    val bBcast = b.unsqueeze(0).expand(batchSize.toLong, lnnDim.toLong, embedDim.toLong)
    var out = preActT.add(bBcast)

    // expm1: exp(x) - 1
    out = torch.expm1(out)

    // ReLU activation
    out = out.relu()

    // Flatten: (batch, lnn_dim * embed_dim)
    out.view(batchSize, lnnOutputDim.toLong)
  }
}
