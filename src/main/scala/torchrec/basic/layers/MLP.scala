package torchrec.basic.layers

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import torchrec.utils.DeviceSupport

/**
 * Multi-Layer Perceptron using SequentialImpl (matching PyTorch's nn.Sequential)
 *
 * Parameters
 * ----------
 * inputDim : Long
 *   Input dimension.
 * hiddenDims : List[Long]
 *   Hidden layer sizes.
 * outputDim : Long
 *   Output dimension (default=1).
 * activation : String
 *   Activation function (sigmoid, relu, prelu, dice, softmax).
 * dropout : Float
 *   Dropout probability.
 * useBatchNorm : Boolean
 *   Whether to use batch norm.
 * useLayerNorm : Boolean
 *   Whether to use layer norm.
 * outputLayer : Boolean
 *   Whether to append a final Linear(*, outputDim) (default=True).
 * device : String
 *   Device for computation.
 */
class MLP(
  inputDim: Long,
  hiddenDims: List[Long],
  outputDim: Long = 1,
  activation: String = "relu",
  dropout: Float = 0.0f,
  useBatchNorm: Boolean = false,
  useLayerNorm: Boolean = false,
  outputLayer: Boolean = true,
  device: String = DeviceSupport.backend
) extends Module {

  // Use SequentialImpl like PyTorch's nn.Sequential
  private val sequential = new SequentialImpl()
  private var prevDim = inputDim
  private val layersRef = scala.collection.mutable.ListBuffer[AnyRef]()
  // Build MLP layers using SequentialImpl

  hiddenDims.foreach { dim =>
    sequential.push_back(new LinearImpl(prevDim, dim))

    if (useLayerNorm) {
      val vec = new LongVector(1)
      vec.put(0, dim)
      sequential.push_back(new LayerNormImpl(vec))
    } else if (useBatchNorm && !activation.equals("relu")) {
      sequential.push_back(new BatchNorm1dImpl(new BatchNormOptions(dim)))
    }
    activation.toLowerCase match {
      case "relu" => sequential.push_back(new ReLUImpl())
      case "sigmoid" =>sequential.push_back(new SigmoidImpl())
      case "tanh" => sequential.push_back(new TanhImpl())
      case "silu" | "swish" =>sequential.push_back(new SiLUImpl())
      case "gelu" => sequential.push_back(new GELUImpl())
      case "prelu" => sequential.push_back(new PReLUImpl())
      case "leaky_relu" | "leakyrelu" => sequential.push_back(new LeakyReLUImpl())
      case "none" | "identity" => sequential.push_back(new IdentityImpl())
      case _ =>sequential.push_back( new ReLUImpl())
    }

//    sequential.push_back(createActivation(activation))

    if (dropout > 0) {
      val dropoutLayer = new DropoutImpl(dropout)
      layersRef += dropoutLayer
      sequential.push_back(s"dropout_${layersRef.size}", dropoutLayer)
//      sequential.push_back(new DropoutImpl(dropout))
    }

    prevDim = dim
  }

  // Output layer
  if (outputLayer) {
    sequential.push_back(new LinearImpl(prevDim, outputDim))
  }

  // Move all layers to target device
  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    sequential.to(dev, false)
    this.to(dev, false)
  }

  def forward(x: Tensor): Tensor = {
    try {
      if (x == null) {
        throw new IllegalArgumentException("Input tensor cannot be null")
      }
      if (x.numel() == 0) {
        throw new IllegalArgumentException("Input tensor has no elements")
      }
      sequential.forward(x)
    } catch {
      case e: Exception =>
        System.err.println(s"[MLP] Forward pass failed: ${e.getMessage}")
        System.err.println(s"[MLP] Input shape: ${x.sizes().vec().get().mkString(", ")}, dtype: ${x.dtype()}")
        e.printStackTrace()
        throw e
    }
  }

  private def createActivation(act: String): Module = {
    act.toLowerCase match {
      case "relu" => new ReLUImpl()
      case "sigmoid" => new SigmoidImpl()
      case "tanh" => new TanhImpl()
      case "silu" | "swish" => new SiLUImpl()
      case "gelu" => new GELUImpl()
      case "prelu" => new PReLUImpl()
      case "leaky_relu" | "leakyrelu" => new LeakyReLUImpl()
      case "none" | "identity" => new IdentityImpl()
      case _ => new ReLUImpl()
    }
  }
}

/**
 * DNN (alias for MLP)
 */
object DNN {
  def apply(
    inputDim: Long,
    hiddenDims: List[Long],
    outputDim: Long = 1,
    activation: String = "relu",
    dropout: Float = 0.0f,
    useBatchNorm: Boolean = false,
    device: String = DeviceSupport.backend
  ): MLP = new MLP(inputDim, hiddenDims, outputDim, activation, dropout, useBatchNorm, device = device)
}

///**
// * Identity layer (no-op)
// */
//class IdentityImpl extends Module {
//  def forward(x: Tensor): Tensor = x
//}



