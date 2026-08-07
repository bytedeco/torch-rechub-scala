package torchrec.basic.layers

import org.bytedeco.javacpp.LongPointer
import org.bytedeco.pytorch._
import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.nn.modules._
import org.bytedeco.pytorch.nn.modules.container._
import org.bytedeco.pytorch.nn.options._
import org.bytedeco.pytorch.optim._
import org.bytedeco.pytorch.data.datasets._
import org.bytedeco.pytorch.data.options._
import org.bytedeco.pytorch.data.sampler._
import org.bytedeco.pytorch.distributed._
import org.bytedeco.pytorch.global.torch

import torchrec.utils.DeviceSupport

/**
 * Compressed Interaction Network (xDeepFM).
 *
 * Mirrors the Python reference: stacks 1×1 `Conv1d` layers over outer-product
 * feature maps; sums each layer's channels and feeds the concatenated vector
 * through a final `Linear` to produce a single logit per batch element.
 *
 * Parameters
 * ----------
 * inputDim : Int
 *   Number of feature fields (the original CIN paper's `input_dim`).
 * cinSize : List[Int]
 *   Output channels per Conv1d layer.
 * splitHalf : Boolean, default = true
 *   When true, halve the output channels of every layer except the last, and
 *   feed the second half into the next layer — the standard xDeepFM trick.
 * device : String
 *   Device for parameters.
 *
 * Shape
 * -----
 * Input  : (batch, numFields, embedDim)
 * Output : (batch, 1)
 */
class CIN(
  inputDim: Int,
  cinSize: List[Int],
  splitHalf: Boolean = true,
  device: String = DeviceSupport.backend
) extends Module {

  require(inputDim > 0, s"CIN: inputDim must be > 0, got $inputDim")
  require(cinSize.nonEmpty, s"CIN: cinSize must not be empty")

  private val numLayers: Int = cinSize.length

  // Mirror Python's nn.ModuleList — registered with PyTorch's module system.
  // We index it directly in forward() instead of caching typed refs.
  private val convLayers: ModuleListImpl = new ModuleListImpl()

  // Build layers eagerly — conv input dim depends on `prev_dim`, which in turn
  // depends on whether the previous layer split its channels in half.
  var prevDim: Int = inputDim
  var fcInputDim: Int = 0
  var i = 0
  while (i < numLayers) {
    val fullSize = cinSize(i)
    val opt = new Conv1dOptions(inputDim.toLong * prevDim.toLong, fullSize.toLong, new LongPointer(Array(1L): _*))
    val conv = new Conv1dImpl(opt)
    convLayers.push_back(conv)
    register_module(s"conv_$i", conv)

    val splitLayer = splitHalf && i != numLayers - 1
    val sizeForNext = if (splitLayer) fullSize / 2 else fullSize
    prevDim = sizeForNext
    fcInputDim += sizeForNext

    i += 1
  }

  private val fc = new LinearImpl(fcInputDim.toLong, 1L)
  register_module("fc", fc)

  override def forward(x: Tensor): Tensor = {
    // x: (batch, inputDim, embedDim)
    val batchSize = x.size(0)
    val embedDim = x.size(2)

    val x0 = x.unsqueeze(2)              // (batch, inputDim, 1, embedDim)
    var h = x                             // (batch, inputDim, embedDim) → updated each layer
    val xs = scala.collection.mutable.ListBuffer[Tensor]()

    var i = 0
    while (i < numLayers) {
      // Outer product across field pairs: x0 (F, 1) × h.unsqueeze(1) (1, prev)
      // gives (batch, inputDim, prevDim, embedDim) via broadcasting.
      val outer = x0.mul(h.unsqueeze(1))  // (batch, inputDim, prevDim, embedDim)
      val reshaped = outer.view(batchSize, inputDim.toLong * h.size(1), embedDim)

      val convOut = convLayers.get(i).forward(reshaped).relu()

      if (splitHalf && i != numLayers - 1) {
        val halfSize = convOut.size(1) / 2
        val first = convOut.narrow(1, 0L, halfSize)
        val second = convOut.narrow(1, halfSize, halfSize)
        xs += first
        h = second
      } else {
        xs += convOut
        h = convOut
      }

      i += 1
    }

    val tensorVec = new TensorVector(xs.toSeq: _*)
    val stacked = torch.cat(tensorVec, 1L)  // (batch, fcInputDim, embedDim)
    val summed = stacked.sum(2L)              // (batch, fcInputDim)
    fc.forward(summed)
  }
}

/**
 * CIN factory — identical to the constructor; mirrors the Python
 * ``CIN(input_dim, cin_size, split_half=True)`` call style.
 */
object CIN {
  def apply(
    inputDim: Int,
    cinSize: List[Int],
    splitHalf: Boolean = true,
    device: String = DeviceSupport.backend
  ): CIN = new CIN(inputDim, cinSize, splitHalf, device)
}