package torchrec.basic.layers

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

import scala.collection.mutable

/**
 * Bilinear feature interaction (FFM-style).
 *
 * Mirrors the Python reference: a stack of `Linear(input_dim, input_dim, bias=False)`
 * projections whose layout depends on `bilinearType`:
 *
 * - ``field_all``         — one shared bilinear across every pair
 * - ``field_each``        — one bilinear per source field (FFM: apply W_i to v_i,
 *                           multiply by v_j)
 * - ``field_interaction`` — one bilinear per (i, j) pair (FiBiNet: pair-specific
 *                           projection)
 *
 * Shape
 * -----
 * Input:  (batch, numFields, embedDim)
 * Output: (batch, numPairs, embedDim)
 */
class BiLinearInteractionLayer(
  inputDim: Long,
  numFields: Int,
  bilinearType: String = "field_interaction",
  device: String = DeviceSupport.backend
) extends Module {

  require(numFields >= 2, s"BiLinearInteractionLayer needs numFields >= 2, got $numFields")

  // Order in which (i, j) pairs are visited — must stay consistent between init
  // (which builds one linear per pair for `field_interaction`) and forward.
  private val pairs: IndexedSeq[(Int, Int)] =
    (0 until numFields).flatMap(i => (i + 1 until numFields).map(j => (i, j)))

  // ModuleList mirrors the Python `nn.ModuleList`. `.to(device)` is intentionally
  // skipped: in this bytedeco build, `LinearImpl` created via
  // `LinearOptions.bias(false)` crashes on `Module.to(Device, false)` (same
  // family of bugs as the Conv1dImpl crash — see CIN notes).
  private val bilinearLayers: ModuleListImpl = new ModuleListImpl()

  bilinearType match {
    case "field_all" =>
      val layer = new LinearImpl(new LinearOptions(inputDim, inputDim).bias(false))
      register_module("bilinear_layer", layer)
      bilinearLayers.push_back(layer)
    case "field_each" =>
      // One bilinear per source field — FFM-style: apply W_i to v_i, multiply by v_j.
      for (i <- 0 until numFields) {
        val layer = new LinearImpl(new LinearOptions(inputDim, inputDim).bias(false))
        register_module(s"bilinear_layer_$i", layer)
        bilinearLayers.push_back(layer)
      }
    case "field_interaction" =>
      // One bilinear per (i, j) pair — FiBiNet-style pair-specific projection.
      var idx = 0
      for ((_, _) <- pairs) {
        val layer = new LinearImpl(new LinearOptions(inputDim, inputDim).bias(false))
        register_module(s"bilinear_layer_$idx", layer)
        bilinearLayers.push_back(layer)
        idx += 1
      }
    case other =>
      throw new NotImplementedError(s"bilinearType $other not implemented")
  }

  override def forward(x: Tensor): Tensor = {
    val nf = x.size(1).toInt
    // Per-field embeddings: IndexedSeq of (batch, embedDim).
    val fields: IndexedSeq[Tensor] = (0 until nf).map(i => x.select(1, i.toLong))

    val out = mutable.ArrayBuffer[Tensor]()
    bilinearType match {
      case "field_all" =>
        val shared = bilinearLayers.get(0)
        for ((i, j) <- pairs) {
          out += shared.forward(fields(i)).mul(fields(j))
        }
      case "field_each" =>
        // bilinear_layer[i](v_i) * v_j for each (i, j)
        for ((i, j) <- pairs) {
          out += bilinearLayers.get(i).forward(fields(i)).mul(fields(j))
        }
      case "field_interaction" =>
        // bilinear_layer[idx](v_i) * v_j for each pair, enumerated
        var idx = 0
        for ((i, j) <- pairs) {
          out += bilinearLayers.get(idx).forward(fields(i)).mul(fields(j))
          idx += 1
        }
    }

    if (out.isEmpty) torch.empty(0L)
    else torch.cat(new TensorVector(out.toSeq: _*), 1L)
  }
}

/**
 * BiLinearInteractionLayer factory — identical to the constructor.
 */
object BiLinearInteractionLayer {
  def apply(
    inputDim: Long,
    numFields: Int,
    bilinearType: String = "field_interaction",
    device: String = DeviceSupport.backend
  ): BiLinearInteractionLayer =
    new BiLinearInteractionLayer(inputDim, numFields, bilinearType, device)
}