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
import torchrec.utils.DeviceSupport

/**
 * Stack of HSTULayer modules with **external residual** wiring, stored in a
 * `ModuleListImpl` so the stack is registered with PyTorch's module system
 * (state-dict, parameter discovery, etc.).
 *
 * Each layer is wrapped as ``x = x + Layer(x)``, matching the HSTU paper /
 * Meta reference.
 *
 * Shape
 * -----
 * Input:  (batch_size, seq_len, d_model)
 * Output: (batch_size, seq_len, d_model)
 */
class HSTUBlock(
  dModel: Int = 512,
  nHeads: Int = 8,
  nLayers: Int = 4,
  dqk: Int = 64,
  dv: Int = 64,
  dropout: Float = 0.1f,
  maxSeqLen: Int = 200,
  numTimeBuckets: Int = 128,
  timeBucketFn: String = "sqrt",
  timeBucketDivisor: Float = 1.0f,
  timeBucketUnit: String = "minutes",
  device: String = DeviceSupport.backend
) extends Module {

  // ModuleListImpl mirrors PyTorch's nn.ModuleList so the stacked HSTULayers
  // participate in state_dict() / parameter discovery.
  private val layers: ModuleListImpl = new ModuleListImpl()
  // Keep typed references — `ModuleListImpl` only hands back generic `Module`
  // entries, which would force an unchecked cast on every forward call.
  private val layerRefs: Array[HSTULayer] = Array.ofDim(nLayers)

  for (i <- 0 until nLayers) {
    val layer = new HSTULayer(
      dModel = dModel,
      nHeads = nHeads,
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
    layers.push_back(layer)
    register_module(s"layer_$i", layer)
    layerRefs(i) = layer
  }

  def forward(
    x: Tensor,
    paddingMask: Option[Tensor] = None,
    timeDiffs: Option[Tensor] = None
  ): Tensor = {
    var h = x
    layerRefs.foreach { layer =>
      h = h.add(layer.forward(h, paddingMask, timeDiffs))
    }
    h
  }
}

/**
 * HSTUBlock factory — identical to the constructor; provided for call-site symmetry
 * with HSTULayer.apply and the Python ``HSTUBlock(...)`` style.
 */
object HSTUBlock {
  def apply(
    dModel: Int = 512,
    nHeads: Int = 8,
    nLayers: Int = 4,
    dqk: Int = 64,
    dv: Int = 64,
    dropout: Float = 0.1f,
    maxSeqLen: Int = 200,
    numTimeBuckets: Int = 128,
    timeBucketFn: String = "sqrt",
    timeBucketDivisor: Float = 1.0f,
    timeBucketUnit: String = "minutes",
    device: String = DeviceSupport.backend
  ): HSTUBlock =
    new HSTUBlock(dModel, nHeads, nLayers, dqk, dv, dropout, maxSeqLen,
      numTimeBuckets, timeBucketFn, timeBucketDivisor, timeBucketUnit, device)
}