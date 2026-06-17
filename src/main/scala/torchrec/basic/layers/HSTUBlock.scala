package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import torchrec.utils.DeviceSupport

/**
 * Stack of HSTULayer modules with external residual wiring.
 *
 * Each layer is wrapped as x = x + Layer(x), matching the HSTU paper / Meta reference.
 *
 * Shape
 * -----
 * Input: (batch_size, seq_len, d_model)
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

  // Store layers in a plain Scala list to avoid casting issues
  private val layerList: List[HSTULayer] = (0 until nLayers).map { i =>
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
    register_module(s"layer_$i", layer)
    layer
  }.toList

  def forward(
    x: Tensor,
    paddingMask: Option[Tensor] = None,
    timeDiffs: Option[Tensor] = None
  ): Tensor = {
    var result = x
    for (layer <- layerList) {
      result = result.add(layer.forward(result, paddingMask, timeDiffs))
    }
    result
  }
}

/**
 * HSTUBlock companion object with factory methods.
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
  ): HSTUBlock = {
    new HSTUBlock(dModel, nHeads, nLayers, dqk, dv, dropout, maxSeqLen, numTimeBuckets, timeBucketFn, timeBucketDivisor, timeBucketUnit, device)
  }
}