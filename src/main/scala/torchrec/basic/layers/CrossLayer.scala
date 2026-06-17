package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.utils.DeviceSupport

/**
 * Cross layer.
 *
 * Parameters
 * ----------
 * inputDim : int
 *   Input dimension.
 *
 * Shape
 * -----
 * Input: ``(B, *)``
 * Output: ``(B, *)``
 */
class CrossLayer(
  inputDim: Long,
  device: String = DeviceSupport.backend
) extends Module {

  private val w = new LinearImpl(inputDim, 1)
  register_module("w", w)
  w.to(new org.bytedeco.pytorch.Device(device), false)

  private val b = torch.zeros(Array[Long](inputDim), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  register_buffer("b", b)

  def forward(x0: Tensor, xi: Tensor): Tensor = {
    val wtx = w.forward(xi) // (batch, 1)
    val x0Wtx = x0.mul(wtx) // (batch, dim)
    x0Wtx.add(b)
  }
}

/**
 * CrossLayer companion object with factory methods.
 */
object CrossLayer {
  def apply(inputDim: Long, device: String = DeviceSupport.backend): CrossLayer = {
    new CrossLayer(inputDim, device)
  }
}