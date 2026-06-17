package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import torchrec.utils.DeviceSupport

/**
 * Logistic regression module.
 *
 * Parameters
 * ----------
 * inputDim : int
 *   Input dimension.
 * sigmoid : bool, default False
 *   Apply sigmoid to output when True.
 *
 * Shape
 * -----
 * Input: ``(B, input_dim)``
 * Output: ``(B, 1)``
 */
class LR(
  inputDim: Long,
  sigmoid: Boolean = false,
  device: String = DeviceSupport.backend
) extends Module {

  private val fc = new LinearImpl(inputDim, 1)
  register_module("fc", fc)
  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    fc.to(dev, false)
  }

  def forward(x: Tensor): Tensor = {
    if (sigmoid) {
      torch.sigmoid(fc.forward(x))
    } else {
      fc.forward(x)
    }
  }
}

/**
 * LR companion object with factory methods.
 */
object LR {
  def apply(inputDim: Long, sigmoid: Boolean = false, device: String = DeviceSupport.backend): LR = {
    new LR(inputDim, sigmoid, device)
  }
}