package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * Sum pooling over sequence embeddings.
 *
 * Shape
 * -----
 * Input
 *   x : ``(B, L, D)``
 *   mask : ``(B, 1, L)``
 * Output
 *   ``(B, D)``
 */
class SumPooling() extends Module {

  def forward(x: Tensor, mask: Option[Tensor] = None): Tensor = {
    if (mask.isEmpty) {
      torch.sum(x, 1L)
    } else {
      torch.bmm(mask.get, x).squeeze(1L)
    }
  }
}

/**
 * SumPooling companion object with factory methods.
 */
object SumPooling {
  def apply(): SumPooling = new SumPooling()
}