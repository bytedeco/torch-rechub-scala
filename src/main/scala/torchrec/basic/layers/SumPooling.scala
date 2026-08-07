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