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
 * Keep original sequence embedding shape.
 *
 * Shape
 * -----
 * Input: ``(B, L, D)``
 * Output: ``(B, L, D)``
 */
class ConcatPooling() extends Module {

   def forward(x: Tensor, mask: Option[Tensor] = None): Tensor = x
}

/**
 * ConcatPooling companion object with factory methods.
 */
object ConcatPooling {
  def apply(): ConcatPooling = new ConcatPooling()
}