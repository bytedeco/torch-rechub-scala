package torchrec.basic.layers

import org.bytedeco.pytorch._
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