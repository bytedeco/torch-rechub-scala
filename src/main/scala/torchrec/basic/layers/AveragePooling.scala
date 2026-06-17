package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * Mean pooling over sequence embeddings.
 *
 * Shape
 * -----
 * Input
 *   x : ``(B, L, D)``
 *   mask : ``(B, 1, L)``
 * Output
 *   ``(B, D)``
 */
class AveragePooling() extends Module {

  def forward(x: Tensor, mask: Option[Tensor] = None): Tensor = {
    if (mask.isEmpty) {
      torch.mean(x, 1L)
    } else {
      val m = mask.get
      val sumPoolingMatrix = torch.bmm(m, x).squeeze(1L)
      val nonPaddingLength = m.sum(1L)
      sumPoolingMatrix.div(nonPaddingLength.add(new Scalar(1e-16f)))
    }
  }
}

/**
 * AveragePooling companion object with factory methods.
 */
object AveragePooling {
  def apply(): AveragePooling = new AveragePooling()
}