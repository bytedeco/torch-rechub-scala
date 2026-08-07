
package torchrec.amp

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.nn.modules._
import org.bytedeco.pytorch.nn.modules.container._
import org.bytedeco.pytorch.nn.options._
import org.bytedeco.pytorch.optim._
import org.bytedeco.pytorch.data.datasets._
import org.bytedeco.pytorch.data.options._
import org.bytedeco.pytorch.data.sampler._
import org.bytedeco.pytorch.distributed._
import org.bytedeco.pytorch.global.torch as pt

/** GradScaler - Hunhe jingdu Xunlian de Tandu Suofangqi */
class GradScaler(
    val enabled: Boolean = true,
    private var scaleFactor: Float = 65536.0f
) {
  def this() = this(true)

  def scale(loss: Tensor): Tensor = {
    if (!enabled) loss
    else loss.mul(new Scalar(scaleFactor))
  }

  def step(optimizer: Optimizer, params: TensorVector): Unit = {
    if (!enabled) {
      optimizer.step()
      return
    }

    var hasInf = false
    var paramIdx = 0L
    while (paramIdx < params.size) {
      val g = params.get(paramIdx).grad()
      if (g != null && g.defined()) {
        if (!pt.isfinite(g).all().item_bool()) {
          hasInf = true
          paramIdx = params.size
        }
      }
      paramIdx += 1
    }

    if (hasInf) {
      var gradIdx = 0L
      while (gradIdx < params.size) {
        val g = params.get(gradIdx).grad()
        if (g != null && g.defined()) g.zero_()
        gradIdx += 1
      }
      scaleFactor *= 0.5f
      return
    }

    val inv = new Scalar(1.0f / scaleFactor)
    var k = 0L
    while (k < params.size) {
      val g = params.get(k).grad()
      if (g != null && g.defined()) g.mul_(inv)
      k += 1
    }

    optimizer.step()
    scaleFactor = Math.min(scaleFactor * 1.01f, 65536.0f)
  }

  def update(): Unit = { }

  def getScale: Float = scaleFactor

  def isEnabled: Boolean = enabled
}