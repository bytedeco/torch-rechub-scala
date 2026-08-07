
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
import org.bytedeco.pytorch.global.torch.{DeviceType, ScalarType}

/** AutoCast - Zidong hunhe jingdu de Wencheng Guanli Qi */
class AutoCast(
    deviceType: DeviceType,
    precision: AutoCast.Precision
) extends AutoCloseable {
  private val prevEnabled: Boolean = pt.is_autocast_enabled(deviceType)
  private val prevDtype: ScalarType = pt.get_autocast_dtype(deviceType)
  private val active: Boolean = precision != AutoCast.Precision.FP32

  if (active) {
    pt.set_autocast_enabled(deviceType, true)
    pt.set_autocast_dtype(deviceType, precision.toScalarType)
  }

  override def close(): Unit = {
    if (active) {
      pt.set_autocast_enabled(DeviceType.CPU, prevEnabled)
      pt.set_autocast_dtype(DeviceType.CPU, prevDtype)
    }
  }

  def isActive: Boolean = active
}

object AutoCast {
  enum Precision {
    case FP16, BF16, FP32

    def toScalarType: ScalarType = this match {
      case FP16 => ScalarType.Half
      case BF16 => ScalarType.BFloat16
      case FP32 => ScalarType.Float
    }
  }

  def apply(deviceType: DeviceType, precision: Precision): AutoCast =
    if (precision == Precision.FP32) new AutoCast(deviceType, Precision.FP32)
    else new AutoCast(deviceType, precision)

  def cpu(precision: Precision): AutoCast = apply(DeviceType.CPU, precision)

  def cuda(precision: Precision): AutoCast = apply(DeviceType.CUDA, precision)
}