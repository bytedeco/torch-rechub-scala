package torchrec.distributed

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
import org.bytedeco.pytorch.global.torch.DeviceType

class DistributedConfig(
    val rank: Int,
    val worldSize: Int,
    val device: Device,
    val backend: String
) {
  def isMainProcess: Boolean = rank == 0
  def isDistributed: Boolean = worldSize > 1

  override def toString: String = s"DistributedConfig{rank=$rank, worldSize=$worldSize, device=$device, backend=$backend}"
}

object DistributedConfig {
  class Builder {
    private var rank: Int = 0
    private var worldSize: Int = 1
    private var backend: String = "GLOO"

    def rank(r: Int): Builder = { rank = r; this }
    def worldSize(ws: Int): Builder = { worldSize = ws; this }
    def backend(b: String): Builder = { backend = b; this }

    def build(): DistributedConfig = {
      val effectiveBackend = if ("NCCL" == backend && pt.cuda_is_available()) "NCCL" else "GLOO"
      val dev: Device = if ("NCCL" == effectiveBackend && pt.cuda_is_available()) {
        new Device(DeviceType.CUDA, rank.toByte)
      } else {
        new Device(DeviceType.CPU, 0.toByte)
      }
      new DistributedConfig(rank, worldSize, dev, effectiveBackend)
    }
  }

  def builder(): Builder = new Builder
}
