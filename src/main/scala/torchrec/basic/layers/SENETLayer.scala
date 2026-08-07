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
import torchrec.utils.DeviceSupport

/**
 * SENet-style feature gating (FiBiNet).
 *
 * Mirrors the Python reference `torch-rechub` implementation:
 *
 *   z = mean(x, dim=-1)              # squeeze over embedding dim
 *   a = Linear -> ReLU -> Linear -> ReLU (z)  # excitation
 *   v = x * a.unsqueeze(-1)          # scale
 *
 * Parameters
 * ----------
 * numFields : Int
 *   Number of feature fields.
 * reduction : Int, default = 3
 *   Bottleneck reduction ratio for the gating MLP.
 *
 * Shape
 * -----
 * Input  : (batch, numFields, embedDim)
 * Output : (batch, numFields, embedDim)
 */
class SENETLayer(
  numFields: Int,
  reduction: Int = 3,
  device: String = DeviceSupport.backend
) extends Module {

  require(numFields > 0, s"SENETLayer: numFields must be > 0, got $numFields")
  require(reduction > 0, s"SENETLayer: reduction must be > 0, got $reduction")

  private val reducedSize: Long = math.max(1L, numFields.toLong / reduction.toLong)

  private val mlp = new SequentialImpl()
  mlp.push_back("fc1", new LinearImpl(numFields.toLong, reducedSize))
  mlp.push_back("relu1", new ReLUImpl())
  mlp.push_back("fc2", new LinearImpl(reducedSize, numFields.toLong))
  mlp.push_back("relu2", new ReLUImpl())
  register_module("mlp", mlp)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    mlp.to(dev, false)
  }

  override def forward(x: Tensor): Tensor = {
    // x: (batch, numFields, embedDim)
    val z = x.mean(-1L)               // (batch, numFields)
    val a = mlp.forward(z)            // (batch, numFields)
    x.mul(a.unsqueeze(-1))            // (batch, numFields, embedDim)
  }
}