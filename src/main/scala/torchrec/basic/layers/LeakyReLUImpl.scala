package torchrec.basic.layers

import org.bytedeco.pytorch.{Scalar, Tensor}
import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.global.torch

/**
 * Leaky ReLU
 */
class LeakyReLUImpl extends Module {
  override def forward(x: Tensor): Tensor = torch.leaky_relu(x, new Scalar(0.01f))
}
