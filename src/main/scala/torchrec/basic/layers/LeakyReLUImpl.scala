package torchrec.basic.layers

import org.bytedeco.pytorch.{Module, Scalar, Tensor}
import org.bytedeco.pytorch.global.torch

/**
 * Leaky ReLU
 */
class LeakyReLUImpl extends Module {
  def forward(x: Tensor): Tensor = torch.leaky_relu(x, new Scalar(0.01f))
}
