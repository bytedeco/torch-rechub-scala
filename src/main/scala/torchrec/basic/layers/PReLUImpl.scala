package torchrec.basic.layers

import org.bytedeco.pytorch.{Module, ScalarTypeOptional, Tensor, TensorOptions}
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * PReLU activation
 */
class PReLUImpl extends Module {
  private var weight: Tensor = _

  def forward(x: Tensor): Tensor = {
    if (weight == null || weight.device() != x.device()) {
      if (weight != null) weight.close()
      weight = torch.zeros(Array[Long](1L), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
        .to(x.device(), ScalarType.Float)
    }
    torch.prelu(x, weight)
  }
}
