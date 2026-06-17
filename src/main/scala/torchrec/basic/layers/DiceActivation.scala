package torchrec.basic.layers

import org.bytedeco.pytorch.{BatchNorm1dImpl, BatchNormOptions, Module, Scalar, ScalarTypeOptional, Tensor, TensorOptions}
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Dice activation function
 * Reference: Alibaba DIN paper, KDD 2018
 *
 * Formula: output = p * x + (1 - p) * alpha * x
 * where p = sigmoid(beta * bn(x)), and alpha, beta are learnable
 *
 * Input: (batch, embed) or (batch, seq, embed)
 * Output: same shape as input
 */
class DiceActivation(
                      val embedSize: Int,
                      val eps: Float = 1e-8f
                    ) extends Module {

  private val bn = new BatchNorm1dImpl(new BatchNormOptions(embedSize))
  register_module("bn", bn)

  // Learnable alpha and beta per dimension
  private val alpha = torch.zeros(Array[Long](embedSize.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  private val beta = torch.zeros(Array[Long](embedSize.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  register_parameter("alpha", alpha)
  register_parameter("beta", beta)

  def forward(x: Tensor): Tensor = {
    val dim = x.dim()
    val batchSize = x.size(0)

    if (dim == 2L) {
      // 2D: (batch, embed)
      val bnOut = bn.forward(x)
      val p = bnOut.mul(beta).sigmoid()
      val alphaB = alpha.reshape(1L, embedSize.toLong).expand(batchSize, embedSize.toLong)
      // Dice: alpha * (1-p) * x + p * x
      val term1 = alphaB.mul(p.neg().add(new Scalar(1.0)))
      val term2 = p.mul(x)
      term1.add(term2)
    } else {
      // 3D: (batch, seq, embed)
      val seqLen = x.size(1)
      val xFlat = x.transpose(1, 2).reshape(batchSize * seqLen, embedSize.toLong)
      val bnOut = bn.forward(xFlat)
      val p = bnOut.mul(beta).sigmoid()
      val alphaB = alpha.reshape(1L, embedSize.toLong).expand(batchSize * seqLen, embedSize.toLong)
      val term1 = alphaB.mul(p.neg().add(new Scalar(1.0)))
      val term2 = p.mul(xFlat)
      term1.add(term2).reshape(batchSize, embedSize.toLong, seqLen).transpose(1, 2)
    }
  }
}