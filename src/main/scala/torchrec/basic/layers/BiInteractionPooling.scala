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
import org.bytedeco.pytorch.global.torch
import torchrec.utils.DeviceSupport

/**
 * Bi-Interaction Pooling layer for NFM.
 * Combines pairwise feature interactions via element-wise product,
 * then pools them using sum (or mean) aggregation.
 *
 * Reference: "Neural Factorization Machines for Sparse Predictive Analytics" (SIGIR 2017)
 * Equation: f_BI = sum_i sum_j <V_i, V_j> * x_i * x_j
 * Which simplifies to: f_BI = 0.5 * (sum_i (sum_j V_j * x_j)^2 - sum_i sum_j (V_i * V_j * x_i * x_j))
 * Implemented as: 0.5 * (sum(S)^2 - sum(S^2)) where S = sum_i V_i * x_i
 */
class BiInteractionPooling(device: String = DeviceSupport.backend) extends Module {

  override def forward(embeddings: Tensor): Tensor = {
    // embeddings: (batch, num_fields, embed_dim)
    // Compute sum of all field embeddings
    val sumOfEmbeddings = embeddings.sum(1)  // (batch, embed_dim)

    // Compute sum of squares of embeddings
    val squaredSum = torch.pow(embeddings, new Scalar(2.0f)).sum(1)  // (batch, embed_dim)

    // Compute square of sum
    val sumSquared = torch.pow(sumOfEmbeddings, new Scalar(2.0f))  // (batch, embed_dim)

    // Bi-interaction: 0.5 * (sum^2 - squared_sum)
    val half = new Scalar(0.5f)
    sumSquared.sub(squaredSum).mul(half)
  }
}
