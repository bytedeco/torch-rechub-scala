package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import torchrec.utils.DeviceSupport

/**
 * Factorization Machine for 2nd-order feature interactions
 * FM: y = sum_i w_i * x_i + sum_i sum_j<i <w_i, w_j> * x_i * x_j
 */
class FM(
  embedDim: Int = 8,
  device: String = DeviceSupport.backend
) extends Module {

  def forward(embeddings: Tensor): Tensor = {
    // embeddings: (batch_size, num_fields, embed_dim)
    val batchSize = embeddings.size(0)
    val numFields = embeddings.size(1)

    // First order: sum of embeddings
    val firstOrder = embeddings.sum(1) // (batch_size, embed_dim)

    // Second order: sum of squared - squared sum
    // Use torch.pow with Scalar for exponent
    val twoScalar = new Scalar(2.0f)
    val squaredSum = torch.pow(embeddings, twoScalar).sum(1)
    val sumSquared = torch.pow(embeddings.sum(1), twoScalar)

    // Interaction: 0.5 * (sum^2 - squared_sum)
    val halfScalar = new Scalar(0.5f)
    val interactions = sumSquared.sub(squaredSum).mul(halfScalar)

    // FM output: sum of first order + sum of interactions
    firstOrder.add(interactions).sum(1).unsqueeze(1)
  }
}

/**
 * 2nd-order FM interaction only (no first-order)
 *
 * Python原版对照: FM(reduce_sum=False) returns (batch, embed_dim)
 * This computes: 0.5 * sum_over_fields(v_i)^2 - sum_over_fields(v_i^2)
 * which gives pairwise interaction vectors of shape (batch, embed_dim)
 */
class FMInteraction(
  embedDim: Int
) extends Module {

  def forward(embeddings: Tensor): Tensor = {
    // embeddings: (batch_size, num_fields, embed_dim)
    val twoScalar = new Scalar(2.0f)
    val halfScalar = new Scalar(0.5f)

    // sum of squared: (batch, embed_dim)
    val squaredSum = torch.pow(embeddings, twoScalar).sum(1)
    // squared sum: (batch, embed_dim)
    val sumSquared = torch.pow(embeddings.sum(1), twoScalar)
    // interaction: (batch, embed_dim) - NOT summing at the end
    sumSquared.sub(squaredSum).mul(halfScalar)
  }
}