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
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.utils.DeviceSupport

/**
 * Full Factorization Machine — 1st-order (linear) + 2nd-order (interaction) terms.
 *
 * FM equation: y = w_0 + sum_i w_i*x_i + sum_i sum_j<i <V_i, V_j> * x_i * x_j
 *
 * This layer expects pre-embedded inputs (batch, num_fields, embed_dim).
 * The 1st-order term is the sum of embeddings (equivalent to learning per-field linear weights
 * when embedding tables are used). The 2nd-order term uses the efficient formula:
 *   0.5 * (sum(S)^2 - sum(S^2)) where S = sum_i V_i * x_i
 *
 * Reference: "Factorization Machines" (Rendle, ICDM 2010)
 *
 * @param embedDim Embedding dimension (used only for consistency)
 * @param device  Device
 */
class FactorizationMachineLayer(
  embedDim: Int = 8,
  device: String = DeviceSupport.backend
) extends Module {

  // No learnable parameters — FM interactions are computed via embeddings

  override def forward(embeddings: Tensor): Tensor = {
    // embeddings: (batch, num_fields, embed_dim)
    val batchSize = embeddings.size(0)
    val numFields = embeddings.size(1)

    // 1st-order: sum of embeddings (each field contributes its embedding vector)
    // This captures the linear (first-order) feature weights
    val firstOrder = embeddings.sum(1)  // (batch, embed_dim)

    // 2nd-order: efficient FM interaction formula
    // sum_i sum_j <V_i, V_j> = 0.5 * (||sum_i V_i||^2 - sum_i ||V_i||^2)
    val two = new Scalar(2.0f)
    val half = new Scalar(0.5f)

    val squaredSum = torch.pow(embeddings, two).sum(1)  // (batch, embed_dim)
    val sumSquared = torch.pow(embeddings.sum(1), two)  // (batch, embed_dim)
    val secondOrder = sumSquared.sub(squaredSum).mul(half)  // (batch, embed_dim)

    // Combine 1st + 2nd order
    // Output shape: (batch, embed_dim) — this is a vector, not scalar
    // For CTR prediction, this would typically be summed or fed to a linear layer
    firstOrder.add(secondOrder)
  }

  /** Compute scalar FM output (sum over embedding dimension) */
  def forwardScalar(embeddings: Tensor): Tensor = {
    forward(embeddings).sum(1).unsqueeze(1)  // (batch, 1)
  }
}
