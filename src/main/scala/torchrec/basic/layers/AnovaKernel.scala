package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import torchrec.utils.DeviceSupport

/**
 * Anova Kernel for computing high-order polynomial interactions.
 *
 * Implements the ANOVA kernel expansion for polynomial interactions of configurable order.
 * Each order-k term represents the interaction between k features.
 *
 * Reference: "Factorization Machines" (Rendle, 2010) - High-order extension
 *
 * The kernel uses a dynamic programming approach with cumsum for efficiency:
 *   a_t = sum_{S subset of {1..m}, |S|=t} prod_{i in S} x_i
 *
 * @param order      The polynomial order (2 = FM, 3 = 3rd order, etc.)
 * @param embedDim   Embedding dimension
 * @param reduceSum  If true, sum over the final dimension; otherwise return full tensor
 * @param device     Device to run on
 */
class AnovaKernel(
  order: Int,
  embedDim: Int,
  reduceSum: Boolean = true,
  device: String = DeviceSupport.backend
) extends Module {

  require(order >= 2, s"order must be >= 2, got $order")
  require(embedDim > 0, s"embedDim must be positive, got $embedDim")

  // No learnable parameters - pure tensor operations

  def forward(embeddings: Tensor): Tensor = {
    // embeddings: (batch, num_fields, embed_dim)
    val batchSize = embeddings.size(0)
    val numFields = embeddings.size(1)
    val eDim = embeddings.size(2)

    val dev = embeddings.device()

    // Safer combinatorial implementation: compute sum over all combinations of
    // `order` fields of the elementwise product of their embeddings.
    // This avoids in-place ops and autograd versioning issues.
    // Note: this is more expensive for large numFields/order but is robust.
    val floatOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
    val zero = torch.zeros(Array(batchSize, eDim), floatOpts).to(dev, ScalarType.Float)

    // Generate all combinations of field indices of size `order`
    val indices = (0 until numFields.toInt).toArray
    def combos(arr: Array[Int], k: Int): Seq[Array[Int]] = {
      if (k == 0) Seq(Array.emptyIntArray)
      else if (arr.length < k) Seq.empty
      else {
        arr.indices.flatMap { i =>
          combos(arr.slice(i + 1, arr.length), k - 1).map(r => Array(arr(i)) ++ r)
        }
      }
    }

    val comb = combos(indices, order)
    var acc = zero
    for (c <- comb) {
      // Start with embedding for first index
      var prod = embeddings.narrow(1, c(0), 1).squeeze(1)
      var idx = 1
      while (idx < c.length) {
        val t = embeddings.narrow(1, c(idx), 1).squeeze(1)
        prod = prod.mul(t)
        idx += 1
      }
      acc = acc.add(prod)
    }

    val result = acc
    if (reduceSum) {
      // return (batch, 1)
      result.sum(1L).unsqueeze(1)
    } else {
      // return (batch, embed_dim)
      result
    }
  }
}
