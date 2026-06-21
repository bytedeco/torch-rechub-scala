package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * High-Order Factorization Machine (HoFM)
 *
 * Extends standard 2nd-order FM to arbitrary high-order polynomial interactions.
 * Uses the AnovaKernel to efficiently compute interactions up to the specified order.
 *
 * Architecture:
 *   Sparse Input → EmbeddingLayer → [AnovaKernel for order >= 3] + FM (order 2) + MLP → Output
 *
 * Reference: "Factorization Machines" (Rendle, 2010) - High-order extension
 *
 * @param features   List of sparse features
 * @param embedDim   Embedding dimension
 * @param order      Polynomial order (2 = standard FM, 3 = 3rd order, etc.)
 * @param mlpDims    Hidden layer dimensions for the MLP
 * @param dropout    Dropout rate
 * @param device     Device to run on
 */
class HoFM(
  features: List[Feature],
  embedDim: Int = 8,
  order: Int = 3,
  mlpDims: List[Long] = List(128L, 64L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(order >= 2, s"order must be >= 2, got $order")
  require(embedDim > 0, s"embedDim must be positive, got $embedDim")

  private val numFields = features.collect { case f: SparseFeature => 1 }.size
  require(numFields >= order, s"numFields ($numFields) must be >= order ($order)")

  // Embedding layer
  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  // FM for 2nd-order interactions
  // Use FMInteraction which returns a vector of shape (batch, embed_dim)
  private val fm = new torchrec.basic.layers.FMInteraction(embedDim)
  register_module("fm", fm)

  // AnovaKernel for order >= 3 interactions
  // Note: We use a fixed embedDim for simplicity. In the original HoFM,
  // the embedding is extended to embedDim * (order - 1) to store
  // different parts for different orders.
  private val anovaKernel = new AnovaKernel(order, embedDim, reduceSum = false, device)
  register_module("anova_kernel", anovaKernel)

  // MLP input dimension:
  // - FM (2nd order): embedDim output
  // - AnovaKernel (order >= 3): embedDim output per kernel
  // - Total: embedDim * (order - 1) if order >= 3, else embedDim
  private val mlpInputDim = if (order >= 3) embedDim * (order - 1) else embedDim

  // MLP
  private val mlp = new MLP(mlpInputDim, mlpDims, 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  if (device != "cpu") {
    mlp.to(new org.bytedeco.pytorch.Device(device), false)
    fm.to(new org.bytedeco.pytorch.Device(device), false)
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Get embeddings: (batch, num_fields, embed_dim)
    val embeddings = embeddingLayer.forward3D(sparseFeats)
    val batchSize = embeddings.size(0).toInt

    // Accumulate interaction outputs
    val interactionOutputs = collection.mutable.ListBuffer[Tensor]()

    if (order >= 2) {
      // FM 2nd-order: (batch, embed_dim)
      val fmOut = fm.forward(embeddings)
      interactionOutputs += fmOut
    }

    if (order >= 3) {
      // AnovaKernel for orders 3..order
      // The AnovaKernel returns (batch, embed_dim) per order
      // We use a single kernel with the maximum order
      val anovaOut = anovaKernel.forward(embeddings)  // (batch, embed_dim)
      interactionOutputs += anovaOut
    }

    // Concatenate all interaction outputs: (batch, mlpInputDim)
    val combined = if (interactionOutputs.size == 1) {
      interactionOutputs.head
    } else {
      val tensorVec = new TensorVector()
      val targetDev = interactionOutputs.head.device()
      interactionOutputs.foreach { t =>
        val onDev = if (t.device().equals(targetDev)) t else t.to(targetDev, t.dtype())
        tensorVec.push_back(onDev)
      }
      torch.cat(tensorVec, 1L)
    }

    // Flatten if needed
    val mlpInput = if (combined.dim() == 3) {
      combined.view(batchSize, -1)
    } else if (combined.dim() == 2 && combined.size(1) != mlpInputDim) {
      combined  // Already (batch, embed_dim) or similar
    } else {
      combined
    }

    // MLP
    mlp.forward(mlpInput).squeeze(1)
  }
}
