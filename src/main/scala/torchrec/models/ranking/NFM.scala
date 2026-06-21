package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Neural Factorization Machine (NFM)
 * Combines:
 *   - Bi-Interaction Pooling: captures 2nd-order feature interactions
 *   - MLP: learns higher-order (implicit) feature interactions on top of bi-interaction output
 *
 * Reference: "Neural Factorization Machines for Sparse Predictive Analytics" (SIGIR 2017)
 *
 * @param features   List of sparse features
 * @param embedDim   Embedding dimension
 * @param mlpDims    Hidden layer dimensions for the MLP (on top of bi-interaction pooling)
 * @param dropout    Dropout rate
 * @param device     Device to run on
 */
class NFM(
  features: List[Feature],
  embedDim: Int = 8,
  mlpDims: List[Long] = List(128L, 64L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")

  // Embedding layer
  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  // Bi-Interaction Pooling (captures 2nd-order interactions)
  private val biInteractionPool = new BiInteractionPooling(device)
  register_module("biInteraction", biInteractionPool)

  // MLP on top of bi-interaction output
  private val mlp = new MLP(embedDim, mlpDims, 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Get embeddings as 3D tensor: (batch, num_fields, embed_dim)
    // NFM relies on field dimension for bi-interaction pooling, use forward3D
    val embeddings = embeddingLayer.forward3D(sparseFeats, sequenceFeats = Map.empty)

    // 1st-order: sum of embeddings
    val firstOrder = embeddings.sum(1)  // (batch, embed_dim)

    // 2nd-order: bi-interaction pooling
    val biOut = biInteractionPool.forward(embeddings)  // (batch, embed_dim)

    // Combine 1st and 2nd order: (batch, embed_dim)
    val combined = firstOrder.add(biOut)

    // MLP for implicit higher-order interactions
    mlp.forward(combined).squeeze(1)
  }
}
