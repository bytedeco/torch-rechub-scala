package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Logistic Regression Model
 *
 * The simplest CTR model — learns a linear combination of feature embeddings.
 * Provides a strong baseline for CTR prediction.
 *
 * Architecture:
 *   Sparse Input → EmbeddingLayer → Sum Pooling → Output
 *
 * @param features   List of sparse features
 * @param embedDim  Embedding dimension (for feature embeddings)
 * @param device    Device to run on
 */
class LR(
  features: List[Feature],
  embedDim: Int = 8,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")

  // Embedding layer
  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  // Sum pooling gives the linear combination
  // Output is already a scalar logit per batch element

  if (device != "cpu") {
    embeddingLayer.to(device)
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Get embeddings: (batch, num_fields, embed_dim)
    val embeddings = embeddingLayer.forward3D(sparseFeats)

    // Sum over fields and embedding dimension → scalar per batch
    // embeddings: (batch, num_fields, embed_dim)
    // sum over fields -> (batch, embed_dim), then sum over embed_dim -> (batch,)
    val summed = embeddings.sum(1L).sum(1L).unsqueeze(1) // (batch,1)
    summed
  }
}
