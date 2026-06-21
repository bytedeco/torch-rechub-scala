package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * Factorization Machine supported Neural Network (FNN)
 *
 * Key idea: Use pre-trained FM embeddings as input to a DNN.
 * The FM component provides high-quality sparse feature representations
 * learned via factorization, which are then fed into a deep network
 * for higher-order interaction learning.
 *
 * Architecture:
 *   Sparse Input → FM Embeddings → MLP → Output
 *
 * Reference: "Factorization Machines with Follow-The-Regularized-Leader for CTR Prediction"
 *            (Zhang et al., 2016)
 *
 * @param features   List of sparse features
 * @param embedDim  Embedding dimension
 * @param mlpDims   Hidden layer dimensions for DNN
 * @param dropout   Dropout rate
 * @param device    Device to run on
 */
class FNN(
  features: List[Feature],
  embedDim: Int = 8,
  mlpDims: List[Long] = List(256L, 128L, 64L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")

  // Embedding layer (FM-style embeddings)
  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  // MLP on top of FM embeddings
  private val sparseDim = Features.calcSparseDim(features)
  private val mlp = new MLP(sparseDim, mlpDims, 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Get FM-style embeddings: (batch, num_fields * embed_dim)
    val embeddings = embeddingLayer.forward(sparseFeats)
    // Forward through MLP -> keep (batch,1) shape for consistency
    mlp.forward(embeddings)
  }
}
