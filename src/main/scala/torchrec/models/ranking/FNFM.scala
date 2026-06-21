package torchrec.models.ranking

import torchrec.basic.features.*
import torchrec.basic.layers.*
import torchrec.Implicits.*
import torchrec.utils.DeviceSupport
import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Field-aware Neural Factorization Machine (FNFM)
 *
 * Combines field-aware FM interactions with a deep neural network.
 * In field-aware FM, each field has its own latent vector for each other field,
 * allowing the model to learn different feature interactions per field pair.
 *
 * Architecture:
 *   Sparse Input → Field-aware Embeddings → FFM Inner Products → MLP → Output
 *
 * Reference: "Field-aware Factorization Machines for CTR Prediction" (Criteo, RecSys 2016)
 *            + "Field-aware Neural Factorization Machine for CTR Prediction"
 *
 * @param features   List of sparse features
 * @param embedDim  Embedding dimension
 * @param mlpDims   Hidden layer dimensions for DNN
 * @param dropout   Dropout rate
 * @param device    Device to run on
 */
class FNFM(
  features: List[Feature],
  embedDim: Int = 8,
  mlpDims: List[Long] = List(256L, 128L, 64L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")

  private val numFields = features.collect { case f: SparseFeature => 1 }.size
  require(numFields >= 2, "FNFM requires at least 2 sparse features")

  // Standard per-field embedding (shared)
  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  // Number of field pairs
  private val numPairs = numFields * (numFields - 1) / 2

  // FFM interaction output dimension
  // Each field pair contributes an embedDim-dimensional vector from inner product
  private val ffmOutputDim = numPairs * embedDim

  // MLP on FFM interaction features (use MLP's built-in BN option)
  private val mlp = new MLP(ffmOutputDim, mlpDims, 1, "relu", dropout, true, device = device)
  register_module("mlp", mlp)

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Get embeddings: (batch, num_fields, embed_dim)
    val embeddings = embeddingLayer.forward3D(sparseFeats)

    // Compute field-aware interactions: for each pair (i,j) where i<j
    // interaction = sum over k of V_{j,k} * V_{i,k}
    // This is the inner product of field i's and field j's embedding vectors
    val interactions = collection.mutable.ListBuffer[Tensor]()
    var i = 0
    while (i < numFields) {
      var j = i + 1
      while (j < numFields) {
        val vi = embeddings.narrow(1, i, 1).squeeze(1)  // (batch, embed_dim)
        val vj = embeddings.narrow(1, j, 1).squeeze(1)  // (batch, embed_dim)
        // Inner product: (batch, embed_dim) dot (batch, embed_dim) -> (batch,)
        // For FNFM we keep the full embed_dim as a vector
        val ip = vi.mul(vj)  // element-wise product -> (batch, embed_dim)
        interactions += ip
        j += 1
      }
      i += 1
    }

    // Concatenate all pairwise interactions: (batch, num_pairs * embed_dim)
    val ffmFeatures = if (interactions.isEmpty) {
      val opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
      torch.zeros(Array(embeddings.size(0).toLong, 1L), opts).to(embeddings.device(), ScalarType.Float)
    } else {
      val vec = new TensorVector()
      val targetDev = interactions.head.device()
      interactions.foreach { t =>
        val onDev = if (t.device().equals(targetDev)) t else t.to(targetDev, t.dtype())
        vec.push_back(onDev)
      }
      torch.cat(vec, 1L)
    }

    // MLP
    mlp.forward(ffmFeatures).squeeze(1)
  }
}
