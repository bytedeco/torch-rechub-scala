package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * xDeepFM: eXtreme Deep Factorization Machine
 * Combines:
 *   - CIN (Compressed Interaction Network) for explicit high-order feature interactions
 *   - FM for 2nd-order feature interactions
 *   - DNN/MLP for implicit high-order feature interactions
 *
 * Reference: "xDeepFM: Combining Explicit and Implicit Feature Interactions
 *            in Recommender Systems" (KDD 2018)
 *
 * @param features        List of sparse features
 * @param embedDim        Embedding dimension
 * @param crossLayerSizes  Output sizes of each CIN layer (e.g. List(128, 64, 32))
 * @param mlpDims          Hidden layer dimensions for DNN part
 * @param splitHalf        If true, halve CIN output size between layers (default: true)
 * @param dropout          Dropout rate
 * @param device           Device to run on
 */
class xDeepFM(
  features: List[Feature],
  embedDim: Int = 8,
  crossLayerSizes: List[Int] = List(128, 64),
  mlpDims: List[Long] = List(256L, 128L, 64L),
  splitHalf: Boolean = true,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")

  private val numFields = features.collect { case f: SparseFeature => 1 }.size
  require(numFields >= 2, "xDeepFM requires at least 2 sparse features")

  // Embedding layer
  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  // CIN for explicit high-order interactions
  private val cin = new CINFixed(numFields, embedDim, crossLayerSizes, splitHalf, device)
  register_module("cin", cin)

  // 2nd-order FM component
  private val fmLayer = new FM(embedDim, device)
  register_module("fm", fmLayer)

  // DNN (MLP) for implicit high-order interactions
  private val sparseDim = numFields * embedDim
  private val mlp = new MLP(sparseDim, mlpDims, 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  // Linear (1st-order) features
  private val linearWeight = {
    val w = new org.bytedeco.pytorch.LinearImpl(sparseDim, 1)
    register_module("linear", w)
    w.to(new org.bytedeco.pytorch.Device(device), false)
    w
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Get embeddings: (batch, num_fields, embed_dim) using forward3D
    val embeddings3D = embeddingLayer.forward3D(sparseFeats)
    val batchSize = embeddings3D.size(0)

    // 1) CIN: explicit high-order feature interactions (scalar per batch)
    val cinOut = cin.forward(embeddings3D)  // (batch,)

    // 2) FM: 2nd-order feature interactions (scalar per batch)
    val fmOut = fmLayer.forward(embeddings3D).squeeze(1)  // (batch,)

    // 3) Flatten 3D embeddings to 2D for DNN and linear
    val flatEmbeddings = embeddings3D.view(batchSize, sparseDim)  // (batch, num_fields * embed_dim)

    // 4) DNN: implicit high-order interactions
    val mlpOut = mlp.forward(flatEmbeddings).squeeze(1)  // (batch,)

    // 5) Linear: 1st-order contributions
    val linearOut = linearWeight.forward(flatEmbeddings).squeeze(1)  // (batch,)

    // Combine all: logit = linear + cin + fm + mlp
    linearOut.add(cinOut).add(fmOut).add(mlpOut)
  }
}
