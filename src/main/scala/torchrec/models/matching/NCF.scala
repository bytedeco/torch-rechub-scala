package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Neural Collaborative Filtering (NCF)
 *
 * Combines Generalized Matrix Factorization (GMF) with MLP for
 * learning user-item interaction patterns.
 *
 * Architecture:
 *   User Embedding (GMF) + Item Embedding (GMF) → Element-wise Product
 *   User Embedding + Item Embedding → MLP → Concatenate
 *   Concatenate → Final MLP → Output
 *
 * Reference: "Neural Collaborative Filtering" (He et al., WWW 2017)
 *
 * @param features     List of features (user + item)
 * @param userFieldIdx Index of the user feature field
 * @param itemFieldIdx Index of the item feature field
 * @param embedDim     Embedding dimension
 * @param mlpDims     Hidden layer dimensions for the MLP tower
 * @param dropout     Dropout rate
 * @param device      Device to run on
 */
class NCF(
  features: List[Feature],
  userFieldIdx: Int = 0,
  itemFieldIdx: Int = 1,
  embedDim: Int = 8,
  mlpDims: List[Long] = List(64L, 32L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(userFieldIdx != itemFieldIdx, "userFieldIdx and itemFieldIdx must be different")
  require(embedDim > 0, "embedDim must be positive")

  // Embedding layer
  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  // Number of fields
  private val numFields = features.collect { case f: SparseFeature => 1 }.size
  private val totalSparseDim = Features.calcSparseDim(features)

  // MLP for NCF part
  // The MLP's output dimension should match the last element of mlpDims
  private val mlp = new MLP(totalSparseDim, mlpDims, mlpDims.last, "relu", dropout, device = device)
  register_module("mlp", mlp)

  // Final fusion layer: GMF output (embedDim) + MLP output (64) = embedDim + 64
  private val finalDim = embedDim + mlpDims.last
  private val finalLinear = new LinearImpl(finalDim, 1)
  register_module("final_linear", finalLinear)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    mlp.to(dev, false)
    finalLinear.to(dev, false)
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Get embeddings: (batch, num_fields * embed_dim)
    val embeddings = embeddingLayer.forward(sparseFeats)
    val batchSize = embeddings.size(0).toInt

    // Reshape to (batch, num_fields, embed_dim)
    val emb3D = embeddings.view(batchSize, numFields, embedDim)

    // Extract user and item embeddings
    val userEmb = emb3D.select(1, userFieldIdx)  // (batch, embed_dim)
    val itemEmb = emb3D.select(1, itemFieldIdx)  // (batch, embed_dim)

    // GMF: element-wise product of user and item embeddings
    val gmfOut = userEmb.mul(itemEmb)  // (batch, embed_dim)

    // MLP: concatenate user and item embeddings
    val concatEmb = torch.cat(new TensorVector().push_back(userEmb).push_back(itemEmb), 1)  // (batch, 2 * embed_dim)
    val mlpOut = mlp.forward(concatEmb)  // (batch, 64)

    // Concatenate GMF and MLP outputs
    val combined = torch.cat(new TensorVector().push_back(gmfOut).push_back(mlpOut), 1)  // (batch, embed_dim + 64)

    // Final linear layer
    val logits = finalLinear.forward(combined)
    logits.squeeze(1)
  }

  /**
   * Get user embedding representation
   */
  def userForward(sparseFeats: Map[String, Tensor]): Tensor = {
    val embeddings = embeddingLayer.forward(sparseFeats)
    val batchSize = embeddings.size(0).toInt
    val emb3D = embeddings.view(batchSize, numFields, embedDim)
    emb3D.select(1, userFieldIdx)
  }

  /**
   * Get item embedding representation
   */
  def itemForward(sparseFeats: Map[String, Tensor]): Tensor = {
    val embeddings = embeddingLayer.forward(sparseFeats)
    val batchSize = embeddings.size(0).toInt
    val emb3D = embeddings.view(batchSize, numFields, embedDim)
    emb3D.select(1, itemFieldIdx)
  }
}
