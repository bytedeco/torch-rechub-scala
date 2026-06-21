package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Deep Field-weighted Factorization Machine (DeepFFM)
 * Reference: "Deep Field-Weighted Factorization Machine"
 * Alibaba, IJCAI 2018
 *
 * @param features List of sparse features
 * @param embedDim Embedding dimension
 * @param fieldNum Number of fields (used for field-aware FM)
 * @param mlpDims Dimensions for deep MLP layers
 * @param dropout Dropout rate
 * @param device Device to run on
 */
class DeepFFM(
  features: List[Feature],
  embedDim: Int = 8,
  fieldNum: Int,
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")

  // Embedding layer for sparse features
  private val embedding = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embedding)

  // Field-aware FM layer
  private val ffm = new FFM(embedDim, fieldNum, device)
  register_module("ffm", ffm)

  // Deep part: MLP
  private val sparseDim = Features.calcSparseDim(features)
  private val mlp = new MLP(sparseDim, mlpDims, 1, "relu", dropout, false, device = device)
  register_module("mlp", mlp)

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Get embeddings:
    // - For FFM (field-aware) we need a 3D tensor: (batch, num_fields, embed_dim)
    // - For the deep MLP we use the flattened 2D representation
    val embeddings3d = embedding.forward3D(sparseFeats)
    val embeddings = embedding.forward(sparseFeats)

    // FFM (2nd-order field-aware interactions) expects 3D embeddings
    val ffmOut = ffm.forward(embeddings3d)

    // Deep part uses flattened embeddings
    val mlpOut = mlp.forward(embeddings)

    // Combine FFM and deep outputs
    val logits = ffmOut.add(mlpOut)
    logits
  }
}

/**
 * Field-weighted Factorization Machine (FFM)
 * Field-aware FM that captures field-level interactions
 *
 * @param embedDim Embedding dimension
 * @param fieldNum Number of fields
 * @param device Device to run on
 */
class FFM(
  embedDim: Int,
  fieldNum: Int,
  device: String = DeviceSupport.backend
) extends Module {

  def forward(embeddings: Tensor): Tensor = {
    // embeddings: (batch, num_fields, embed_dim)
    val batchSize = embeddings.size(0)
    val numFields = embeddings.size(1)

    // First order: sum of embeddings
    val firstOrder = embeddings.sum(1) // (batch, embed_dim)

    // Second order: field-aware interactions
    val twoScalar = new Scalar(2.0f)
    val squaredSum = torch.pow(embeddings, twoScalar).sum(1)
    val sumSquared = torch.pow(embeddings.sum(1), twoScalar)

    // Field-aware interaction: 0.5 * (sum^2 - squared_sum)
    val halfScalar = new Scalar(0.5f)
    val interactions = sumSquared.sub(squaredSum).mul(halfScalar)

    // FM output: first order + interactions
    firstOrder.add(interactions).sum(1).unsqueeze(1)
  }
}

/**
 * Field-Attentive Deep Field-weighted FM (FatDeepFFM)
 * Variant that uses all features as fields
 *
 * @param features List of sparse features
 * @param embedDim Embedding dimension
 * @param mlpDims Dimensions for deep MLP layers
 * @param dropout Dropout rate
 * @param device Device to run on
 */
class FatDeepFFM(
  features: List[Feature],
  embedDim: Int = 8,
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends DeepFFM(
  features,
  embedDim,
  features.collect { case f: SparseFeature => 1 }.size,
  mlpDims,
  dropout,
  device
)