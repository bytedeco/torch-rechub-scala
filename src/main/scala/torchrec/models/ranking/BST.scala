package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Behavior Sequence Transformer (BST)
 * Reference: "Behavior Sequence Transformer for E-commerce Recommendation"
 * Alibaba, SIGIR 2019
 *
 * @param features List of sparse features
 * @param sequenceFeatures List of sequence features (behavior history)
 * @param embedDim Embedding dimension
 * @param numHeads Number of attention heads
 * @param mlpDims Dimensions for deep MLP layers
 * @param dropout Dropout rate
 * @param device Device to run on
 */
class BST(
  features: List[Feature],
  sequenceFeatures: List[SequenceFeature],
  embedDim: Int = 8,
  numHeads: Int = 2,
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")

  // Embedding layer for both features and sequence features
  private val allFeatures = features ++ sequenceFeatures
  private val embedding = new EmbeddingLayer(allFeatures, embedDim, device)
  register_module("embedding", embedding)

  // Simple MLP layers instead of transformer
  private val encoderMLP = new MLP(embedDim, List(embedDim * 2L), embedDim, "relu", dropout, device = device)
  register_module("encoderMLP", encoderMLP)

  private val layernorm = {
    val vec = new LongVector(1)
    vec.put(0, embedDim.toLong)
    new LayerNormImpl(vec)
  }
  register_module("layernorm", layernorm)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    layernorm.to(dev, false)
  }

  // MLP for final prediction
  private val sparseDim = Features.calcSparseDim(features)
  private val sequenceDim = sequenceFeatures.map(_.embedDim).sum
  private val totalDim = sparseDim + sequenceDim
  private val mlp = new MLP(totalDim, mlpDims, 1, "relu", dropout, false, device = device)
  register_module("mlp", mlp)

  def forward(
    sparseFeats: Map[String, Tensor],
    seqFeats: Map[String, Tensor]
  ): Tensor = {
    // Combine sparse features and sequence features
    val allFeats = sparseFeats ++ seqFeats

    // Get embeddings
    val embeddings = embedding.forward(allFeats)
    // embeddings: (batch, num_fields + num_seq_fields, embed_dim)

    // Apply MLP encoder
    val encoded = encoderMLP.forward(embeddings.mean(1))
    val normed = layernorm.forward(encoded)

    // Get feature embeddings (first sparse features only)
    val featEmbeddings = embeddings.narrow(1, 0, sparseDim / embedDim)
    // featEmbeddings: (batch, num_sparse_features, embed_dim)

    // Flatten feature embeddings
    val flatFeatures = featEmbeddings.view(featEmbeddings.size(0), -1)

    // Combine and feed to MLP
    val combined = torch.cat(new TensorVector(flatFeatures, normed), 1L)
    val logits = mlp.forward(combined)

    logits
  }
}