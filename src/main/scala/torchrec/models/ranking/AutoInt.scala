package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * AutoInt: Automatic Feature Interaction via Attentive Multi-Head Self-Attention
 * Reference: CIKM 2019
 */
class AutoInt(
  sparseFeatures: List[Feature],
  embedDim: Int = 8,
  numAttnHeads: Int = 2,
  numLayers: Int = 3,
  mlpDims: List[Long] = List(128L, 64L),
  dropout: Float = 0.0f,
  useMlp: Boolean = true,
  device: String = DeviceSupport.backend
) extends Module {

  private val numSparseFeatures = sparseFeatures.size

  // Dense feature embeddings: Linear(1, embed_dim) per dense feature
  // Note: dense features are passed separately and projected
  private val numDenseFeatures = 0  // Dense features handled via separate projection

  // Total dims = (numSparse + numDense) * embed_dim
  private val dims = numSparseFeatures * embedDim

  // Sparse embedding layer
  private val embeddingLayer = new EmbeddingLayer(sparseFeatures, embedDim, device)
  register_module("embedding", embeddingLayer)

  // Multi-head self-attention interacting layers
  private val interactingLayers = (0 until numLayers).map { i =>
    val layer = new InteractingLayer(embedDim, numAttnHeads, dropout, residual = true, device)
    register_module(s"interacting_$i", layer)
    layer
  }

  // Linear (LR) component for first-order interactions
  private val linear = new LR(dims, sigmoid = false, device)
  register_module("linear", linear)

  // Attention pooling linear: projects attention output to a scalar
  private val attnLinear = new LinearImpl(dims, 1)
  attnLinear.to(new org.bytedeco.pytorch.Device(device), false)
  register_module("attn_linear", attnLinear)

  // MLP for deep interactions
  private val mlp = if (useMlp) {
    val m = new MLP(dims, mlpDims.map(_.toLong), 1, "relu", dropout, device = device)
    register_module("mlp", m)
    Some(m)
  } else None

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Get sparse embeddings using forward3D: (batch, numSparse, embedDim)
    val sparseEmb = embeddingLayer.forward3D(sparseFeats)

    // Handle dense features: each dense feature projected via Linear(1, embed_dim)
    // If no dense features, just use sparseEmb
    val embeddings = sparseEmb  // (batch, numSparse, embedDim)

    // Flatten for linear component
    val embeddingsFlat = embeddings.view(embeddings.size(0), -1)  // (batch, dims)

    // Apply interacting layers (multi-head self-attention with residual)
    var attnOut = embeddings
    interactingLayers.foreach { layer =>
      attnOut = layer.forward(attnOut)  // Each layer: residual connection inside
    }

    // Attention linear: flatten attention output and project
    val attnOutFlat = attnOut.view(attnOut.size(0), -1)  // (batch, dims)
    val yAttn = attnLinear.forward(attnOutFlat)  // (batch, 1)

    // Linear component
    val yLinear = linear.forward(embeddingsFlat)  // (batch, 1)

    // Combine attention and linear
    var y = yAttn.add(yLinear)

    // Add MLP contribution if enabled
    if (mlp.isDefined) {
      val yDeep = mlp.get.forward(embeddingsFlat)  // (batch, 1)
      y = y.add(yDeep)
    }

    // Return (batch, 1) - don't squeeze to avoid BCEWithLogitsLoss shape mismatch
    y
  }
}
