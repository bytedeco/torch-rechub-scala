package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * AutoInt: Automatic Feature Interaction via Attentive Multi-Head Self-Attention
 * Reference: RecSys 2019
 */
class AutoInt(
  features: List[Feature],
  embedDim: Int = 8,
  numAttnHeads: Int = 2,
  numLayers: Int = 2,
  mlpDims: List[Long] = List(128L, 64L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  private val numFields = features.collect { case f: SparseFeature => 1 }.size

  // Multi-head self-attention layers
  private val attentionLayers = (0 until numLayers).map { i =>
    val layer = new MultiHeadSelfAttention(embedDim, numAttnHeads, dropout, device)
    register_module(s"attention_$i", layer)
    layer
  }

  // MLP
  private val mlp = new MLP(numFields * embedDim, mlpDims.map(_.toLong), 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    val embeddings = embeddingLayer.forward(sparseFeats)
    // Reshape to (batch, num_fields, embed_dim)
    val batchSize = embeddings.size(0)
    val reshaped = embeddings.view(batchSize, numFields, embedDim)

    // Apply attention layers
    var output = reshaped
    attentionLayers.foreach { attn =>
      output = attn.forward(output).add(output)  // Residual connection
    }

    // Flatten and MLP
    val flattened = output.flatten(1, 2)
    val logits = mlp.forward(flattened)
    logits
  }
}

/**
 * Multi-Head Self-Attention
 */
class MultiHeadSelfAttention(
  embedDim: Int,
  numHeads: Int,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0, s"embedDim must be divisible by numHeads")
  private val headDim = embedDim / numHeads

  private val query = new LinearImpl(embedDim, embedDim)
  private val key = new LinearImpl(embedDim, embedDim)
  private val value = new LinearImpl(embedDim, embedDim)
  private val output = new LinearImpl(embedDim, embedDim)
  private val dropoutLayer = new DropoutImpl(dropout)

  register_module("query", query)
  register_module("key", key)
  register_module("value", value)
  register_module("output", output)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    query.to(dev, false)
    key.to(dev, false)
    value.to(dev, false)
    output.to(dev, false)
    dropoutLayer.to(dev, false)
  }

  def forward(x: Tensor): Tensor = {
    // x: (batch, num_fields, embed_dim)
    val batchSize = x.size(0)
    val numFields = x.size(1)

    val q = query.forward(x).view(batchSize, numFields, numHeads, headDim).transpose(1, 2)
    val k = key.forward(x).view(batchSize, numFields, numHeads, headDim).transpose(1, 2)
    val v = value.forward(x).view(batchSize, numFields, numHeads, headDim).transpose(1, 2)

    // Scaled dot-product attention
    val scale = new Scalar(scala.math.sqrt(headDim.toFloat).toFloat)
    val scores = torch.matmul(q, k.transpose(2, 3)).div(scale)
    val attnWeights = scores.softmax(-1)
    val attended = torch.matmul(dropoutLayer.forward(attnWeights), v)

    // Reshape back
    val reshaped = attended.transpose(1, 2).contiguous().view(batchSize, numFields, embedDim)
    output.forward(reshaped)
  }
}
