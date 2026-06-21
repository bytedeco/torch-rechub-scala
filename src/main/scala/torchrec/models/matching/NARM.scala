package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import torchrec.Implicits._

/**
 * NARM - Neural Attentive Sequential Recommender
 * Reference: "Neural Attentive Sequential Recommender" (Li et al., 2017)
 */
class NARM(
  features: List[Feature],
  embedDim: Int = 8,
  hiddenDim: Int = 8,
  attentionDim: Int = 8,
  device: String = DeviceSupport.backend
) extends Module {

  private val embedding = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embedding)

  // Determine sequence feature name from provided features (fallback to "seq_feat")
  private val seqFeatureName: String = features.collect { case f: SequenceFeature => f.name }.headOption.getOrElse("seq_feat")

  private val encoder = new MLP(embedDim, List(hiddenDim.toLong), hiddenDim, "relu", 0f, device = device)
  register_module("encoder", encoder)

  // Attention MLP takes pooled sequence embedding (embedDim) as input and outputs hiddenDim
  private val attention = new MLP(embedDim, List(attentionDim), hiddenDim, "relu", 0f, device = device)
  register_module("attention", attention)

  private val output = new LinearImpl(hiddenDim * 2, embedDim)
  output.to(new Device(device),false)
  register_module("output", output)

  def forward(sequence: Tensor): Tensor = {
    // Get raw sequence embeddings (batch, seqLen, embedDim)
    val raw = embedding.forwardSeqRaw(Map(seqFeatureName -> sequence))
    val emb = if (raw.dim() == 4L) raw.squeeze(1L) else raw
    // Pool per-sequence (mean) then encode and repeat per position as original implementation intended
    val pooled = emb.mean(1) // (batch, embedDim)
    val encodedPooled = encoder.forward(pooled) // (batch, hiddenDim)
    val encoded = encodedPooled.unsqueeze(1).repeat(1, emb.size(1), 1)
    val last = encoded.select(1, encoded.size(1) - 1)
    val attn = attention.forward(pooled)
    val combined = torch.cat(new TensorVector(last, attn), 1L)
    output.forward(combined)
  }
}
