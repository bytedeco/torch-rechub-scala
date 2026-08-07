package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.javacpp.LongPointer
import org.bytedeco.pytorch._
import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.nn.modules._
import org.bytedeco.pytorch.nn.modules.container._
import org.bytedeco.pytorch.nn.options._
import org.bytedeco.pytorch.optim._
import org.bytedeco.pytorch.data.datasets._
import org.bytedeco.pytorch.data.options._
import org.bytedeco.pytorch.data.sampler._
import org.bytedeco.pytorch.distributed._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import torchrec.Implicits._

/**
 * SASRec — Self-Attentive Sequential Recommendation (Kang & McAuley, ICDM'2018).
 *
 * Mirrors the Python `torch-rechub` reference: item embedding + learned positional
 * embedding (scaled by `sqrt(embed_dim)`), `num_blocks` self-attention blocks
 * each with its own `LayerNorm`s and a `PointWiseFeedForward`, then a final
 * `LayerNorm`.
 *
 * Each block is structured as four `ModuleListImpl`s (one per `LayerNorm` /
 * attention / feed-forward layer family), exactly mirroring the Python's
 * `nn.ModuleList`s — so `state_dict()` and parameter discovery see the same
 * hierarchy as the reference.
 *
 * Reference: https://arxiv.org/pdf/1808.09781.pdf
 *
 * @param sequenceFeatures SequenceFeature column list. The head drives vocab / max-len.
 * @param embedDim         Embedding dimension.
 * @param numHeads         Number of attention heads.
 * @param numLayers        Number of stacked blocks.
 * @param ffnDim           PointWiseFeedForward hidden dimension.
 * @param dropout          Dropout probability.
 * @param device           Device for parameters.
 */
class SASRec(
  sequenceFeatures: List[Feature],
  embedDim: Int = 8,
  numHeads: Int = 2,
  numLayers: Int = 2,
  ffnDim: Int = 128,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")
  require(embedDim % numHeads == 0, s"embedDim ($embedDim) must be divisible by numHeads ($numHeads)")
  require(numLayers > 0, s"numLayers must be > 0, got $numLayers")

  // The (first) sequence feature drives the item vocabulary / sequence length.
  private val seqFeature: SequenceFeature = sequenceFeatures.head match {
    case sf: SequenceFeature => sf
    case other => throw new IllegalArgumentException(
      s"SASRec expects a SequenceFeature, got: ${other.getClass.getSimpleName}")
  }

  val seqFeatureName: String = seqFeature.name

  private val vocabSize: Long = seqFeature.vocabSize
  private val maxLen: Int = seqFeature.maxLen
  private val headDim: Int = embedDim / numHeads

  // Item embedding table (padding idx 0).
  private val itemEmbedding = {
    val opts = new EmbeddingOptions(vocabSize, embedDim)
    opts.padding_idx().put(0L)
    new EmbeddingImpl(opts)
  }
  register_module("item_embedding", itemEmbedding)

  // Learned positional embedding.
  private val positionEmbedding = new EmbeddingImpl(new EmbeddingOptions(maxLen.toLong, embedDim))
  register_module("position_embedding", positionEmbedding)

  private def applyDropout(x: Tensor): Tensor = torch.dropout(x, dropout.toDouble, false)

  // Per-block LayerNorms and feed-forward modules — mirrors the four
  // `nn.ModuleList`s in the Python reference. We index them directly via
  // `.get(layer).forward(...)` rather than caching typed refs.
  private val attnLayerNorms: ModuleListImpl = new ModuleListImpl()
  private val attnQProjs: ModuleListImpl = new ModuleListImpl()
  private val attnKProjs: ModuleListImpl = new ModuleListImpl()
  private val attnVProjs: ModuleListImpl = new ModuleListImpl()
  private val attnOProjs: ModuleListImpl = new ModuleListImpl()
  private val fwdLayerNorms: ModuleListImpl = new ModuleListImpl()
  private val fwdLayers: ModuleListImpl = new ModuleListImpl()

  for (i <- 0 until numLayers) {
    val normShape = new LongVector(1); normShape.put(0, embedDim.toLong)

    // attn LayerNorm (pre-LN of the MHA block, like the Python)
    val attnLn = new LayerNormImpl(normShape)
    register_module(s"attn_layer_norm_$i", attnLn)
    attnLayerNorms.push_back(attnLn)

    // Q/K/V/out projections (functionally equivalent to nn.MultiheadAttention,
    // exposed as separate Linear projections so the forward stays in pure
    // Tensor ops and avoids the `T_TensorTensor_T` tuple round-trip).
    Seq(("q", attnQProjs), ("k", attnKProjs), ("v", attnVProjs), ("o", attnOProjs)).foreach {
      case (which, list) =>
        val proj = new LinearImpl(embedDim.toLong, embedDim.toLong)
        register_module(s"${which}_proj_$i", proj)
        list.push_back(proj)
    }

    // fwd LayerNorm
    val fwdLn = new LayerNormImpl(normShape)
    register_module(s"fwd_layer_norm_$i", fwdLn)
    fwdLayerNorms.push_back(fwdLn)

    // PointWiseFeedForward (Conv1d→ReLU→Conv1d with residuals, mirroring Python).
    val ffn = new PointWiseFeedForward(embedDim, ffnDim, dropout, device)
    fwdLayers.push_back(ffn)
    register_module(s"fwd_layer_$i", ffn)
  }

  // Final LayerNorm (after all blocks).
  private val lastLayerNorm = {
    val normShape = new LongVector(1); normShape.put(0, embedDim.toLong)
    val ln = new LayerNormImpl(normShape)
    register_module("last_layer_norm", ln)
    ln
  }

  // Output projection — used for the binary relevance logit over the pooled sequence.
  private val outputProj = new LinearImpl(embedDim.toLong, 1L)
  register_module("output", outputProj)

  override def forward(sequence: Tensor): Tensor = {
    val seq = sequence.toType(ScalarType.Long)
    val batch = seq.size(0)
    val len = seq.size(1)

    // Item embeddings: [batch, len, embedDim]
    val itemEmb = itemEmbedding.forward(seq)

    // Positional embeddings broadcast over batch.
    val posIds = torch.arange(new Scalar(len),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
    val posEmb = positionEmbedding.forward(posIds).unsqueeze(0L) // [1, len, embedDim]

    // Scale-by-sqrt(embed_dim) trick from the Transformer paper, used by SASRec.
    val scale = math.sqrt(embedDim.toDouble).toFloat
    var hidden = itemEmb.mul(new Scalar(scale)).add(posEmb)
    hidden = applyDropout(hidden)

    // Padding mask: positions equal to padding idx (0) are masked out.
    val padScalar = torch.zeros(Array(1L),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
    val keyMask = seq.ne(padScalar)            // [batch, len] (Bool)
    val keyMaskF = keyMask.toType(ScalarType.Float)

    // Causal mask: positions can only attend to themselves or earlier.
    val causalMask = torch.triu(
      torch.ones(Array(len.toLong, len.toLong),
        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))), 1L
    ).eq(new Scalar(0.0))  // [len, len] True where allowed

    var layer = 0
    while (layer < numLayers) {
      hidden = attentionBlock(hidden, keyMaskF, causalMask, batch, len, layer)
      layer += 1
    }

    // Final LayerNorm, then masked mean-pool over the sequence.
    hidden = lastLayerNorm.forward(hidden)
    val maskExpanded = keyMaskF.unsqueeze(2L)               // [batch, len, 1]
    val summed = hidden.mul(maskExpanded).sum(1L)             // [batch, embedDim]
    val counts = keyMaskF.sum(1L).clamp_min(new Scalar(1.0)).unsqueeze(1L) // [batch, 1]
    val pooled = summed.div(counts)

    outputProj.forward(pooled) // [batch, 1]
  }

  /** One self-attention block: pre-LN → Q/K/V → scaled-dot-product attention with
   *  causal + key masks → residual → pre-LN → PointWiseFeedForward → residual. */
  private def attentionBlock(
    input: Tensor,
    keyMaskF: Tensor,
    causalMask: Tensor,
    batch: Long,
    len: Long,
    layer: Int
  ): Tensor = {
    // Pre-attention LayerNorm (applied to Q, the Python uses the same input for
    // Q, K, V but normalises only Q — matches the original SASRec).
    val qNorm = attnLayerNorms.get(layer).forward(input)      // [batch, len, embedDim]
    val q = attnQProjs.get(layer).forward(qNorm)
    val k = attnKProjs.get(layer).forward(input)
    val v = attnVProjs.get(layer).forward(input)

    // Reshape to [batch, numHeads, len, headDim]
    val qh = q.reshape(batch, len, numHeads.toLong, headDim.toLong).transpose(1L, 2L)
    val kh = k.reshape(batch, len, numHeads.toLong, headDim.toLong).transpose(1L, 2L)
    val vh = v.reshape(batch, len, numHeads.toLong, headDim.toLong).transpose(1L, 2L)

    val scaleAttn = new Scalar((1.0 / math.sqrt(headDim.toDouble)).toFloat)
    val scores = torch.matmul(qh, kh.transpose(-2L, -1L)).mul(scaleAttn) // [B, H, L, L]

    // Combine causal mask [len, len] and key mask [batch, len].
    //   final_mask[b, h, i, j] = causal[i, j] & key[b, j]
    // We build it by broadcasting: causal [1, 1, len, len] AND key [batch, 1, 1, len].
    val causalBroad = causalMask.reshape(1L, 1L, len, len).toType(ScalarType.Float)
    val keyBroad = keyMaskF.reshape(batch, 1L, 1L, len)
    val combined = causalBroad.mul(keyBroad) // float, 1 where allowed, 0 where masked

    val negInf = torch.full(Array(1L), new Scalar(-1e9)).to(input.device(), ScalarType.Float)
    val maskedScores = scores.mul(combined).add(
      combined.mul(new Scalar(-1.0)).add(new Scalar(1.0)).mul(negInf)
    )

    val attn = torch.softmax(maskedScores, -1L)
    val context = torch.matmul(attn, vh) // [batch, numHeads, len, headDim]

    // Merge heads back: [batch, len, embedDim]
    val merged = context.transpose(1L, 2L).contiguous().reshape(batch, len, embedDim.toLong)
    val attnOut = attnOProjs.get(layer).forward(merged)

    // Residual + feed-forward block.
    val res1 = input.add(attnOut)
    val res1Norm = fwdLayerNorms.get(layer).forward(res1)
    val ffOut = fwdLayers.get(layer).forward(res1Norm)
    res1.add(ffOut)
  }
}

/**
 * PointWiseFeedForward — SASRec's per-position feed-forward sublayer.
 *
 * Mirrors the Python reference: two `Conv1d(hidden, hidden, kernel_size=1)` layers
 * sandwich a ReLU and Dropout, applied along the channel dimension. A residual
 * connection is added around the block.
 *
 * Shape
 * -----
 * Input:  (batch, len, hidden)
 * Output: (batch, len, hidden)
 *
 * Note: `Module.to(Device, false)` is intentionally skipped — `Conv1dImpl` in
 * this bytedeco build SIGSEGVs inside `Module.to()` (same family as the bug
 * surfaced in CIN).
 */
class PointWiseFeedForward(
  hiddenDim: Int,
  ffnDim: Int,
  dropout: Float,
  device: String = DeviceSupport.backend
) extends Module {

  private val conv1 = {
    val opt = new Conv1dOptions(hiddenDim.toLong, ffnDim.toLong, new LongPointer(Array(1L): _*))
    new Conv1dImpl(opt)
  }
  register_module("conv1", conv1)

  private val conv2 = {
    val opt = new Conv1dOptions(ffnDim.toLong, hiddenDim.toLong, new LongPointer(Array(1L): _*))
    new Conv1dImpl(opt)
  }
  register_module("conv2", conv2)

  override def forward(x: Tensor): Tensor = {
    // x: (batch, len, hidden) → transpose to (batch, hidden, len) for Conv1d.
    val xt = x.transpose(-1L, -2L)
    val h = conv1.forward(xt)                                 // (batch, ffnDim, len)
    val hDrop = torch.dropout(h.relu(), dropout.toDouble, false)
    val h2 = conv2.forward(hDrop)                             // (batch, hidden, len)
    val h2Drop = torch.dropout(h2, dropout.toDouble, false)
    // Back to (batch, len, hidden), add residual.
    h2Drop.transpose(-1L, -2L).add(x)
  }
}