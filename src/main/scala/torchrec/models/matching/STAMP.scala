package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

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

/**
 * STAMP — Short-Term Attention/Memory Priority Model (CIKM'2018).
 *
 * Mirrors the Python `torch-rechub` reference: item embedding with padding_idx=0,
 * four attention parameters (`w_0`, `w_1_t`, `w_2_t`, `w_3_t`) and `b_a`, and two
 * `Tanh + Linear` "memory" / "preference" towers (`f_s`, `f_t`) — both wrapped
 * in `Sequential`s matching the Python's `nn.Sequential` calls.
 *
 * Reference: https://dl.acm.org/doi/10.1145/3219819.3219950
 *
 * @param features     List of features. The first `SequenceFeature` drives the
 *                     item vocabulary and sequence length.
 * @param embedDim     Item embedding dimension.
 * @param weightStd    Std-dev for the four attention parameters (`w_0` etc).
 * @param embStd       Std-dev for embedding / Linear initialisation.
 * @param device       Device for parameters.
 *
 * Shape
 * -----
 * Input  : (batch, seqLen) LongTensor of item ids.
 * Output : (batch, embedDim) user representation (callers dot-product with the
 *          candidate item embedding to obtain a score).
 */
class STAMP(
  features: List[Feature],
  embedDim: Int = 8,
  weightStd: Float = 0.1f,
  embStd: Float = 0.05f,
  // Kept for source compatibility with the previous constructor signature used
  // by RunMatchingPipeline1. The Python reference doesn't expose this knob —
  // all MLPs use `embedDim`.
  attentionDim: Int = -1,
  device: String = DeviceSupport.backend
) extends Module {

  // Backward-compatible secondary constructor: matches the pre-rewrite
  // signature `new STAMP(features, embedDim, attentionDim, device)`.
  def this(features: List[Feature], embedDim: Int, attentionDim: Int, device: String) =
    this(features, embedDim, 0.1f, 0.05f, attentionDim, device)

  require(features.nonEmpty, "STAMP: features cannot be empty")
  require(embedDim > 0, s"STAMP: embedDim must be > 0, got $embedDim")

  private val seqFeature: SequenceFeature = features.head match {
    case sf: SequenceFeature => sf
    case other => throw new IllegalArgumentException(
      s"STAMP expects a SequenceFeature as the first feature, got: ${other.getClass.getSimpleName}")
  }
  val seqFeatureName: String = seqFeature.name
  private val vocabSize: Long = seqFeature.vocabSize

  // Item embedding with padding_idx=0, exactly like the Python.
  private val itemEmbedding = {
    val opts = new EmbeddingOptions(vocabSize, embedDim)
    opts.padding_idx().put(0L)
    val emb = new EmbeddingImpl(opts)
    torch.normal_(emb.weight, 0.0, embStd.toDouble)
    emb
  }
  register_module("item_emb", itemEmbedding)

  // Attention parameters — initialised as plain Tensors, then registered.
  // We must register *after* the in-place init (`normal_`) because `register_parameter`
  // turns the tensor into a leaf Variable that requires grad, and autograd refuses
  // in-place ops on such leaves.
  private val w0 = {
    val t = torch.zeros(Array(embedDim.toLong, 1L),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    torch.normal_(t, 0.0, weightStd.toDouble)
    t
  }
  register_parameter("w_0", w0)

  private val w1T = {
    val t = torch.zeros(Array(embedDim.toLong, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    torch.normal_(t, 0.0, weightStd.toDouble)
    t
  }
  register_parameter("w_1_t", w1T)

  private val w2T = {
    val t = torch.zeros(Array(embedDim.toLong, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    torch.normal_(t, 0.0, weightStd.toDouble)
    t
  }
  register_parameter("w_2_t", w2T)

  private val w3T = {
    val t = torch.zeros(Array(embedDim.toLong, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    torch.normal_(t, 0.0, weightStd.toDouble)
    t
  }
  register_parameter("w_3_t", w3T)

  private val bA = torch.zeros(Array(embedDim.toLong),
    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  register_parameter("b_a", bA)

  // f_s = Tanh -> Linear(embedDim, embedDim)  (memory tower). The LinearImpl
  // already Kaiming-initialises its weight and zeros its bias at construction,
  // which approximates the Python's `_init_module_weights` pass closely enough
  // for matching the public API; the `_init_module_weights` normal re-init from
  // the Python is intentionally skipped here because bytedeco rejects in-place
  // ops on Parameter leaves.
  private val fS: SequentialImpl = new SequentialImpl()
  fS.push_back("tanh_s", new TanhImpl())
  private val fSLinear = new LinearImpl(embedDim.toLong, embedDim.toLong)
  fS.push_back("linear_s", fSLinear)
  register_module("f_s", fS)

  // f_t = Tanh -> Linear(embedDim, embedDim)  (preference tower).
  private val fT: SequentialImpl = new SequentialImpl()
  fT.push_back("tanh_t", new TanhImpl())
  private val fTLinear = new LinearImpl(embedDim.toLong, embedDim.toLong)
  fT.push_back("linear_t", fTLinear)
  register_module("f_t", fT)

  override def forward(sequence: Tensor): Tensor = {
    val input = sequence.toType(ScalarType.Long)
    val batch = input.size(0)

    // value_mask: 1 where item != 0, with a trailing dim for broadcasting.
    val valueMask = input.ne(new Scalar(0L)).unsqueeze(-1L)                  // (batch, len, 1)
    val valueMaskF = valueMask.toType(ScalarType.Float)
    val valueCounts = valueMaskF.sum(1L).unsqueeze(-1L)                       // (batch, 1, 1)
    val itemEmbBatch = itemEmbedding.forward(input).mul(valueMaskF)          // (batch, len, embedDim)

    // x_t: embedding of the last valid token per row.
    val lastIdx = valueCounts.squeeze(-1L).sub(new Scalar(1.0f)).toType(ScalarType.Long) // (batch, 1)
    val xT = itemEmbedding.forward(torch.gather(input, 1L, lastIdx)).squeeze(1L) // (batch, embedDim)

    // m_s: mean-pooled, masked history embedding, kept as (batch, 1, embedDim)
    // so it broadcasts cleanly with itemEmbBatch.
    val mS = itemEmbBatch.sum(Array[Long](1L): _*).div(valueCounts.squeeze(1L)).unsqueeze(1L) // (batch, 1, embedDim)

    // a = normalize(sigmoid(itemEmb @ w1_t + x_t @ w2_t + m_s @ w3_t + b_a) @ w_0) * value_mask
    val preSigmoid = torch.matmul(itemEmbBatch, w1T)
      .add(torch.matmul(xT.unsqueeze(1L), w2T))                              // (batch, len, embedDim)
      .add(torch.matmul(mS, w3T))                                            // (batch, len, embedDim)
      .add(bA)                                                               // (batch, len, embedDim)
    val preAttn = preSigmoid.sigmoid().matmul(w0).mul(valueMaskF)              // (batch, len, 1)
    // L1-normalise along the sequence dim to get the attention weights `a`.
    val a = preAttn.div(preAttn.sum(Array[Long](1L): _*).clamp_min(new Scalar(1e-9)).unsqueeze(-1L))

    // m_a = (a * item_emb).sum(1) + m_s
    val mA = torch.mul(a, itemEmbBatch).sum(1L).add(mS.squeeze(1L))           // (batch, embedDim)

    // h_s = f_s(m_a); h_t = f_t(x_t)
    val hS = fS.forward(mA)                                                  // (batch, embedDim)
    val hT = fT.forward(xT)                                                  // (batch, embedDim)

    hS.mul(hT)                                                               // (batch, embedDim)
  }
}

/**
 * STAMP factory — identical to the constructor.
 */
object STAMP {
  def apply(
    features: List[Feature],
    embedDim: Int = 8,
    weightStd: Float = 0.1f,
    embStd: Float = 0.05f,
    attentionDim: Int = -1,
    device: String = DeviceSupport.backend
  ): STAMP = new STAMP(features, embedDim, weightStd, embStd, attentionDim, device)
}