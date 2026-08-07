package torchrec.models.ranking

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

import torchrec.Implicits._
import torchrec.Implicits.SeqTensorRichSeq

import scala.collection.mutable

/**
 * Deep Interest Evolution Network (DIEN, AAAI'2019).
 *
 * Mirrors the Python `torch-rechub` reference: per-history-field stacks of
 * `nn.GRU` interest extractors and `AUGRU` interest-evolving cells, both
 * stored in `nn.ModuleList`s — implemented here as `ModuleListImpl`s.
 *
 * Reference: https://arxiv.org/pdf/1809.03672
 *
 * @param features         Context / user-profile features.
 * @param sequenceFeatures History sequence features (one `GRU` + one `AUGRU` each).
 * @param embedDim         Item-embedding dimension.
 * @param mlpDims          Hidden dims for the top MLP.
 * @param dropout          Dropout for the top MLP.
 * @param device           Device for parameters.
 */
class DIEN(
  features: List[Feature],
  sequenceFeatures: List[SequenceFeature],
  embedDim: Int = 8,
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  /** Backward-compatible secondary constructor: pre-rewrite signature
   *  `new DIEN(features, sequenceFeatures, embedDim, hiddenDim, mlpDims, dropout, device)`.
   *  The pre-rewrite `hiddenDim` was the same as `embedDim` for GRU hidden size;
   *  modern DIEN derives it from each feature's `embedDim`, so we just drop it. */
  def this(
    features: List[Feature],
    sequenceFeatures: List[SequenceFeature],
    embedDim: Int,
    hiddenDim: Int,
    mlpDims: List[Long],
    dropout: Float,
    device: String
  ) = this(features, sequenceFeatures, embedDim, mlpDims, dropout, device)

  require(features.nonEmpty, "DIEN: features cannot be empty")
  require(sequenceFeatures.nonEmpty, "DIEN: sequenceFeatures cannot be empty")

  private val sparseDim: Long = Features.calcSparseDim(features)
  private val historyDim: Long = sequenceFeatures.map(_.embedDim).sum
  private val totalDim: Long = sparseDim + historyDim

  private val embedding = new EmbeddingLayer(features ++ sequenceFeatures, embedDim, device)
  register_module("embedding", embedding)

  // Mirrors `interest_extractor_layers = nn.ModuleList([nn.GRU(...) for ...])`.
  private val interestExtractorLayers: ModuleListImpl = new ModuleListImpl()
  for (i <- sequenceFeatures.indices) {
    val fea = sequenceFeatures(i)
    val opts = new GRUOptions(fea.embedDim, fea.embedDim)
    opts.batch_first().put(true)
    val gru = new GRUImpl(opts)
    register_module(s"interest_extractor_$i", gru)
    interestExtractorLayers.push_back(gru)
  }

  // Mirrors `interest_evolving_layers = nn.ModuleList([AUGRU(...) for ...])`.
  private val interestEvolvingLayers: ModuleListImpl = new ModuleListImpl()
  for (i <- sequenceFeatures.indices) {
    val augru = new AUGRU(sequenceFeatures(i).embedDim, device)
    register_module(s"interest_evolving_$i", augru)
    interestEvolvingLayers.push_back(augru)
  }

  // Top MLP — activation fixed to "dice" as in the paper.
  private val mlp = new MLP(totalDim, mlpDims.map(_.toLong), 1L, "dice", dropout, device = device)
  register_module("mlp", mlp)

  def forward(
    sparseFeats: Map[String, Tensor],
    seqFeats: Map[String, Tensor]
  ): Tensor = {
    val featEmb = embedding.forward(sparseFeats, squeeze = true)
    val seqEmb = embedding.forwardSeqRaw(seqFeats)

    val feaDims = sequenceFeatures.map(_.embedDim)
    val interestOut = mutable.ListBuffer[Tensor]()
    var offset = 0
    for (i <- sequenceFeatures.indices) {
      val dim = feaDims(i)
      val seq = seqEmb.narrow(2, offset.toLong, dim.toLong)
      offset += dim
      val mask = getMask(seqFeats, sequenceFeatures(i))

      // Interest Extractor: GRU. The bytedeco `forwardT_TensorTensor_T` returns
      // a (output, hidden) pair — `.get1()` is the per-step output sequence.
      val gruOut = interestExtractorLayers.get(i).forwardT_TensorTensor_T(seq).get1()

      val seqLen = seq.size(1)
      val targetEmb = if (gruOut.dim() == 3L) gruOut.select(1, seqLen - 1) else gruOut

      // Interest Evolving: AUGRU. Returns (allSteps, lastHidden).
      val augru = interestEvolvingLayers.get(i).asInstanceOf[AUGRU]
      val augruOut = augru.run(gruOut, targetEmb, mask)
      interestOut += augruOut._2.unsqueeze(1L)
    }

    val historyOut = torch.cat(new TensorVector(interestOut.toSeq: _*), 1L)
    val mlpIn = torch.cat(new TensorVector(historyOut.view(-1L, historyDim), featEmb), 1L)
    mlp.forward(mlpIn).squeeze(1L)
  }

  private def getMask(seqFeats: Map[String, Tensor], sf: SequenceFeature): Tensor = {
    val raw = seqFeats(sf.name)
    raw.ne(new Scalar(0L)).toType(ScalarType.Float).squeeze(-1L)
  }
}

/**
 * AUGRU cell + wrapper, mirroring the Python's `AUGRU` / `AUGRU_Cell`.
 *
 * `forward(history, target, mask)` returns `(allSteps, lastHidden)`.
 */
class AUGRU(
  embedDim: Int,
  device: String = DeviceSupport.backend
) extends Module {

  val cell: AUGRUCell = new AUGRUCell(embedDim)
  register_module("cell", cell)

  def run(history: Tensor, target: Tensor, mask: Tensor): (Tensor, Tensor) = {
    val scores = history.mul(target.unsqueeze(1L)).sum(-1L)  // simplified attention
    val m = mask.gt(new Scalar(0L)).toType(ScalarType.Bool)
    val masked = scores.masked_fill(m.logical_not(), new Scalar(Double.NegativeInfinity))
    val attn = masked.softmax(1L).unsqueeze(-1L)

    val batch = history.size(0)
    val time = history.size(1)
    var h = torch.zeros(Array(batch, embedDim),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    val outs = mutable.ArrayBuffer[Tensor]()
    var t = 0
    while (t < time) {
      h = cell.forward(history.select(1L, t.toLong), h, attn.select(1L, t.toLong))
      outs += h.unsqueeze(1L)
      t += 1
    }
    (torch.cat(new TensorVector(outs.toSeq: _*), 1L), h)
  }
}

/**
 * AUGRU gate cell. Mirrors the Python's `AUGRU_Cell` with xavier-uniform
 * initialised parameters.
 */
class AUGRUCell(embedDim: Int) extends Module {

  val Wu: Tensor = {
    val t = torch.zeros(Array(embedDim.toLong, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    torch.xavier_uniform_(t)
    t
  }
  register_parameter("Wu", Wu)
  val Uu: Tensor = initXavier(Array(embedDim.toLong, embedDim.toLong)); register_parameter("Uu", Uu)
  val bu: Tensor = initXavier(Array(1L, embedDim.toLong)); register_parameter("bu", bu)
  val Wr: Tensor = initXavier(Array(embedDim.toLong, embedDim.toLong)); register_parameter("Wr", Wr)
  val Ur: Tensor = initXavier(Array(embedDim.toLong, embedDim.toLong)); register_parameter("Ur", Ur)
  val br: Tensor = initXavier(Array(1L, embedDim.toLong)); register_parameter("br", br)
  val Wh: Tensor = initXavier(Array(embedDim.toLong, embedDim.toLong)); register_parameter("Wh", Wh)
  val Uh: Tensor = initXavier(Array(embedDim.toLong, embedDim.toLong)); register_parameter("Uh", Uh)
  val bh: Tensor = initXavier(Array(1L, embedDim.toLong)); register_parameter("bh", bh)

  private def initXavier(shape: Array[Long]): Tensor = {
    val t = torch.zeros(shape, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    torch.xavier_uniform_(t)
    t
  }

  override def forward(x: Tensor, h1: Tensor, a: Tensor): Tensor = {
    val u = torch.sigmoid(x.matmul(Wu).add(h1.matmul(Uu)).add(bu))
    val r = torch.sigmoid(x.matmul(Wr).add(h1.matmul(Ur)).add(br))
    val hHat = torch.tanh(x.matmul(Wh).add(r.mul(h1.matmul(Uh))).add(bh))
    val uHat = a.mul(u)
    (uHat.neg().add(new Scalar(1.0f))).mul(h1).add(uHat.mul(hHat))
  }
}