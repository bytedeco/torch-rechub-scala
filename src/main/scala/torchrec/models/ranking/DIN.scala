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

/**
 * Deep Interest Network (Alibaba, KDD'2018).
 *
 * Mirrors the Python `torch-rechub` reference: one `ActivationUnit` per
 * history feature, stored in a `nn.ModuleList`/`ModuleListImpl`, and a top
 * MLP that consumes the flattened attention-pooled history, flattened target
 * features and flattened context features.
 *
 * Reference: https://arxiv.org/abs/1706.06978
 *
 * @param features         Context / user-profile features.
 * @param sequenceFeatures Per-field history sequence features.
 * @param mlpDims          Hidden dims for the top MLP.
 * @param dropout          Dropout for the top MLP.
 * @param attentionUnits   Hidden dim of each `ActivationUnit`'s MLP (default 36,
 *                         matching the Python's `dims = [36]`).
 * @param device           Device for parameters.
 */
class DIN(
  features: List[Feature],
  sequenceFeatures: List[SequenceFeature],
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  attentionUnits: Int = 36,
  device: String = DeviceSupport.backend
) extends Module {

  /** Backward-compatible secondary constructor: pre-rewrite signature
   *  `new DIN(features, sequenceFeatures, embedDim, mlpDims, dropout, attentionUnits, device)`. */
  def this(
    features: List[Feature],
    sequenceFeatures: List[SequenceFeature],
    embedDim: Int,
    mlpDims: List[Long],
    dropout: Float,
    attentionUnits: Int,
    device: String
  ) = this(features, sequenceFeatures, mlpDims, dropout, attentionUnits, device)

  require(features.nonEmpty, "DIN: features cannot be empty")
  require(sequenceFeatures.nonEmpty, "DIN: sequenceFeatures cannot be empty")

  private val contextDim: Int = features.map(_.embedDim).sum
  private val historyDim: Int = sequenceFeatures.map(_.embedDim).sum
  private val targetDim: Int = historyDim
  private val allDims: Int = contextDim + historyDim + targetDim

  private val embedding = new EmbeddingLayer(
    features ++ sequenceFeatures ++ sequenceFeatures /* targets share with history */,
    device = device)
  register_module("embedding", embedding)

  // One ActivationUnit per history feature — mirrors
  // `nn.ModuleList([ActivationUnit(fea.embed_dim, **attention_mlp_params) for ...])`.
  private val attentionLayers: ModuleListImpl = new ModuleListImpl()
  for (sf <- sequenceFeatures) {
    val unit = new ActivationUnit(sf.embedDim, attentionUnits, device)
    register_module(s"attentionUnit_${sf.name}", unit)
    attentionLayers.push_back(unit)
  }

  // Top MLP over the concatenated [attention-pooled, target, context] vector.
  private val mlp = new MLP(allDims.toLong, mlpDims.map(_.toLong), 1L, "dice", dropout, device = device)
  register_module("mlp", mlp)

  /** Primary forward — per-field target feature map (matches the Python API). */
  def forward(
    sparseFeats: Map[String, Tensor],
    sequenceFeats: Map[String, Tensor],
    targetFeats: Map[String, Tensor]
  ): Tensor = {
    val (pooled, flatTarget) = computeAttentions(sparseFeats, sequenceFeats, targetFeats)
    val embedCtx = embedding.forward(sparseFeats = sparseFeats, sequenceFeats = Map.empty, squeeze = true)
    val mlpIn = torch.cat(new TensorVector(
      pooled.flatten(1L, 2L), flatTarget, embedCtx), 1L)
    mlp.forward(mlpIn).squeeze(1L)
  }

  /** Backward-compat overload — single target index broadcast over every
   *  history field. Used by CTRTrainer. */
  def forward(
    sparseFeats: Map[String, Tensor],
    sequenceFeats: Map[String, Tensor],
    targetIdx: Tensor
  ): Tensor = {
    val targetFeats = sequenceFeatures.map(sf => sf.name -> targetIdx).toMap
    forward(sparseFeats, sequenceFeats, targetFeats)
  }

  /** Shared computation: returns (attentionPooled: B*H*D, flatTarget: B*targetDim). */
  private def computeAttentions(
    sparseFeats: Map[String, Tensor],
    sequenceFeats: Map[String, Tensor],
    targetFeats: Map[String, Tensor]
  ): (Tensor, Tensor) = {
    // History per field, each (batch, seqLen, embedDim).
    val historyByName = sequenceFeatures.map { sf =>
      sf.name -> embedding.forward(sparseFeats = Map.empty,
        sequenceFeats = Map(sf.name -> sequenceFeats(sf.name)),
        squeeze = false).select(1L, 0L)
    }.toMap

    // Target per field, each (batch, embedDim).
    val targetByName = sequenceFeatures.map { sf =>
      sf.name -> embedding.forward(sparseFeats = Map.empty,
        sequenceFeats = Map(sf.name -> targetFeats(sf.name)),
        squeeze = false).select(1L, 0L).squeeze(1L)
    }.toMap

    // attention_pooling list → (batch, num_history_fields, embedDim).
    val pooled = scala.collection.mutable.ArrayBuffer[Tensor]()
    var i = 0
    while (i < sequenceFeatures.size) {
      val sf = sequenceFeatures(i)
      val h = historyByName(sf.name)
      val t = targetByName(sf.name)
      val att = attentionLayers.get(i).forward(h, t)        // (batch, embedDim)
      pooled += att.unsqueeze(1L)
      i += 1
    }
    val attentionPooled = torch.cat(new TensorVector(pooled.toSeq: _*), 1L) // (batch, H, embedDim)

    val flatTarget = torch.cat(new TensorVector(
      sequenceFeatures.map(sf => targetByName(sf.name).unsqueeze(1L)): _*), 1L).view(-1L, targetDim.toLong)
    (attentionPooled, flatTarget)
  }
}

/**
 * Activation Unit — DIN's per-position target attention sublayer.
 *
 * Mirrors the Python reference: `MLP(4 * emb_dim, dims=[36], activation="dice")`
 * applied to `[target, history, target - history, target * history]`.
 *
 * Shape
 * -----
 * Input  history: (batch, seqLen, embedDim)
 * Input  target : (batch, embedDim)
 * Output        : (batch, embedDim)  — softmax-weighted sum over the sequence.
 */
class ActivationUnit(
  embedDim: Int,
  hiddenUnits: Int = 36,
  device: String = DeviceSupport.backend
) extends Module {

  private val attention = new MLP((4 * embedDim).toLong, List(hiddenUnits.toLong), 1L, "dice", 0.0f,
    device = device)
  register_module("attention", attention)

  override def forward(history: Tensor, target: Tensor): Tensor = {
    val seqLen = history.size(1)
    // Expand target across the sequence: (batch, seqLen, embedDim).
    val targetExp = target.unsqueeze(1L).expand(-1L, seqLen, -1L)
    val attInput = torch.cat(new TensorVector(targetExp, history,
      targetExp.sub(history), targetExp.mul(history)), 2L) // (batch, seqLen, 4*embedDim)
    val flat = attInput.view(-1L, (4 * embedDim).toLong)
    val scores = attention.forward(flat).view(-1L, seqLen) // (batch, seqLen)
    val weights = scores.softmax(1L).unsqueeze(-1L)        // (batch, seqLen, 1)
    weights.mul(history).sum(1L)                            // (batch, embedDim)
  }
}