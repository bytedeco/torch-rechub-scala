package torchrec.models.multi_task

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

import torchrec.basic.features._
import torchrec.basic.layers.{EmbeddingLayer, MLP, PredictionLayer}
import torchrec.utils.DeviceSupport

import scala.collection.mutable

/**
 * MMOE — Multi-gate Mixture-of-Experts (KDD'2018).
 *
 * Mirrors the Python reference: `nn.ModuleList`s of experts / gates / towers /
 * prediction layers, with a per-task softmax gate mixing the shared experts.
 *
 * Reference:
 *   "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts"
 *   https://dl.acm.org/doi/pdf/10.1145/3219819.3220007
 *
 * @param features        List of feature columns.
 * @param taskTypes       Per-task type: `"classification"` or `"regression"`.
 * @param nExpert         Number of shared experts.
 * @param expertParams    Expert MLP params: `dims`, `activation`, `dropout`.
 * @param towerParamsList Per-task tower params (same keys as expertParams).
 * @param device          Device for parameters.
 */
class MMOE(
  features: List[Feature],
  taskTypes: List[String],
  nExpert: Int = 4,
  expertParams: Map[String, Any] = Map(),
  towerParamsList: List[Map[String, Any]] = List(),
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "MMOE: features cannot be empty")
  require(taskTypes.nonEmpty, "MMOE: taskTypes cannot be empty")
  require(nExpert > 0, s"MMOE: nExpert must be > 0, got $nExpert")
  require(taskTypes.forall(t => t == "classification" || t == "regression"),
    "MMOE: taskTypes must be 'classification' or 'regression'")

  private val nTask: Int = taskTypes.size
  private val inputDims: Int = features.map(_.embedDim).sum
  private val expertDims: List[Long] =
    expertParams.getOrElse("dims", List(128L)).asInstanceOf[List[Long]]
  private val expertActivation: String =
    expertParams.getOrElse("activation", "relu").asInstanceOf[String]
  private val expertDropout: Float =
    expertParams.getOrElse("dropout", 0.0f).asInstanceOf[Float]
  private val expertLast: Long = expertDims.last

  private val embedding = new EmbeddingLayer(features, features.head.embedDim, device)
  register_module("embedding", embedding)

  // Experts — mirrors `nn.ModuleList(MLP(...) for i in range(n_expert))`.
  private val experts: ModuleListImpl = new ModuleListImpl()
  for (i <- 0 until nExpert) {
    val m = new MLP(inputDims, expertDims, expertLast, expertActivation, expertDropout,
      outputLayer = false, device = device)
    register_module(s"expert_$i", m)
    experts.push_back(m)
  }

  // Gates — one per task; each outputs a softmax over the experts.
  private val gates: ModuleListImpl = new ModuleListImpl()
  for (i <- 0 until nTask) {
    val m = new MLP(inputDims, List(nExpert.toLong), nExpert.toLong, "softmax", 0.0f,
      outputLayer = false, device = device)
    register_module(s"gate_$i", m)
    gates.push_back(m)
  }

  // Towers — one per task.
  private val towers: ModuleListImpl = new ModuleListImpl()
  for (i <- 0 until nTask) {
    val params = if (towerParamsList.isEmpty) Map[String, Any]() else towerParamsList(i)
    val dims = params.getOrElse("dims", List(expertLast)).asInstanceOf[List[Long]]
    val activation = params.getOrElse("activation", "relu").asInstanceOf[String]
    val dropout = params.getOrElse("dropout", 0.0f).asInstanceOf[Float]
    val m = new MLP(expertLast, dims, 1L, activation, dropout, outputLayer = true, device = device)
    register_module(s"tower_$i", m)
    towers.push_back(m)
  }

  // Prediction layers — one per task.
  private val predictLayers: ModuleListImpl = new ModuleListImpl()
  for (i <- 0 until nTask) {
    val m = new PredictionLayer(taskTypes(i))
    register_module(s"predictLayer_$i", m)
    predictLayers.push_back(m)
  }

  def forward(x: Map[String, Tensor]): Tensor = {
    val embedX = embedding.forward(sparseFeats = x, sequenceFeats = Map.empty, squeeze = true)
    val batchSize = embedX.size(0)

    // expert_outs[i]: (batch, 1, expertLast)  → cat along dim=1 → (batch, n_expert, expertLast).
    val expertOuts = mutable.ArrayBuffer[Tensor]()
    var ei = 0
    while (ei < nExpert) {
      expertOuts += experts.get(ei).forward(embedX).unsqueeze(1)
      ei += 1
    }
    val expertCat = torch.cat(new TensorVector(expertOuts.toSeq: _*), 1L)

    // For each task: gate → weighted expert pool → tower → prediction.
    val ys = mutable.ArrayBuffer[Tensor]()
    var ti = 0
    while (ti < nTask) {
      // gate_out: (batch, n_expert, 1)
      val gateOut = gates.get(ti).forward(embedX).unsqueeze(-1)
      // weighted experts: (batch, n_expert, expertLast)
      val expertWeight = torch.mul(gateOut, expertCat)
      // pooled: (batch, expertLast)
      val pooled = expertWeight.sum(1L)
      // tower: (batch, 1) → prediction: (batch, 1) or (batch, 1) post-sigmoid.
      val towerOut = towers.get(ti).forward(pooled)
      ys += predictLayers.get(ti).forward(towerOut)
      ti += 1
    }

    torch.cat(new TensorVector(ys.toSeq: _*), 1L)
  }
}

/**
 * MMOE factory — identical to the constructor.
 */
object MMOE {
  def apply(
    features: List[Feature],
    taskTypes: List[String] = List("classification"),
    nExpert: Int = 4,
    expertParams: Map[String, Any] = Map("dims" -> List(128L), "activation" -> "relu", "dropout" -> 0.0f),
    towerParamsList: List[Map[String, Any]] = List(),
    device: String = DeviceSupport.backend
  ): MMOE = new MMOE(features, taskTypes, nExpert, expertParams, towerParamsList, device)
}