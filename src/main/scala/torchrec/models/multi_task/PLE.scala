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
 * PLE — Progressive Layered Extraction (RecSys'2020).
 *
 * Mirrors the Python reference: a stack of CGC layers, each a `ModuleList` of
 * task-specific / shared experts, plus task-specific gates (and a shared gate
 * for all but the last level). Final stage: per-task towers and prediction
 * layers, all in `ModuleListImpl`s.
 *
 * Reference:
 *   "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model
 *    for Personalized Recommendations"
 *   https://dl.acm.org/doi/abs/10.1145/3383313.3412236
 *
 * @param features         List of feature columns.
 * @param taskTypes        Per-task type: `"classification"` or `"regression"`.
 * @param nLevel           Number of CGC layers.
 * @param nExpertSpecific  Task-specific experts per task.
 * @param nExpertShared    Shared experts.
 * @param expertParams     Expert MLP params.
 * @param towerParamsList  Per-task tower params.
 * @param device           Device for parameters.
 */
class PLE(
  features: List[Feature],
  taskTypes: List[String],
  nLevel: Int = 3,
  nExpertSpecific: Int = 1,
  nExpertShared: Int = 1,
  expertParams: Map[String, Any] = Map(),
  towerParamsList: List[Map[String, Any]] = List(),
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "PLE: features cannot be empty")
  require(taskTypes.nonEmpty, "PLE: taskTypes cannot be empty")
  require(nLevel > 0, s"PLE: nLevel must be > 0, got $nLevel")
  require(nExpertSpecific >= 0, s"PLE: nExpertSpecific must be >= 0, got $nExpertSpecific")
  require(nExpertShared >= 0, s"PLE: nExpertShared must be >= 0, got $nExpertShared")
  require(taskTypes.forall(t => t == "classification" || t == "regression"),
    "PLE: taskTypes must be 'classification' or 'regression'")

  private val nTask: Int = taskTypes.size
  private val inputDims: Int = features.map(_.embedDim).sum
  private val expertDims: List[Long] =
    expertParams.getOrElse("dims", List(128L)).asInstanceOf[List[Long]]
  private val expertLast: Long = expertDims.last

  private val embedding = new EmbeddingLayer(features, features.head.embedDim, device)
  register_module("embedding", embedding)

  // CGC layers — mirrors `nn.ModuleList(CGC(...) for i in range(n_level))`.
  private val cgcLayers: ModuleListImpl = new ModuleListImpl()
  private val cgcRefs = Array.ofDim[CGC](nLevel)
  for (level <- 0 until nLevel) {
    val cgc = new CGC(
      curLevel = level + 1,
      nLevel = nLevel,
      nTask = nTask,
      nExpertSpecific = nExpertSpecific,
      nExpertShared = nExpertShared,
      inputDims = if (level == 0) inputDims else expertLast.toInt,
      expertParams = expertParams,
      device = device
    )
    register_module(s"cgc_$level", cgc)
    cgcLayers.push_back(cgc)
    cgcRefs(level) = cgc
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

    // ple_inputs starts as [embed_x] * (n_task + 1) — one slot per task + shared.
    var pleInputs: List[Tensor] = List.fill(nTask + 1)(embedX)

    var level = 0
    while (level < nLevel) {
      pleInputs = cgcRefs(level).forward(pleInputs)
      level += 1
    }

    val ys = mutable.ArrayBuffer[Tensor]()
    var ti = 0
    while (ti < nTask) {
      val towerOut = towers.get(ti).forward(pleInputs(ti))
      ys += predictLayers.get(ti).forward(towerOut)
      ti += 1
    }

    torch.cat(new TensorVector(ys.toSeq: _*), 1L)
  }
}

/**
 * CGC — Customized Gate Control layer (per level of PLE).
 *
 * Mirrors the Python reference exactly:
 *   - `n_task * n_expert_specific` task-specific experts
 *   - `n_expert_shared` shared experts
 *   - `n_task` task-specific gates (softmax over `n_expert_specific + n_expert_shared`)
 *   - one shared gate (softmax over all experts) at every level except the last
 */
class CGC(
  curLevel: Int,
  nLevel: Int,
  nTask: Int,
  nExpertSpecific: Int,
  nExpertShared: Int,
  inputDims: Int,
  expertParams: Map[String, Any],
  device: String = DeviceSupport.backend
) extends Module {

  private val expertDims: List[Long] =
    expertParams.getOrElse("dims", List(128L)).asInstanceOf[List[Long]]
  private val expertActivation: String =
    expertParams.getOrElse("activation", "relu").asInstanceOf[String]
  private val expertDropout: Float =
    expertParams.getOrElse("dropout", 0.0f).asInstanceOf[Float]
  private val expertLast: Long = expertDims.last

  // Number of experts a task-specific gate sees (own + shared).
  private val nExpertPerTask: Long = (nExpertSpecific + nExpertShared).toLong
  // Number of experts the shared gate sees (all of them).
  private val nExpertAll: Long = (nExpertSpecific * nTask + nExpertShared).toLong

  // Task-specific experts — `n_task * n_expert_specific` of them.
  private val expertsSpecific: ModuleListImpl = new ModuleListImpl()
  for (i <- 0 until nTask * nExpertSpecific) {
    val m = new MLP(inputDims, expertDims, expertLast, expertActivation, expertDropout,
      outputLayer = false, device = device)
    register_module(s"expert_specific_$i", m)
    expertsSpecific.push_back(m)
  }

  // Shared experts.
  private val expertsShared: ModuleListImpl = new ModuleListImpl()
  for (i <- 0 until nExpertShared) {
    val m = new MLP(inputDims, expertDims, expertLast, expertActivation, expertDropout,
      outputLayer = false, device = device)
    register_module(s"expert_shared_$i", m)
    expertsShared.push_back(m)
  }

  // Task-specific gates — one per task, softmax over `n_expert_per_task`.
  private val gatesSpecific: ModuleListImpl = new ModuleListImpl()
  for (i <- 0 until nTask) {
    val m = new MLP(inputDims, List(nExpertPerTask), nExpertPerTask, "softmax", 0.0f,
      outputLayer = false, device = device)
    register_module(s"gate_specific_$i", m)
    gatesSpecific.push_back(m)
  }

  // Shared gate — only at non-final levels.
  private val gateShared: ModuleListImpl = new ModuleListImpl()
  private val hasSharedGate: Boolean = curLevel < nLevel
  if (hasSharedGate) {
    val m = new MLP(inputDims, List(nExpertAll), nExpertAll, "softmax", 0.0f,
      outputLayer = false, device = device)
    register_module("gate_shared", m)
    gateShared.push_back(m)
  }

  def forward(xList: List[Tensor]): List[Tensor] = {
    // xList: List of (batch, inputDims) — one per task + shared at the tail.

    // expert_specific_outs: (batch, 1, expertLast) per expert.
    val expertSpecificOuts = mutable.ArrayBuffer[Tensor]()
    var i = 0
    while (i < nTask * nExpertSpecific) {
      val taskIdx = i / nExpertSpecific
      expertSpecificOuts += expertsSpecific.get(i).forward(xList(taskIdx)).unsqueeze(1)
      i += 1
    }

    // expert_shared_outs: (batch, 1, expertLast) per shared expert.
    val expertSharedOuts = mutable.ArrayBuffer[Tensor]()
    var s = 0
    while (s < nExpertShared) {
      expertSharedOuts += expertsShared.get(s).forward(xList.last).unsqueeze(1)
      s += 1
    }

    // gate_specific_outs: (batch, n_expert_per_task, 1) per task.
    val gateSpecificOuts = mutable.ArrayBuffer[Tensor]()
    var g = 0
    while (g < nTask) {
      gateSpecificOuts += gatesSpecific.get(g).forward(xList(g)).unsqueeze(-1)
      g += 1
    }

    // Per-task output: gate over (own experts + shared experts).
    val cgcOuts = mutable.ListBuffer[Tensor]()
    var ti = 0
    while (ti < nTask) {
      // Slice this task's specific experts out of expertSpecificOuts.
      val taskExpertOuts = expertSpecificOuts.slice(ti * nExpertSpecific, (ti + 1) * nExpertSpecific)
      val allExpertsForTask = taskExpertOuts ++ expertSharedOuts
      // (batch, n_expert_per_task, expertLast)
      val expertConcat = torch.cat(new TensorVector(allExpertsForTask.toSeq: _*), 1L)
      // (batch, n_expert_per_task, expertLast)
      val expertWeight = torch.mul(gateSpecificOuts(ti), expertConcat)
      // (batch, expertLast)
      cgcOuts += expertWeight.sum(1L)
      ti += 1
    }

    // Append shared-gate output at non-final levels — length becomes n_task + 1.
    if (hasSharedGate) {
      val allExpertOuts = expertSpecificOuts ++ expertSharedOuts
      val expertConcatShared = torch.cat(new TensorVector(allExpertOuts.toSeq: _*), 1L)
      val gateSharedOut = gateShared.get(0).forward(xList.last).unsqueeze(-1)
      val expertWeightShared = torch.mul(gateSharedOut, expertConcatShared)
      cgcOuts += expertWeightShared.sum(1L)
    }

    cgcOuts.toList
  }
}

/**
 * PLE factory — identical to the constructor.
 */
object PLE {
  def apply(
    features: List[Feature],
    taskTypes: List[String] = List("classification"),
    nLevel: Int = 3,
    nExpertSpecific: Int = 1,
    nExpertShared: Int = 1,
    expertParams: Map[String, Any] = Map("dims" -> List(128L), "activation" -> "relu", "dropout" -> 0.0f),
    towerParamsList: List[Map[String, Any]] = List(),
    device: String = DeviceSupport.backend
  ): PLE = new PLE(features, taskTypes, nLevel, nExpertSpecific, nExpertShared, expertParams, towerParamsList, device)
}