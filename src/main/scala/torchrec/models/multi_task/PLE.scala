package torchrec.models.multi_task

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.basic.features._
import torchrec.basic.layers.{EmbeddingLayer, MLP, PredictionLayer}
import torchrec.utils.DeviceSupport

import scala.collection.mutable

/**
 * PLE - Progressive Layered Extraction Multi-Task Learning Model
 *
 * Full implementation matching the Python torch_rechub PLE model.
 * Uses multi-level CGC (Customized Gate Control) layers for progressive
 * extraction of task-specific and shared representations.
 *
 * Reference:
 *   "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model
 *   for Personalized Recommendations"
 *   RecSys'2020 - https://dl.acm.org/doi/abs/10.1145/3383313.3412236
 *
 * Architecture:
 *   - CGC Layers: Customized Gate Control for expert selection - stored in List for type-safe access
 *   - Task-specific Experts: Task-specific feature extractors (inside CGC)
 *   - Shared Experts: Common feature extractors shared across tasks (inside CGC)
 *   - Task-specific Towers and Prediction layers - stored in Map for type-safe access
 *
 * @param features List of features
 * @param taskTypes List of task types: "classification" or "regression"
 * @param nLevel Number of CGC layers
 * @param nExpertSpecific Number of task-specific experts per task
 * @param nExpertShared Number of shared experts
 * @param expertParams Expert MLP parameters: dims, activation, dropout
 * @param towerParamsList List of tower params dict (one per task)
 * @param device Device for computation
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

  require(features.nonEmpty, "features cannot be empty")
  require(taskTypes.nonEmpty, "taskTypes cannot be empty")
  require(nLevel > 0, "nLevel must be positive")
  require(nExpertSpecific >= 0, "nExpertSpecific must be non-negative")
  require(nExpertShared >= 0, "nExpertShared must be non-negative")

  private val targetDevice = new Device(device)
  private val nTask = taskTypes.size

  // Feature dimensions
  private val inputDims: Int = features.map(_.embedDim).sum

  // Embedding layer
  private val embedding = new EmbeddingLayer(features, features.head.embedDim, device)
  register_module("embedding", embedding)

  // Expert params
  private val expertDims = expertParams.getOrElse("dims", List(128L)).asInstanceOf[List[Long]]
  private val expertActivation = expertParams.getOrElse("activation", "relu").asInstanceOf[String]
  private val expertDropout = expertParams.getOrElse("dropout", 0.0f).asInstanceOf[Float]

  // ====== Store typed references in List/Map for type-safe Scala access ======
  // CGC Layers - List for type-safe access
  private val cgcLayers: List[CGC] = (0 until nLevel).map { level =>
    val cgc = new CGC(
      curLevel = level + 1,
      nLevel = nLevel,
      nTask = nTask,
      nExpertSpecific = nExpertSpecific,
      nExpertShared = nExpertShared,
      inputDims = if (level == 0) inputDims else expertDims.last.toInt,
      expertParams = expertParams,
      device = device
    )
    register_module(s"cgc_$level", cgc)
    cgc
  }.toList

  // Towers - Map for type-safe access
  private val towersMap: Map[String, MLP] = (0 until nTask).map { i =>
    val name = s"tower_$i"
    val params = if (towerParamsList.isEmpty) Map[String, Any]() else towerParamsList(i)
    val dims = params.getOrElse("dims", List(expertDims.last)).asInstanceOf[List[Long]]
    val activation = params.getOrElse("activation", "relu").asInstanceOf[String]
    val dropout = params.getOrElse("dropout", 0.0f).asInstanceOf[Float]
    val tower = new MLP(expertDims.last.toInt, dims, 1, activation, dropout, outputLayer = true, device = device)
    register_module(name, tower)
    (name, tower)
  }.toMap

  // Prediction layers - Map for type-safe access
  private val predictLayersMap: Map[String, PredictionLayer] = (0 until nTask).map { i =>
    val name = s"predictLayer_$i"
    val predictLayer = new PredictionLayer(taskTypes(i))
    register_module(name, predictLayer)
    (name, predictLayer)
  }.toMap

  def forward(x: Map[String, Tensor]): Tensor = {
    // Embedding
    val embedX = embedding.forward(sparseFeats = x, sequenceFeats = Map(), squeeze = true)

    // Initialize PLE inputs: [embed_x] * (n_task + 1) - one for each task + shared
    var pleInputs: List[Tensor] = List.fill(nTask + 1)(embedX)

    // Progressive CGC layers
    for (level <- 0 until nLevel) {
      val pleOuts = cgcLayers(level).forward(pleInputs)
      pleInputs = pleOuts
    }

    // Task predictions
    val outputs = (0 until nTask).map { i =>
      val tower = towersMap(s"tower_$i")
      val predictLayer = predictLayersMap(s"predictLayer_$i")
      val towerOut = tower.forward(pleInputs(i))
      predictLayer.forward(towerOut)
    }

    // Concatenate: torch.cat(ys, dim=1)
    torch.cat(new TensorVector(outputs: _*), 1)
  }
}

/**
 * CGC - Customized Gate Control Layer for PLE
 *
 * Implements the gating mechanism for expert selection in PLE.
 *
 * @param curLevel Current level of CGC in PLE
 * @param nLevel Total number of CGC levels
 * @param nTask Number of tasks
 * @param nExpertSpecific Number of task-specific experts per task
 * @param nExpertShared Number of shared experts
 * @param inputDims Input dimension for experts
 * @param expertParams Expert MLP parameters
 * @param device Device for computation
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

  private val targetDevice = new Device(device)

  // Total experts = nExpertSpecific per task * nTask + nExpertShared
  // This is the dimension for gates (each task gates access to all experts)
  private val nExpertAll = nExpertSpecific * nTask + nExpertShared

  // Expert dims
  private val expertDims = expertParams.getOrElse("dims", List(128L)).asInstanceOf[List[Long]]
  private val expertActivation = expertParams.getOrElse("activation", "relu").asInstanceOf[String]
  private val expertDropout = expertParams.getOrElse("dropout", 0.0f).asInstanceOf[Float]

  // ====== Store typed references in List for type-safe access ======
  // Task-specific Experts - List for type-safe access
  private val expertsSpecific: List[MLP] = (0 until nTask * nExpertSpecific).map { i =>
    val expert = new MLP(inputDims, expertDims, expertDims.last, expertActivation, expertDropout, outputLayer = false, device = device)
    expert.to(targetDevice, false)
    register_module(s"expert_specific_$i", expert)
    expert
  }.toList

  // Shared Experts - List for type-safe access
  private val expertsShared: List[MLP] = (0 until nExpertShared).map { i =>
    val expert = new MLP(inputDims, expertDims, expertDims.last, expertActivation, expertDropout, outputLayer = false, device = device)
    expert.to(targetDevice, false)
    register_module(s"expert_shared_$i", expert)
    expert
  }.toList

  // Task-specific Gates - List for type-safe access
  // Each task-specific gate controls: nExpertSpecific (private) + nExpertShared (shared) = nExpertPerTask experts
  private val nExpertPerTask = nExpertSpecific + nExpertShared
  private val gatesSpecific: List[MLP] = (0 until nTask).map { i =>
    val gate = new MLP(inputDims, List(nExpertPerTask), nExpertPerTask, "softmax", 0.0f, outputLayer = false, device = device)
    gate.to(targetDevice, false)
    register_module(s"gate_specific_$i", gate)
    gate
  }.toList

  // Shared Gate (only if not last level)
  // Shared gate controls ALL experts: nExpertSpecific * nTask + nExpertShared
  private val gateSharedOption: Option[MLP] = if (curLevel < nLevel) {
    val gate = new MLP(inputDims, List(nExpertAll), nExpertAll, "softmax", 0.0f, outputLayer = false, device = device)
    gate.to(targetDevice, false)
    register_module("gate_shared", gate)
    Some(gate)
  } else None

  def forward(xList: List[Tensor]): List[Tensor] = {
    // xList: List of [batch_size, input_dims] - one per task + shared

    // Expert specific outputs for each task
    val expertSpecificOuts: mutable.ListBuffer[Tensor] = mutable.ListBuffer()
    for (taskIdx <- 0 until nTask) {
      for (expertIdxLocal <- 0 until nExpertSpecific) {
        val expertIdx = taskIdx * nExpertSpecific + expertIdxLocal
        val expert = expertsSpecific(expertIdx)
        expertSpecificOuts += expert.forward(xList(taskIdx)).unsqueeze(1)
      }
    }

    // Expert shared outputs
    val expertSharedOuts: Seq[Tensor] = expertsShared.map(_.forward(xList.last).unsqueeze(1))

    // Gate specific outputs
    val gateSpecificOuts: Seq[Tensor] = gatesSpecific.zipWithIndex.map { case (gate, taskIdx) =>
      gate.forward(xList(taskIdx)).unsqueeze(-1)
    }

    // CGC outputs for each task
    val cgcOuts: mutable.ListBuffer[Tensor] = mutable.ListBuffer()

    for (taskIdx <- 0 until nTask) {
      // Get expert outputs for this task
      val taskExpertOuts = expertSpecificOuts.slice(taskIdx * nExpertSpecific, (taskIdx + 1) * nExpertSpecific)

      // Combine with shared experts
      val allExpertsForTask = taskExpertOuts ++ expertSharedOuts

      // Concatenate experts: [batch_size, n_expert_specific + n_expert_shared, expert_dim]
      val expertConcat = torch.cat(new TensorVector(allExpertsForTask.toSeq*), 1)

      // Weighted sum using gate
      val gateOut = gateSpecificOuts(taskIdx) // [batch_size, n_expert_all, 1]
      val expertWeight = torch.mul(gateOut, expertConcat) // [batch_size, n_expert_all, expert_dim]

      // Sum: [batch_size, expert_dim]
      val expertPooling = expertWeight.sum(1)
      cgcOuts += expertPooling
    }

    // Add shared gate output if not last level
    if (gateSharedOption.isDefined) {
      val allExpertOuts = expertSpecificOuts.toList ++ expertSharedOuts.toList
      val expertConcatShared = torch.cat(new TensorVector(allExpertOuts: _*), 1)
      val gateSharedOut = gateSharedOption.get.forward(xList.last).unsqueeze(-1)
      val expertWeightShared = torch.mul(gateSharedOut, expertConcatShared)
      val expertPoolingShared = expertWeightShared.sum(1)
      cgcOuts += expertPoolingShared
    }

    cgcOuts.toList
  }
}

/**
 * PLE companion object with factory methods.
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
           ): PLE = {
    new PLE(features, taskTypes, nLevel, nExpertSpecific, nExpertShared, expertParams, towerParamsList, device)
  }
}