package torchrec.models.multi_task

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.basic.features._
import torchrec.basic.layers.{EmbeddingLayer, MLP, PredictionLayer}
import torchrec.utils.DeviceSupport

/**
 * MMOE - Multi-gate Mixture-of-Experts Model
 *
 * Full implementation matching the Python torch_rechub MMOE model.
 * Uses multiple gates for expert selection per task.
 *
 * Reference:
 *   "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts"
 *   KDD'2018 - https://dl.acm.org/doi/pdf/10.1145/3219819.3220007
 *
 * Architecture:
 *   - Experts: Shared expert networks (MLPs) - stored in Map for type-safe access
 *   - Gates: Task-specific gating networks - stored in Map for type-safe access
 *   - Towers: Task-specific output predictors - stored in Map for type-safe access
 *   - Prediction layers for final output - stored in Map for type-safe access
 *
 * @param features List of features
 * @param taskTypes List of task types: "classification" or "regression"
 * @param nExpert Number of expert networks
 * @param expertParams Expert MLP parameters: dims, activation, dropout
 * @param towerParamsList List of tower params dict (one per task)
 * @param device Device for computation
 */
class MMOE(
            features: List[Feature],
            taskTypes: List[String],
            nExpert: Int = 4,
            expertParams: Map[String, Any] = Map(),
            towerParamsList: List[Map[String, Any]] = List(),
            device: String = DeviceSupport.backend
          ) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(taskTypes.nonEmpty, "taskTypes cannot be empty")
  require(nExpert > 0, "nExpert must be positive")

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

  // ====== Store typed references in Map for type-safe Scala access ======
  // Modules are registered via register_module() which adds them to the ModuleListImpl internally

  // Experts - Map for type-safe access
  private val expertsMap: Map[String, MLP] = (0 until nExpert).map { i =>
    val name = s"expert_$i"
    val expert = new MLP(inputDims, expertDims, expertDims.last, expertActivation, expertDropout, outputLayer = false, device = device)
    expert.to(targetDevice, false)
    register_module(name, expert)
    (name, expert)
  }.toMap

  // Gates - Map for type-safe access (one per task)
  private val gatesMap: Map[String, MLP] = (0 until nTask).map { i =>
    val name = s"gate_$i"
    // Gate outputs softmax over n_expert
    val gate = new MLP(inputDims, List(nExpert), nExpert, "softmax", 0.0f, outputLayer = false, device = device)
    gate.to(targetDevice, false)
    register_module(name, gate)
    (name, gate)
  }.toMap

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
    // Embedding: [batch_size, input_dims]
    val embedX = embedding.forward(sparseFeats = x, sequenceFeats = Map(), squeeze = true)

    // Expert outputs: List of [batch_size, expert_dim]
    val expertOuts: Seq[Tensor] = expertsMap.values.toList.map(_.forward(embedX))

    // Concatenate experts along dim=1 then reshape to [batch_size, n_expert, expert_dim]
    val batchSize = embedX.size(0)
    val expertDim = expertDims.last
    // torch.cat along dim=1 gives [batch_size, n_expert * expert_dim]
    val expertOutsCatFlat = torch.cat(new TensorVector(expertOuts: _*), 1)
    // Reshape to [batch_size, n_expert, expert_dim]
    val expertOutsCat = expertOutsCatFlat.view(batchSize, nExpert.toLong, expertDim)

    // Gate outputs and weighted combination for each task
    val outputs = (0 until nTask).map { i =>
      val gate = gatesMap(s"gate_$i")
      val tower = towersMap(s"tower_$i")
      val predictLayer = predictLayersMap(s"predictLayer_$i")

      // Gate: [batch_size, n_expert]
      val gateOut = gate.forward(embedX).unsqueeze(-1) // [batch_size, n_expert, 1]

      // Weighted experts: [batch_size, n_expert, expert_dim]
      val expertWeight = torch.mul(gateOut, expertOutsCat)

      // Pool: [batch_size, expert_dim]
      val expertPooling = expertWeight.sum(1)

      // Tower: [batch_size, 1]
      val towerOut = tower.forward(expertPooling)

      // Prediction: sigmoid for classification
      predictLayer.forward(towerOut)
    }

    // Concatenate: torch.cat(ys, dim=1)
    torch.cat(new TensorVector(outputs: _*), 1)
  }
}

/**
 * MMOE companion object with factory methods.
 */
object MMOE {
  def apply(
             features: List[Feature],
             taskTypes: List[String] = List("classification"),
             nExpert: Int = 4,
             expertParams: Map[String, Any] = Map("dims" -> List(128L), "activation" -> "relu", "dropout" -> 0.0f),
             towerParamsList: List[Map[String, Any]] = List(),
             device: String = DeviceSupport.backend
           ): MMOE = {
    new MMOE(features, taskTypes, nExpert, expertParams, towerParamsList, device)
  }
}