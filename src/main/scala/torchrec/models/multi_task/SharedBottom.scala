package torchrec.models.multi_task

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.basic.features._
import torchrec.basic.layers.{EmbeddingLayer, MLP, PredictionLayer}
import torchrec.utils.DeviceSupport

/**
 * SharedBottom - Shared Bottom Multi-Task Learning Model
 *
 * Full implementation matching the Python torch_rechub SharedBottom model.
 * Uses a shared bottom network with task-specific towers.
 *
 * Reference:
 *   Caruana, R. (1997). Multitask learning. Machine learning, 28(1), 41-75.
 *
 * Architecture:
 *   - Shared Bottom MLP: Common feature extractor for all tasks
 *   - Task-specific Towers: Task-specific output predictors - stored in Map for type-safe access
 *   - Prediction Layers: Apply sigmoid for classification, pass-through for regression - stored in Map for type-safe access
 *
 * @param features List of features for bottom
 * @param taskTypes List of task types: "classification" or "regression"
 * @param bottomParams MLP parameters for bottom: dims, activation, dropout
 * @param towerParamsList List of tower params dict (one per task)
 * @param device Device for computation
 */
class SharedBottom(
                    features: List[Feature],
                    taskTypes: List[String],
                    bottomParams: Map[String, Any] = Map(),
                    towerParamsList: List[Map[String, Any]] = List(),
                    device: String = DeviceSupport.backend
                  ) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(taskTypes.nonEmpty, "taskTypes cannot be empty")
  require(taskTypes.forall(t => t == "classification" || t == "regression"),
    "taskTypes must be 'classification' or 'regression'")
  require(towerParamsList.size == taskTypes.size || towerParamsList.isEmpty,
    "towerParamsList length must match taskTypes size")

  private val targetDevice = new Device(device)
  private val nTask = taskTypes.size

  // Feature dimensions
  private val bottomDims: Int = features.map(_.embedDim).sum

  // Embedding layer
  private val embedding = new EmbeddingLayer(features, features.head.embedDim, device)
  register_module("embedding", embedding)

  // Bottom MLP (shared)
  private val bottomMlp: MLP = {
    val dims = bottomParams.getOrElse("dims", List(128L)).asInstanceOf[List[Long]]
    val activation = bottomParams.getOrElse("activation", "relu").asInstanceOf[String]
    val dropout = bottomParams.getOrElse("dropout", 0.0f).asInstanceOf[Float]
    new MLP(bottomDims, dims, dims.last, activation, dropout, outputLayer = false, device = device)
  }
  register_module("bottom_mlp", bottomMlp)

  // Tower dims
  private val towerDims = bottomParams.getOrElse("dims", List(128L)).asInstanceOf[List[Long]].last

  // ====== Store typed references in Map for type-safe Scala access ======
  // Towers - Map for type-safe access
  private val towersMap: Map[String, MLP] = (0 until nTask).map { i =>
    val name = s"tower_$i"
    val params = if (towerParamsList.isEmpty) Map[String, Any]() else towerParamsList(i)
    val dims = params.getOrElse("dims", List(towerDims)).asInstanceOf[List[Long]]
    val activation = params.getOrElse("activation", "relu").asInstanceOf[String]
    val dropout = params.getOrElse("dropout", 0.0f).asInstanceOf[Float]
    val tower = new MLP(towerDims, dims, 1, activation, dropout, outputLayer = true, device = device)
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
    val inputBottom = embedding.forward(sparseFeats = x, sequenceFeats = Map(), squeeze = true)

    // Bottom
    val bottomOut = bottomMlp.forward(inputBottom)

    // Tower and prediction outputs
    val outputs = (0 until nTask).map { i =>
      val tower = towersMap(s"tower_$i")
      val predictLayer = predictLayersMap(s"predictLayer_$i")
      val towerOut = tower.forward(bottomOut)
      predictLayer.forward(towerOut)
    }

    // Concatenate: torch.cat(ys, dim=1)
    torch.cat(new TensorVector(outputs: _*), 1)
  }
}

/**
 * SharedBottom companion object with factory methods.
 */
object SharedBottom {
  def apply(
             features: List[Feature],
             taskTypes: List[String] = List("classification"),
             bottomParams: Map[String, Any] = Map("dims" -> List(128L), "activation" -> "relu", "dropout" -> 0.0f),
             towerParamsList: List[Map[String, Any]] = List(),
             device: String = DeviceSupport.backend
           ): SharedBottom = {
    new SharedBottom(features, taskTypes, bottomParams, towerParamsList, device)
  }
}