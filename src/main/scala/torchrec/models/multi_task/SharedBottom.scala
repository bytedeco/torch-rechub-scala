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
 * SharedBottom — Caruana 1997 multi-task baseline.
 *
 * Mirrors the Python reference: one shared bottom MLP, then per-task towers
 * and prediction layers stored in `ModuleListImpl`s.
 *
 * Reference:
 *   Caruana, R. (1997). Multitask learning. Machine learning, 28(1), 41-75.
 *
 * @param features        List of feature columns.
 * @param taskTypes       Per-task type: `"classification"` or `"regression"`.
 * @param bottomParams    Shared bottom MLP params (kept `output_layer=False`).
 * @param towerParamsList Per-task tower params.
 * @param device          Device for parameters.
 */
class SharedBottom(
  features: List[Feature],
  taskTypes: List[String],
  bottomParams: Map[String, Any] = Map(),
  towerParamsList: List[Map[String, Any]] = List(),
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "SharedBottom: features cannot be empty")
  require(taskTypes.nonEmpty, "SharedBottom: taskTypes cannot be empty")
  require(taskTypes.forall(t => t == "classification" || t == "regression"),
    "SharedBottom: taskTypes must be 'classification' or 'regression'")

  private val nTask: Int = taskTypes.size
  private val bottomDims: Int = features.map(_.embedDim).sum

  private val bottomDimsList: List[Long] =
    bottomParams.getOrElse("dims", List(128L)).asInstanceOf[List[Long]]
  private val bottomActivation: String =
    bottomParams.getOrElse("activation", "relu").asInstanceOf[String]
  private val bottomDropout: Float =
    bottomParams.getOrElse("dropout", 0.0f).asInstanceOf[Float]
  private val towerLast: Long = bottomDimsList.last

  private val embedding = new EmbeddingLayer(features, features.head.embedDim, device)
  register_module("embedding", embedding)

  // Single shared bottom MLP — `output_layer=False` as the Python mandates.
  private val bottomMlp: MLP = new MLP(
    bottomDims, bottomDimsList, towerLast, bottomActivation, bottomDropout,
    outputLayer = false, device = device)
  register_module("bottom_mlp", bottomMlp)

  // Towers — mirrors `nn.ModuleList(MLP(...) for i in range(n_task))`.
  private val towers: ModuleListImpl = new ModuleListImpl()
  for (i <- 0 until nTask) {
    val params = if (towerParamsList.isEmpty) Map[String, Any]() else towerParamsList(i)
    val dims = params.getOrElse("dims", List(towerLast)).asInstanceOf[List[Long]]
    val activation = params.getOrElse("activation", "relu").asInstanceOf[String]
    val dropout = params.getOrElse("dropout", 0.0f).asInstanceOf[Float]
    val m = new MLP(towerLast, dims, 1L, activation, dropout, outputLayer = true, device = device)
    register_module(s"tower_$i", m)
    towers.push_back(m)
  }

  // Prediction layers — mirrors `nn.ModuleList(PredictionLayer(...) for task_type in task_types)`.
  private val predictLayers: ModuleListImpl = new ModuleListImpl()
  for (i <- 0 until nTask) {
    val m = new PredictionLayer(taskTypes(i))
    register_module(s"predictLayer_$i", m)
    predictLayers.push_back(m)
  }

  def forward(x: Map[String, Tensor]): Tensor = {
    val inputBottom = embedding.forward(sparseFeats = x, sequenceFeats = Map.empty, squeeze = true)
    val shared = bottomMlp.forward(inputBottom)

    val ys = mutable.ArrayBuffer[Tensor]()
    var i = 0
    while (i < nTask) {
      val towerOut = towers.get(i).forward(shared)
      ys += predictLayers.get(i).forward(towerOut)
      i += 1
    }
    torch.cat(new TensorVector(ys.toSeq: _*), 1L)
  }
}

/**
 * SharedBottom factory — identical to the constructor.
 */
object SharedBottom {
  def apply(
    features: List[Feature],
    taskTypes: List[String] = List("classification"),
    bottomParams: Map[String, Any] = Map("dims" -> List(128L), "activation" -> "relu", "dropout" -> 0.0f),
    towerParamsList: List[Map[String, Any]] = List(),
    device: String = DeviceSupport.backend
  ): SharedBottom = new SharedBottom(features, taskTypes, bottomParams, towerParamsList, device)
}