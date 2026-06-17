package torchrec.models.multi_task

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.basic.features._
import torchrec.basic.layers.{EmbeddingLayer, MLP}
import torchrec.utils.DeviceSupport

/**
 * Single-Task Model (Independent Multi-Task)
 *
 * Each task has its own independent embedding layer, bottom network, and tower.
 * No parameter sharing between tasks — the opposite of SharedBottom.
 * This provides an upper bound for multi-task model performance.
 *
 * Architecture:
 *   Input → [Embedding_i + Bottom_i + Tower_i for each task] → Task Outputs
 *
 * @param features List of sparse features
 * @param taskNames Names of tasks
 * @param embedDim Embedding dimension
 * @param bottomDims Hidden layer dimensions for task-specific bottom networks
 * @param towerDims Hidden layer dimensions for task-specific towers
 * @param dropout Dropout rate
 * @param device Device for computation
 */
class SingleTaskModel(
  features: List[Feature],
  taskNames: List[String],
  embedDim: Int = 8,
  bottomDims: List[Long] = List(128L),
  towerDims: List[Long] = List(64L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(taskNames.nonEmpty, "taskNames cannot be empty")

  private val targetDevice = new Device(device)
  private val nTask = taskNames.size

  // Feature dimensions
  private val inputDims: Int = features.map(_.embedDim).sum

  // ====== Store typed references in List for type-safe Scala access ======
  // Per-task embedding layers - List for type-safe access
  private val taskEmbeddingsList: List[EmbeddingLayer] = (0 until nTask).map { i =>
    val emb = new EmbeddingLayer(features, embedDim, device)
    register_module(s"embedding_$i", emb)
    emb
  }.toList

  // Per-task bottom networks - List for type-safe access
  private val taskBottomsList: List[MLP] = (0 until nTask).map { i =>
    val bottom = new MLP(inputDims, bottomDims, bottomDims.last, "relu", dropout, outputLayer = false, device = device)
    register_module(s"bottom_$i", bottom)
    bottom
  }.toList

  // Per-task towers - List for type-safe access
  private val taskTowersList: List[MLP] = (0 until nTask).map { i =>
    val tower = new MLP(bottomDims.last.toInt, towerDims, 1, "relu", dropout, outputLayer = true, device = device)
    register_module(s"tower_$i", tower)
    tower
  }.toList

  def forward(x: Map[String, Tensor]): Tensor = {
    // Process each task independently and collect outputs
    val outputs = (0 until nTask).map { i =>
      // Get task-specific embedding, bottom, and tower
      val emb = taskEmbeddingsList(i)
      val bottom = taskBottomsList(i)
      val tower = taskTowersList(i)

      // Forward pass: embedding -> bottom -> tower -> sigmoid
      val embeddings = emb.forward(sparseFeats = x, sequenceFeats = Map.empty, squeeze = true)
      val bottomOut = bottom.forward(embeddings)
      val towerOut = tower.forward(bottomOut)
      torch.sigmoid(towerOut)
    }

    // Concatenate: torch.cat(ys, dim=1)
    torch.cat(new TensorVector(outputs: _*), 1)
  }
}

/**
 * SingleTaskModel companion object with factory methods.
 */
object SingleTaskModel {
  def apply(
    features: List[Feature],
    taskNames: List[String] = List("cvr", "ctr"),
    embedDim: Int = 8,
    bottomDims: List[Long] = List(128L),
    towerDims: List[Long] = List(64L),
    dropout: Float = 0.2f,
    device: String = DeviceSupport.backend
  ): SingleTaskModel = {
    new SingleTaskModel(features, taskNames, embedDim, bottomDims, towerDims, dropout, device)
  }
}