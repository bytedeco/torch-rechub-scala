package torchrec.models.multi_task

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.basic.features._
import torchrec.basic.layers.{EmbeddingLayer, MLP}
import torchrec.utils.DeviceSupport

/**
 * One-gate Mixture-of-Experts (OMoE)
 *
 * Full implementation with type-safe access pattern for experts and towers.
 *
 * A simplified MoE variant with a single shared gate across all tasks.
 * Unlike MMOE which has per-task gates, OMoE uses one gate that routes
 * all tasks through the same expert selection.
 *
 * Architecture:
 *   Input → Embedding → Shared Gate → Expert Routing → Per-Task Towers → Outputs
 *
 * Reference:
 *   "Modeling Task Relationships in Multi-Task Learning with Multi-gate Mixture-of-Experts"
 *   Ma et al., KDD 2018 - The OMoE is the single-gate variant from the same paper.
 *
 * @param features List of sparse features
 * @param taskNames Names of tasks
 * @param embedDim Embedding dimension
 * @param numExperts Number of expert networks
 * @param expertDims Hidden layer dimensions for expert networks
 * @param towerDims Hidden layer dimensions for task towers
 * @param dropout Dropout rate
 * @param device Device for computation
 */
class OMoE(
  features: List[Feature],
  taskNames: List[String],
  embedDim: Int = 8,
  numExperts: Int = 4,
  expertDims: List[Long] = List(128L),
  towerDims: List[Long] = List(64L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(taskNames.nonEmpty, "taskNames cannot be empty")
  require(numExperts >= 1, s"numExperts must be >= 1, got $numExperts")

  private val targetDevice = new Device(device)
  private val nTask = taskNames.size

  // Embedding layer
  private val embedding = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embedding)

  // Feature dimensions
  private val inputDims: Int = features.map(_.embedDim).sum

  // ====== Store typed references in List for type-safe Scala access ======
  // Experts - List for type-safe access
  // Expert input is pooled embedding (embedDim), not inputDims
  private val expertsList: List[MLP] = (0 until numExperts).map { i =>
    val expert = new MLP(embedDim, expertDims, expertDims.last, "relu", dropout, outputLayer = false, device = device)
    expert.to(targetDevice, false)
    register_module(s"expert_$i", expert)
    expert
  }.toList

  // Single shared gate - takes pooled embedding (embedDim) as input
  private val gate: LinearImpl = {
    val opts = new LinearOptions(embedDim, numExperts)
    val linear = new LinearImpl(opts)
    linear.to(targetDevice, false)
    linear
  }
  register_module("gate", gate)

  // Task-specific towers - List for type-safe access
  private val towersList: List[MLP] = (0 until nTask).map { i =>
    val tower = new MLP(expertDims.last.toInt, towerDims, 1, "relu", dropout, outputLayer = true, device = device)
    register_module(s"tower_$i", tower)
    tower
  }.toList

  def forward(x: Map[String, Tensor]): Tensor = {
    // EmbeddingLayer.forward() returns (batch, numFields * embedDim)
    val embeddings = embedding.forward(sparseFeats = x, sequenceFeats = Map.empty, squeeze = true)
    val batchSize = embeddings.size(0)

    // Reshape to (batch, numFields, embedDim) then sum-pool over fields
    val numSparseFeatures = features.collect { case f: SparseFeature => 1 }.size
    val reshaped = embeddings.view(batchSize, numSparseFeatures.toLong, embedDim.toLong)
    val pooled = reshaped.sum(1)  // (batch, embedDim)

    // Single shared gate (softmax over experts): (batch, num_experts)
    val gateWeights = torch.softmax(gate.forward(pooled), 1)

    // Compute expert outputs: List of (batch, expert_dim)
    val expertOutputs = expertsList.map(_.forward(pooled))

    // Route through shared gate and combine experts
    val weightedExperts = expertOutputs.zipWithIndex.map { case (expertOut, i) =>
      gateWeights.select(1, i).unsqueeze(1).mul(expertOut)
    }
    val combined = weightedExperts.reduce(_.add(_))  // (batch, expert_dim)

    // Pass through task-specific towers and get sigmoid predictions
    val outputs = towersList.map { tower =>
      torch.sigmoid(tower.forward(combined))
    }

    // Concatenate: torch.cat(ys, dim=1)
    torch.cat(new TensorVector(outputs: _*), 1)
  }
}

/**
 * OMoE companion object with factory methods.
 */
object OMoE {
  def apply(
    features: List[Feature],
    taskNames: List[String] = List("cvr", "ctr"),
    embedDim: Int = 8,
    numExperts: Int = 4,
    expertDims: List[Long] = List(128L),
    towerDims: List[Long] = List(64L),
    dropout: Float = 0.2f,
    device: String = DeviceSupport.backend
  ): OMoE = {
    new OMoE(features, taskNames, embedDim, numExperts, expertDims, towerDims, dropout, device)
  }
}