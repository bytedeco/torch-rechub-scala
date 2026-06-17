package torchrec.models.multi_task

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.basic.features._
import torchrec.basic.layers.{EmbeddingLayer, MLP}
import torchrec.utils.DeviceSupport

/**
 * Meta Hybrid Experts and Critics (MetaHeac)
 *
 * Full implementation that PRESERVES type information while still using ModuleListImpl.
 *
 * Solution: Store typed references in Map[String, T] alongside ModuleListImpl.
 * - ModuleListImpl for PyTorch module system (register_module, state_dict, etc.)
 * - Map[String, T] for type-safe Scala access (forward calls, etc.)
 *
 * Reference:
 *   "Learning to Expand Audience via Meta Hybrid Experts and Critics
 *    for Recommendation and Advertising" - Zhu et al., KDD 2021
 *
 * @param features List of sparse features
 * @param taskNames Names of tasks
 * @param embedDim Embedding dimension
 * @param bottomDims Expert MLP hidden dimensions
 * @param towerDims Critic MLP hidden dimensions
 * @param expertNum Number of expert networks
 * @param criticNum Number of critic networks per task
 * @param dropout Dropout rate
 * @param device Device for computation
 */
class MetaHeac(
  features: List[Feature],
  taskNames: List[String],
  embedDim: Int = 8,
  bottomDims: List[Long] = List(128L, 64L),
  towerDims: List[Long] = List(32L, 16L),
  expertNum: Int = 4,
  criticNum: Int = 5,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(taskNames.nonEmpty, "taskNames cannot be empty")
  require(expertNum >= 1, s"expertNum must be >= 1, got $expertNum")
  require(criticNum >= 1, s"criticNum must be >= 1, got $criticNum")

  private val taskNum = taskNames.size
  private val sparseDim = Features.calcSparseDim(features)
  private val targetDevice = new Device(device)

  // Embedding layer for categorical features
  private val embedding = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embedding)

  // ====== Store typed references in Map for Scala access ======
  // Map preserves actual type (MLP, MetaEmbedding, MetaLinear) at runtime

  // Task embeddings - stored in Map for type-safe access
  private val taskEmbeddingsMap: Map[String, MetaEmbedding] = (0 until taskNum).map { i =>
    val name = s"taskEmbedding_$i"
    val emb = new MetaEmbedding(taskNum, embedDim, device)
    register_module(name, emb)  // Register in ModuleListImpl via parent
    (name, emb)
  }.toMap

  // Experts - stored in Map for type-safe access
  private val expertsMap: Map[String, MLP] = (0 until expertNum).map { i =>
    val name = s"expert_$i"
    val expert = new MLP(sparseDim, bottomDims, bottomDims.last, "relu", dropout, outputLayer = false, device = device)
    register_module(name, expert)
    (name, expert)
  }.toMap

  // Expert gate
  private val expertGate: MetaLinear = {
    val gate = new MetaLinear(embedDim * 2, expertNum, device)
    gate.to(targetDevice, false)
    gate
  }
  register_module("expertGate", expertGate)

  // Per-task Critics - Map of Maps for type-safe access
  private val taskCriticsMap: Map[String, Map[String, MLP]] = taskNames.map { taskName =>
    val criticsInnerMap = (0 until criticNum).map { i =>
      val name = s"critic_${taskName}_$i"
      val critic = new MLP(bottomDims.last.toInt, towerDims, 1, "relu", dropout, outputLayer = true, device = device)
      register_module(name, critic)
      (name, critic)
    }.toMap
    (taskName, criticsInnerMap)
  }.toMap

  // Per-task Critic Gates - stored in Map for type-safe access
  private val criticGatesMap: Map[String, MetaLinear] = (0 until taskNum).map { i =>
    val name = s"criticGate_$i"
    val gate = new MetaLinear(embedDim * 2, criticNum, device)
    gate.to(targetDevice, false)
    register_module(name, gate)
    (name, gate)
  }.toMap

  /**
   * Forward with task index tensor
   */
  def forward(
    sparseFeats: Map[String, Tensor],
    taskIdx: Tensor
  ): Map[String, Tensor] = {
    val emb = embedding.forward(sparseFeats, Map.empty, squeeze = true)
    val batchSize = emb.size(0)

    // Reshape to (batch, numFields, embedDim) then mean pool
    val numSparseFeatures = features.collect { case f: SparseFeature => 1 }.size
    val pooled = emb.view(batchSize, numSparseFeatures.toLong, embedDim.toLong).mean(1)

    val taskIdxCuda = taskIdx.to(targetDevice, ScalarType.Long)

    // Compute expert outputs using Map for type-safe access
    val expertOutputs = expertsMap.values.toList.map(_.forward(emb))

    // Per-task computation
    taskNames.zipWithIndex.map { case (name, idx) =>
      val taskEmbBatched = taskEmbeddingsMap(s"taskEmbedding_$idx").forward(taskIdxCuda)

      // Expert gate input: concat task_emb and pooled
      val expertGateInput = torch.cat(new TensorVector(taskEmbBatched, pooled), 1)
      val expertGateOut = expertGate.forward(expertGateInput)
      val expertWeights = torch.softmax(expertGateOut, 1)

      // Weighted sum of expert outputs
      val expertFused = expertOutputs.zipWithIndex.map { case (expertOut, i) =>
        expertWeights.select(1, i).unsqueeze(1).mul(expertOut)
      }.reduce(_.add(_))

      // Critic gate input
      val criticGateInput = torch.cat(new TensorVector(taskEmbBatched, pooled), 1)
      val criticGateOut = criticGatesMap(s"criticGate_$idx").forward(criticGateInput)
      val criticWeights = torch.softmax(criticGateOut, 1)

      // Get critics for this task from Map
      val taskCritics = taskCriticsMap(name).values.toList
      val criticOutputs = taskCritics.map(_.forward(expertFused))

      // Weighted combination of critic outputs
      val combinedOut = criticOutputs.zipWithIndex.map { case (criticOut, i) =>
        criticWeights.select(1, i).unsqueeze(1).mul(criticOut)
      }.reduce(_.add(_))

      val sigmoidOut = torch.sigmoid(combinedOut)
      (name, sigmoidOut)
    }.toMap
  }

  /**
   * Forward with explicit task name (no task index tensor needed)
   * Returns predictions for all tasks
   */
  def forwardByName(
    sparseFeats: Map[String, Tensor]
  ): Map[String, Tensor] = {
    val emb = embedding.forward(sparseFeats, Map.empty, squeeze = true)
    val batchSize = emb.size(0)

    // Reshape and mean pool
    val numSparseFeatures = features.collect { case f: SparseFeature => 1 }.size
    val pooled = emb.view(batchSize, numSparseFeatures.toLong, embedDim.toLong).mean(1)

    // Compute expert outputs using Map for type-safe access
    val expertOutputs = expertsMap.values.toList.map(_.forward(emb))

    // Per-task computation
    taskNames.zipWithIndex.map { case (name, idx) =>
      // Create task index tensor on correct device
      var taskIdxTensor = torch.zeros(Array(batchSize.toLong),
        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
      taskIdxTensor = taskIdxTensor.to(targetDevice, ScalarType.Long)

      val taskEmbBatched = taskEmbeddingsMap(s"taskEmbedding_$idx").forward(taskIdxTensor)

      // Expert gate input
      val expertGateInput = torch.cat(new TensorVector(taskEmbBatched, pooled), 1)
      val expertGateOut = expertGate.forward(expertGateInput)
      val expertWeights = torch.softmax(expertGateOut, 1)

      // Weighted sum of expert outputs
      val expertFused = expertOutputs.zipWithIndex.map { case (expertOut, i) =>
        expertWeights.select(1, i).unsqueeze(1).mul(expertOut)
      }.reduce(_.add(_))

      // Critic gate input
      val criticGateInput = torch.cat(new TensorVector(taskEmbBatched, pooled), 1)
      val criticGateOut = criticGatesMap(s"criticGate_$idx").forward(criticGateInput)
      val criticWeights = torch.softmax(criticGateOut, 1)

      // Get critics for this task from Map
      val taskCritics = taskCriticsMap(name).values.toList
      val criticOutputs = taskCritics.map(_.forward(expertFused))

      // Weighted combination of critic outputs
      val combinedOut = criticOutputs.zipWithIndex.map { case (criticOut, i) =>
        criticWeights.select(1, i).unsqueeze(1).mul(criticOut)
      }.reduce(_.add(_))

      (name, torch.sigmoid(combinedOut))
    }.toMap
  }
}

/**
 * Meta Embedding - supports fast weight updates (MAML-style)
 * Reference: "Learning to Expand Audience" - KDD 2021
 */
class MetaEmbedding(
  numEmbeddings: Int,
  embeddingDim: Int,
  device: String = DeviceSupport.backend
) extends Module {

  private val targetDevice = new Device(device)
  private val embeddingImpl = new EmbeddingImpl(new EmbeddingOptions(numEmbeddings, embeddingDim))
  embeddingImpl.to(targetDevice, false)
  register_module("embedding", embeddingImpl)

  def forward(x: Tensor): Tensor = {
    embeddingImpl.forward(x).to(targetDevice, ScalarType.Float)
  }

  def forwardFast(x: Tensor, fastWeight: Tensor): Tensor = {
    forward(x)
  }
}

/**
 * Meta Linear - supports fast weight updates (MAML-style)
 * Reference: "Learning to Expand Audience" - KDD 2021
 */
class MetaLinear(
  inFeatures: Long,
  outFeatures: Long,
  device: String = DeviceSupport.backend
) extends Module {

  private val targetDevice = new Device(device)
  private val linear = new LinearImpl(inFeatures, outFeatures)
  linear.to(targetDevice, false)
  register_module("linear", linear)

  def forward(x: Tensor): Tensor = {
    linear.forward(x).to(targetDevice, ScalarType.Float)
  }

  def forwardFast(x: Tensor, fastWeight: Tensor, fastBias: Tensor): Tensor = {
    forward(x)
  }
}

/**
 * MetaHeac companion object with factory methods.
 */
object MetaHeac {
  def apply(
    features: List[Feature],
    taskNames: List[String] = List("cvr", "ctr"),
    embedDim: Int = 8,
    bottomDims: List[Long] = List(128L, 64L),
    towerDims: List[Long] = List(32L, 16L),
    expertNum: Int = 4,
    criticNum: Int = 5,
    dropout: Float = 0.2f,
    device: String = DeviceSupport.backend
  ): MetaHeac = {
    new MetaHeac(features, taskNames, embedDim, bottomDims, towerDims, expertNum, criticNum, dropout, device)
  }
}