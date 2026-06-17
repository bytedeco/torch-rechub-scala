package torchrec.models.multi_task

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.basic.features.*
import torchrec.basic.layers.{EmbeddingLayer, MLP}
import torchrec.utils.DeviceSupport

import scala.math

/**
 * AITM - Adaptive Information Transfer Multi-task Learning
 *
 * Full implementation matching the Python torch_rechub AITM model.
 * Uses information transfer between tasks with attention mechanism.
 *
 * Reference:
 *   "Modeling the Sequential Dependence among Audience Multi-step Conversions
 *   with Multi-task Learning in Targeted Display Advertising"
 *   KDD'2021 - https://arxiv.org/abs/2105.08489
 *
 * Architecture:
 *   - Bottom MLPs: Task-specific feature extractors (ModuleListImpl)
 *   - Information Gates: Control information flow between tasks (ModuleListImpl)
 *   - AITs (Attention Information Transfer): Transfer knowledge from previous tasks (ModuleListImpl)
 *   - Towers: Task-specific output predictors (ModuleListImpl)
 *
 * @param features List of features
 * @param nTask Number of binary classification tasks
 * @param bottomParams MLP parameters for bottom: dims, activation, dropout
 * @param towerParamsList List of tower params dict (one per task)
 * @param device Device for computation
 */
class AITM(
            features: List[Feature],
            nTask: Int,
            bottomParams: Map[String, Any],
            towerParamsList: List[Map[String, Any]],
            device: String = DeviceSupport.backend
          ) extends Module {

  require(nTask > 0, "nTask must be positive")
  require(features.nonEmpty, "features cannot be empty")
  require(towerParamsList.size == nTask, "towerParamsList length must match nTask")

  private val targetDevice = new Device(DeviceSupport.backend)

  // Feature dimensions
  private val inputDims: Int = features.map(_.embedDim).sum

  // Embedding layer
  private val embedding = new EmbeddingLayer(features, features.head.embedDim, DeviceSupport.backend)
  register_module("embedding", embedding)

  // Bottom MLPs - stored in a List for direct access
  private val bottoms: List[MLP] = {
    val dims = bottomParams.getOrElse("dims", List(128L)).asInstanceOf[List[Long]]
    val activation = bottomParams.getOrElse("activation", "relu").asInstanceOf[String]
    val dropout = bottomParams.getOrElse("dropout", 0.0f).asInstanceOf[Float]
    (0 until nTask).map { i =>
      val bottom = new MLP(inputDims, dims, dims.last, activation, dropout, outputLayer = false, device = DeviceSupport.backend)
      register_module(s"bottom_$i", bottom)
      bottom
    }.toList
  }

  // Towers - stored in a List for direct access
  private val towers: List[MLP] = {
    val bottomDim = bottomParams.getOrElse("dims", List(128L)).asInstanceOf[List[Long]].last
    (0 until nTask).map { i =>
      val params = towerParamsList(i)
      val dims = params.getOrElse("dims", List(bottomDim)).asInstanceOf[List[Long]]
      val activation = params.getOrElse("activation", "relu").asInstanceOf[String]
      val dropout = params.getOrElse("dropout", 0.0f).asInstanceOf[Float]
      val tower = new MLP(bottomDim, dims, 1, activation, dropout, outputLayer = true, device = DeviceSupport.backend)
      register_module(s"tower_$i", tower)
      tower
    }.toList
  }

  // Info Gates - stored in a List for direct access (nTask - 1 gates)
  private val infoGates: List[MLP] = {
    if (nTask > 1) {
      val bottomDim = bottomParams.getOrElse("dims", List(128L)).asInstanceOf[List[Long]].last
      (0 until nTask - 1).map { i =>
        val gate = new MLP(bottomDim, List(bottomDim), bottomDim, "relu", 0.0f, outputLayer = false, device = DeviceSupport.backend)
        register_module(s"infoGate_$i", gate)
        gate
      }.toList
    } else List.empty
  }

  // AITs (Attention Information Transfer) - stored in a List for direct access
  private val aits: List[AttentionLayer] = {
    if (nTask > 1) {
      val dim = bottomParams.getOrElse("dims", List(128L)).asInstanceOf[List[Long]].last.toInt
      (0 until nTask - 1).map { i =>
        val ait = new AttentionLayer(dim, DeviceSupport.backend)
        register_module(s"ait_$i", ait)
        ait
      }.toList
    } else List.empty
  }

  def forward(x: Map[String, Tensor]): Tensor = {
    // Embedding: [batch_size, input_dims]
    val embedX = embedding.forward(sparseFeats = x, sequenceFeats = Map(), squeeze = true)

    // Bottom outputs: List of [batch_size, bottom_dims[-1]]
    val inputTowers = bottoms.map(_.forward(embedX))

    // Information transfer: for task 1 to n-1
    var currentInputs = inputTowers
    for (i <- 1 until nTask) {
      // info = info_gate[i-1](input_towers[i-1]).unsqueeze(1)
      val info = infoGates(i - 1).forward(currentInputs(i - 1)).unsqueeze(1)

      // ait_input = torch.cat([input_towers[i].unsqueeze(1), info], dim=1)
      val aitInput = torch.cat(new TensorVector(inputTowers(i).unsqueeze(1), info), 1)

      // input_towers[i] = aits[i-1](ait_input)
      currentInputs = currentInputs.updated(i, aits(i - 1).forward(aitInput))
    }

    // Tower outputs with sigmoid
    val outputs = (0 until nTask).map { i =>
      torch.sigmoid(towers(i).forward(currentInputs(i)))
    }

    // Concatenate: torch.cat(ys, dim=1)
    torch.cat(new TensorVector(outputs.toSeq*), 1)
  }
}

/**
 * AttentionLayer for AITM information transfer
 *
 * Args:
 *   dim: attention dimension
 *
 * Shape:
 *   Input: (batch_size, 2, dim)
 *   Output: (batch_size, dim)
 */
class AttentionLayer(
                      dim: Int = 32,
                      device: String = DeviceSupport.backend
                    ) extends Module {

  private val targetDevice = new Device(device)

  // Q, K, V layers
  private val qLayer: LinearImpl = {
    val opts = new LinearOptions(dim, dim)
    opts.bias().put(false)
    val l = new LinearImpl(opts)
    l.to(targetDevice, false)
    l
  }
  register_module("q_layer", qLayer)

  private val kLayer: LinearImpl = {
    val opts = new LinearOptions(dim, dim)
    opts.bias().put(false)
    val l = new LinearImpl(opts)
    l.to(targetDevice, false)
    l
  }
  register_module("k_layer", kLayer)

  private val vLayer: LinearImpl = {
    val opts = new LinearOptions(dim, dim)
    opts.bias().put(false)
    val l = new LinearImpl(opts)
    l.to(targetDevice, false)
    l
  }
  register_module("v_layer", vLayer)

  def forward(x: Tensor): Tensor = {
    // x: (batch_size, 2, dim)
    val batchSize = x.size(0)

    // Q, K, V: each (batch_size, 2, dim)
    val Q = qLayer.forward(x)
    val K = kLayer.forward(x)
    val V = vLayer.forward(x)

    // a = softmax(sum(Q * K, -1) / sqrt(dim))
    // Shape: (batch_size, 2)
    val scale = new Scalar(math.sqrt(dim).toFloat)
    val scores = torch.mul(Q, K).sum(-1).div(scale)
    val a = torch.softmax(scores, 1)

    // outputs = sum(a.unsqueeze(-1) * V, dim=1)
    // Shape: (batch_size, dim)
    val aExpanded = a.unsqueeze(-1) // (batch_size, 2, 1)
    val weightedV = torch.mul(aExpanded, V) // (batch_size, 2, dim)
    weightedV.sum(1) // (batch_size, dim)
  }
}

/**
 * AITM companion object with factory methods.
 */
object AITM {
  def apply(
             features: List[Feature],
             nTask: Int,
             bottomParams: Map[String, Any] = Map("dims" -> List(128L), "activation" -> "relu"),
             towerParamsList: List[Map[String, Any]] = List(Map("dims" -> List(64L), "activation" -> "relu")),
             device: String = DeviceSupport.backend
           ): AITM = {
    new AITM(features, nTask, bottomParams, towerParamsList, device)
  }
}