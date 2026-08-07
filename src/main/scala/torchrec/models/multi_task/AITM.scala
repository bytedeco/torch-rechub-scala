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
import torchrec.basic.layers.{EmbeddingLayer, MLP}
import torchrec.utils.DeviceSupport

import scala.collection.mutable

/**
 * AITM — Adaptive Information Transfer Multi-task framework (KDD'2021).
 *
 * Mirrors the Python reference: each task has a bottom MLP and a tower; an
 * info-gate MLP and an attention layer pass information from task `i-1` to
 * task `i` for `i = 1..n-1`. All submodules live in `ModuleListImpl`s so they
 * participate in `state_dict()` / parameter discovery.
 *
 * Reference:
 *   "Modeling the Sequential Dependence among Audience Multi-step Conversions
 *    with Multi-task Learning in Targeted Display Advertising"
 *   https://arxiv.org/abs/2105.08489
 *
 * @param features        List of feature columns.
 * @param nTask           Number of binary classification tasks.
 * @param bottomParams    Bottom MLP params: keys `dims`, `activation`, `dropout`.
 * @param towerParamsList Per-task tower params (same keys as bottomParams).
 * @param device          Device for parameters.
 */
class AITM(
  features: List[Feature],
  nTask: Int,
  bottomParams: Map[String, Any],
  towerParamsList: List[Map[String, Any]],
  device: String = DeviceSupport.backend
) extends Module {

  require(nTask > 0, "AITM: nTask must be > 0")
  require(features.nonEmpty, "AITM: features cannot be empty")
  require(towerParamsList.size == nTask,
    s"AITM: towerParamsList.size (${towerParamsList.size}) must equal nTask ($nTask)")

  private val bottomDims: List[Long] =
    bottomParams.getOrElse("dims", List(128L)).asInstanceOf[List[Long]]
  private val bottomActivation: String =
    bottomParams.getOrElse("activation", "relu").asInstanceOf[String]
  private val bottomDropout: Float =
    bottomParams.getOrElse("dropout", 0.0f).asInstanceOf[Float]
  private val bottomLast: Long = bottomDims.last
  private val inputDims: Int = features.map(_.embedDim).sum

  // Embedding
  private val embedding = new EmbeddingLayer(features, features.head.embedDim, device)
  register_module("embedding", embedding)

  // Bottom MLPs — one per task, mirrors nn.ModuleList(MLP(...) for i in range(nTask)).
  private val bottoms: ModuleListImpl = new ModuleListImpl()
  for (i <- 0 until nTask) {
    val bottom = new MLP(inputDims, bottomDims, bottomLast, bottomActivation, bottomDropout,
      outputLayer = false, device = device)
    register_module(s"bottom_$i", bottom)
    bottoms.push_back(bottom)
  }

  // Towers — one per task.
  private val towers: ModuleListImpl = new ModuleListImpl()
  for (i <- 0 until nTask) {
    val params = towerParamsList(i)
    val dims = params.getOrElse("dims", List(bottomLast)).asInstanceOf[List[Long]]
    val activation = params.getOrElse("activation", bottomActivation).asInstanceOf[String]
    val dropout = params.getOrElse("dropout", bottomDropout).asInstanceOf[Float]
    val tower = new MLP(bottomLast, dims, 1L, activation, dropout, outputLayer = true, device = device)
    register_module(s"tower_$i", tower)
    towers.push_back(tower)
  }

  // Info gates — one fewer than tasks; gates task `i-1`'s output for transfer to task `i`.
  private val infoGates: ModuleListImpl = new ModuleListImpl()
  for (_ <- 1 until nTask) {
    val gate = new MLP(bottomLast, List(bottomLast), bottomLast, "relu", 0.0f,
      outputLayer = false, device = device)
    infoGates.push_back(gate)
    register_module(s"infoGate_${infoGates.size() - 1}", gate)
  }

  // Attention layers — one per info gate.
  private val aits: ModuleListImpl = new ModuleListImpl()
  for (_ <- 1 until nTask) {
    val ait = new AttentionLayer(bottomLast.toInt, device)
    aits.push_back(ait)
    register_module(s"ait_${aits.size() - 1}", ait)
  }

  def forward(x: Map[String, Tensor]): Tensor = {
    // Embedding → (batch, inputDims).
    val embedX = embedding.forward(sparseFeats = x, sequenceFeats = Map.empty, squeeze = true)

    // Bottom outputs — mutable so the info-transfer loop can overwrite in place
    // exactly like the Python `input_towers[i] = self.aits[i-1](ait_input)`.
    val towerInputs = mutable.ArrayBuffer[Tensor]()
    var i = 0
    while (i < nTask) {
      towerInputs += bottoms.get(i).forward(embedX)
      i += 1
    }

    // Information transfer for tasks 1..n-1.
    var k = 0
    while (k < nTask - 1) {
      // info = info_gates[k](input_towers[k]).unsqueeze(1)  → (batch, 1, dim)
      val info = infoGates.get(k).forward(towerInputs(k)).unsqueeze(1)
      // ait_input = cat([input_towers[k+1].unsqueeze(1), info], dim=1)  → (batch, 2, dim)
      val aitInput = torch.cat(
        new TensorVector(towerInputs(k + 1).unsqueeze(1), info), 1)
      // input_towers[k+1] = aits[k](ait_input)  → (batch, dim)
      towerInputs(k + 1) = aits.get(k).forward(aitInput)
      k += 1
    }

    // Tower + sigmoid, then concat along dim=1.
    val ys = mutable.ArrayBuffer[Tensor]()
    var j = 0
    while (j < nTask) {
      ys += torch.sigmoid(towers.get(j).forward(towerInputs(j)))
      j += 1
    }
    torch.cat(new TensorVector(ys.toSeq: _*), 1L)
  }
}

/**
 * AITM factory — identical to the constructor.
 */
object AITM {
  def apply(
    features: List[Feature],
    nTask: Int,
    bottomParams: Map[String, Any] = Map("dims" -> List(128L), "activation" -> "relu"),
    towerParamsList: List[Map[String, Any]] = List(Map("dims" -> List(64L), "activation" -> "relu")),
    device: String = DeviceSupport.backend
  ): AITM = new AITM(features, nTask, bottomParams, towerParamsList, device)
}

/**
 * AttentionLayer for AITM information transfer (KDD'2021).
 *
 * Mirrors the Python reference exactly:
 *   - Q/K/V are `Linear(dim, dim, bias=False)`.
 *   - scores = softmax(sum(Q*K, -1) / sqrt(dim))
 *   - output = sum(unsqueeze(scores, -1) * V, dim=1)
 *
 * Note: `Module.to(Device, false)` is intentionally skipped on the Q/K/V
 * Linears — they are constructed with `LinearOptions.bias(false)` and that
 * variant triggers a SIGSEGV inside `Module.to()` in this bytedeco build
 * (same family as the CIN/Conv1dImpl crash).
 *
 * Shape
 * -----
 * Input:  (batch, 2, dim)
 * Output: (batch, dim)
 */
class AttentionLayer(
  dim: Int,
  device: String = DeviceSupport.backend
) extends Module {

  private val qLayer = new LinearImpl(new LinearOptions(dim.toLong, dim.toLong).bias(false))
  register_module("q_layer", qLayer)
  private val kLayer = new LinearImpl(new LinearOptions(dim.toLong, dim.toLong).bias(false))
  register_module("k_layer", kLayer)
  private val vLayer = new LinearImpl(new LinearOptions(dim.toLong, dim.toLong).bias(false))
  register_module("v_layer", vLayer)

  override def forward(x: Tensor): Tensor = {
    val Q = qLayer.forward(x)
    val K = kLayer.forward(x)
    val V = vLayer.forward(x)

    // a = softmax(sum(Q * K, dim=-1) / sqrt(dim))   → (batch, 2)
    val scale = new Scalar(math.sqrt(dim.toDouble).toFloat)
    val a = torch.mul(Q, K).sum(-1).div(scale).softmax(1)

    // output = sum(unsqueeze(a, -1) * V, dim=1)     → (batch, dim)
    torch.mul(a.unsqueeze(-1), V).sum(1L)
  }
}