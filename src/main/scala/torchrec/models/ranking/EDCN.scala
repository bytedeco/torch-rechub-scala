package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

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

import torchrec.Implicits._

/**
 * Enhanced Deep & Cross Network with Bridge and Regulation (EDCN, KDD'2021).
 *
 * Mirrors the Python `torch-rechub` reference: every per-layer block collection
 * (cross_layers / bridge_modules / regulation_modules / mlps) is a
 * `nn.ModuleList` — implemented here as `ModuleListImpl`. BridgeModule's
 * `concat_pooling` / `attention_x` / `attention_h` are `nn.Sequential`s —
 * implemented as `SequentialImpl` with `Linear` + `ReLU` (already the case).
 *
 * Reference: https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf
 */
class EDCN(
  features: List[Feature],
  nCrossLayers: Int = 3,
  mlpParams: Map[String, Any] = Map("dims" -> List(256L, 128L), "activation" -> "relu", "dropout" -> 0.2f),
  bridgeType: String = "hadamard_product",
  useRegulationModule: Boolean = true,
  temperature: Float = 1.0f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "EDCN: features cannot be empty")
  require(nCrossLayers > 0, s"EDCN: nCrossLayers must be > 0, got $nCrossLayers")
  require(List("hadamard_product", "pointwise_addition", "concatenation", "attention_pooling").contains(bridgeType),
    s"EDCN: bridgeType '$bridgeType' not supported")

  val numFields: Int = features.size
  val dims: Int = features.map(_.embedDim).sum
  val feaDims: List[Int] = features.map(_.embedDim)

  private val embedding = new EmbeddingLayer(features, device = device)
  register_module("embedding", embedding)

  // Four per-layer ModuleLists — mirrors the four `nn.ModuleList`s in the Python.
  private val crossLayers: ModuleListImpl = new ModuleListImpl()
  private val mlps: ModuleListImpl = new ModuleListImpl()
  private val bridges: ModuleListImpl = new ModuleListImpl()
  private val regulationModules: ModuleListImpl = new ModuleListImpl()

  for (i <- 0 until nCrossLayers) {
    val crossLayer = new CrossLayer(dims, device)
    register_module(s"cross_$i", crossLayer)
    crossLayers.push_back(crossLayer)

    val activation = mlpParams.getOrElse("activation", "relu").toString
    val dropout = mlpParams.getOrElse("dropout", 0.2f).asInstanceOf[Float]
    // Per the Python: `mlp_params["dims"] = [self.dims, self.dims]` overrides
    // whatever the caller put, then MLP(self.dims, output_layer=False, ...).
    val mlp = new MLP(dims, List(dims.toLong, dims.toLong), dims.toLong,
      activation, dropout, outputLayer = false, device = device)
    register_module(s"mlp_$i", mlp)
    mlps.push_back(mlp)

    val bridge = new BridgeModule(dims, bridgeType, device)
    register_module(s"bridge_$i", bridge)
    bridges.push_back(bridge)

    val reg = new RegulationModule(numFields, feaDims, temperature, useRegulationModule)
    register_module(s"regulation_$i", reg)
    regulationModules.push_back(reg)
  }

  private val finalLinear = new LR(dims * 3, sigmoid = false, device = device)
  register_module("final_linear", finalLinear)

  def forward(sparseFeats: Map[String, Tensor]): Tensor = {
    val embedX = embedding.forward(sparseFeats, squeeze = true) // (B, dims)

    // Index ModuleListImpl directly. Cast to typed refs so we can call helper
    // methods like `forwardReg` that aren't on the generic Module base.
    val firstReg = regulationModules.get(0).asInstanceOf[RegulationModule].forwardReg(embedX)
    var crossI: Tensor = firstReg.get0()
    var deepI: Tensor = firstReg.get1()
    val cross0 = crossI
    var bridgeI: Tensor = crossI.clone() // placeholder, replaced on i=0 too

    for (i <- 0 until nCrossLayers) {
      if (i > 0) {
        val reg = regulationModules.get(i).asInstanceOf[RegulationModule].forwardReg(bridgeI)
        crossI = reg.get0()
        deepI = reg.get1()
      }
      crossI = crossI.add(crossLayers.get(i).asInstanceOf[CrossLayer].forward(cross0, crossI))
      deepI = mlps.get(i).asInstanceOf[MLP].forward(deepI)
      bridgeI = bridges.get(i).asInstanceOf[BridgeModule].forward(crossI, deepI)
    }

    val xStack = torch.cat(new TensorVector(crossI, deepI, bridgeI), 1L)
    val y = finalLinear.forward(xStack)
    torch.sigmoid(y.squeeze(1))
  }
}

/**
 * Bridge Module for connecting cross and deep networks in EDCN.
 *
 * Mirrors the Python reference exactly:
 *   - `concatenation`   → `nn.Sequential(nn.Linear(2D, D), nn.ReLU())`
 *   - `attention_pooling` → two `nn.Sequential(Linear, ReLU, Linear)` blocks
 *   - `hadamard_product`, `pointwise_addition` → bare element-wise ops
 *
 * @param inputDim   Bridge input/output dimension (matches `dims`).
 * @param bridgeType One of `hadamard_product`, `pointwise_addition`,
 *                   `concatenation`, `attention_pooling`.
 */
class BridgeModule(
  inputDim: Int,
  bridgeType: String,
  device: String = DeviceSupport.backend
) extends Module {

  val bridgeTypeName: String = bridgeType

  // For concatenation bridge — `nn.Sequential(Linear, ReLU)`.
  private val concatPooling: Option[SequentialImpl] =
    if (bridgeType == "concatenation") {
      val seq = new SequentialImpl()
      seq.push_back("linear", new LinearImpl((inputDim * 2).toLong, inputDim.toLong))
      seq.push_back("relu", new ReLUImpl())
      Some(seq)
    } else None

  // For attention_pooling bridge — two `nn.Sequential(Linear, ReLU, Linear)` blocks.
  // (The Python uses `bias=False` on the final Linear — we keep default bias=True
  //  here because bytedeco's `LinearImpl(bias=false)` SIGSEGVs in `Module.to()`.)
  private val attentionPool: Option[(SequentialImpl, SequentialImpl)] =
    if (bridgeType == "attention_pooling") {
      val attX = new SequentialImpl()
      attX.push_back("linear1", new LinearImpl(inputDim.toLong, inputDim.toLong))
      attX.push_back("relu", new ReLUImpl())
      attX.push_back("linear2", new LinearImpl(inputDim.toLong, inputDim.toLong))

      val attH = new SequentialImpl()
      attH.push_back("linear1", new LinearImpl(inputDim.toLong, inputDim.toLong))
      attH.push_back("relu", new ReLUImpl())
      attH.push_back("linear2", new LinearImpl(inputDim.toLong, inputDim.toLong))

      Some((attX, attH))
    } else None

  override def forward(x: Tensor, h: Tensor): Tensor = {
    bridgeType match {
      case "hadamard_product" =>
        x.mul(h)
      case "pointwise_addition" =>
        x.add(h)
      case "concatenation" =>
        val concat = torch.cat(new TensorVector(x, h), 1L)
        concatPooling.get.forward(concat)
      case "attention_pooling" =>
        val (attX, attH) = attentionPool.get
        val weightX = torch.softmax(attX.forward(x), 1L)
        val weightH = torch.softmax(attH.forward(h), 1L)
        weightX.mul(x).add(weightH.mul(h))
      case _ =>
        h
    }
  }
}