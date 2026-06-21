package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits._

/**
 * Enhanced Deep & Cross Network with Bridge and Regulation
 * Reference: EDCN, KDD 2021
 *
 * Architecture:
 * - Multi-layer iteration with cross network, deep MLP, bridge, and regulation
 * - Bridge connects cross and deep towers
 * - Regulation module provides learnable field gating
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

  require(features.nonEmpty, "features cannot be empty")
  require(nCrossLayers > 0, "nCrossLayers must be positive")

  // Feature dimensions
  val numFields: Int = features.size
  val dims: Int = features.map(_.embedDim).sum
  val feaDims: List[Int] = features.map(_.embedDim)

  // Embedding layer
  private val embedding = new EmbeddingLayer(features, device = device)
  register_module("embedding", embedding)

  // Cross layers (ModuleList)
  private val crossLayers = (0 until nCrossLayers).map { _ =>
    new CrossLayer(dims, device)
  }.toList
  crossLayers.zipWithIndex.foreach { case (layer, i) =>
    register_module(s"cross_$i", layer)
  }

  // MLPs for deep tower (output_layer=False, dims=[dims, dims])
  private val mlps = (0 until nCrossLayers).map { _ =>
    val hiddenDims = mlpParams.getOrElse("dims", List(256L, 128L)) match {
      case d: List[_] => d.asInstanceOf[List[Long]]
      case _ => List(256L, 128L)
    }
    val activation = mlpParams.getOrElse("activation", "relu").toString
    val dropout = mlpParams.getOrElse("dropout", 0.2f).asInstanceOf[Float]
    new MLP(
      inputDim = dims,
      hiddenDims = List(dims, dims),  // Override: always [dims, dims]
      outputDim = dims,
      activation = activation,
      dropout = dropout,
      outputLayer = false,  // No output layer, output dim = dims
      device = device
    )
  }.toList
  mlps.zipWithIndex.foreach { case (mlp, i) =>
    register_module(s"mlp_$i", mlp)
  }

  // Bridge modules (ModuleList)
  private val bridges = (0 until nCrossLayers).map { _ =>
    new BridgeModule(dims, bridgeType, device)
  }.toList
  bridges.zipWithIndex.foreach { case (bridge, i) =>
    register_module(s"bridge_$i", bridge)
  }

  // Regulation modules (ModuleList)
  private val regulationModules = (0 until nCrossLayers).map { _ =>
    new RegulationModule(numFields, feaDims, temperature, useRegulationModule)
  }.toList
  regulationModules.zipWithIndex.foreach { case (reg, i) =>
    register_module(s"regulation_$i", reg)
  }

  // Final linear layer (input dim = dims * 3 for concat of cross, deep, bridge)
  private val finalLinear = new LR(dims * 3, sigmoid = false, device = device)
  register_module("final_linear", finalLinear)

  def forward(sparseFeats: Map[String, Tensor]): Tensor = {
    // Get embeddings
    val embedX = embedding.forward(sparseFeats, squeeze = true)  // (B, dims)

    // Initial regulation on embeddings
    var (crossI, deepI) = regulationModules(0).forward(embedX)
    val cross0 = crossI

    // Multi-layer iteration
    var bridgeI: Tensor = null.asInstanceOf[Tensor]
    for (i <- 0 until nCrossLayers) {
      if (i > 0) {
        // Subsequent layers: regulation takes previous bridge output
        val (regCross, regDeep) = regulationModules(i).forward(bridgeI)
        crossI = regCross
        deepI = regDeep
      }

      // Cross network: cross_i = cross_i + cross_layer(cross_0, cross_i)
      crossI = crossI.add(crossLayers(i).forward(cross0, crossI))

      // Deep network: deep_i = mlp(deep_i)
      deepI = mlps(i).forward(deepI)

      // Bridge: bridge_i = bridge(cross_i, deep_i)
      bridgeI = bridges(i).forward(crossI, deepI)
    }

    // Final concatenation and linear
    val xStack = torch.cat(new TensorVector(crossI, deepI, bridgeI), 1)
    val y = finalLinear.forward(xStack)
    torch.sigmoid(y.squeeze(1))
  }
}

/**
 * Bridge Module for connecting cross and deep networks in EDCN
 * Reference: EDCN paper, KDD 2021
 *
 * Supports 4 bridge types:
 * - hadamard_product: x * h
 * - pointwise_addition: x + h
 * - concatenation: Linear(concat(x, h)) -> input_dim
 * - attention_pooling: softmax(W1x)*x + softmax(W2h)*h
 */
class BridgeModule(
  inputDim: Int,
  bridgeType: String,
  device: String = DeviceSupport.backend
) extends Module {

//  require(
//    List("hadamard_product", "pointwise_addition", "concatenation", "attention_pooling").contains(bridgeType),
//    s"bridgeType '$bridgeType' not supported. Must be one of: hadamard_product, pointwise_addition, concatenation, attention_pooling"
//  )

  val bridgeTypeName: String = bridgeType

  // For concatenation bridge
  private val concatPooling = if (bridgeType == "concatenation") {
    val layer = new SequentialImpl()
    layer.push_back(new LinearImpl(inputDim * 2, inputDim))
    layer.push_back(new ReLUImpl())
    if (device != "cpu") {
      layer.to(new org.bytedeco.pytorch.Device(device), false)
    }
    Some(layer)
  } else None

  // For attention_pooling bridge
  private val attentionPool = if (bridgeType == "attention_pooling") {
    val attX = new SequentialImpl()
    attX.push_back(new LinearImpl(inputDim, inputDim))
    attX.push_back(new ReLUImpl())
    attX.push_back(new LinearImpl(inputDim, inputDim))
    // No sigmoid - softmax is applied separately

    val attH = new SequentialImpl()
    attH.push_back(new LinearImpl(inputDim, inputDim))
    attH.push_back(new ReLUImpl())
    attH.push_back(new LinearImpl(inputDim, inputDim))

    if (device != "cpu") {
      val dev = new org.bytedeco.pytorch.Device(device)
      attX.to(dev, false)
      attH.to(dev, false)
    }
    Some((attX, attH))
  } else None

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    this.to(dev, false)
  }

  def forward(x: Tensor, h: Tensor): Tensor = {
    bridgeType match {
      case "hadamard_product" =>
        x.mul(h)

      case "pointwise_addition" =>
        x.add(h)

      case "concatenation" =>
        val concat = torch.cat(new TensorVector(x, h), 1)
        concatPooling.get.forward(concat)

      case "attention_pooling" =>
        val (attX, attH) = attentionPool.get
        val attXOut = attX.forward(x)  // (B, D)
        val attHOut = attH.forward(h)  // (B, D)
        val weightX = torch.softmax(attXOut, 1)  // (B, D)
        val weightH = torch.softmax(attHOut, 1)  // (B, D)
        weightX.mul(x).add(weightH.mul(h))

      case _ =>
        h  // Fallback
    }
  }
}