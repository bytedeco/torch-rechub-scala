package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._
import torchrec.utils.DeviceSupport
import torchrec.TorchRec

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.mutable

/**
 * XGBoost风格模型 - 简化版
 */
class XGBoostModel(
                    features: List[Feature],
                    numTrees: Int = 64,
                    treeDepth: Int = 6,
                    embedDim: Int = 8,
                    linkFeatDim: Long = 128L,
                    device: String = DeviceSupport.backend
                  ) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(treeDepth >= 1, s"treeDepth must be >= 1, got $treeDepth")
  require(linkFeatDim > 0, s"linkFeatDim must be > 0, got $linkFeatDim")

  private val numLeaves = 1 << treeDepth
  private val targetDevice = new Device(device)

  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  private val trees = (0 until numTrees).map { i =>
    val tree = new SoftDecisionTree(linkFeatDim, treeDepth, numLeaves, device)
    register_module(s"tree_$i", tree)
    tree
  }

  def forward(
               sparseFeats: Map[String, Tensor],
               denseFeats: Map[String, Tensor] = Map.empty
             ): Tensor = {
    val batchSize = sparseFeats.values.head.size(0).toInt

    // 嵌入
    val sparseFeatureNames = features.collect { case f: SparseFeature => f.name }.toSet
    val validSparseFeats = sparseFeats.filter { case (k, _) => sparseFeatureNames.contains(k) }
    val embeddings = embeddingLayer.forward(validSparseFeats, Map.empty, squeeze = false)
    val sparseFlat = embeddings.view(batchSize.toLong, -1)

    // 合并稠密特征
    val input = if (denseFeats.nonEmpty) {
      val denseSeq = denseFeats.values.toSeq
      val denseCat = if (denseSeq.size == 1) denseSeq.head else {
        val vec = new TensorVector(denseSeq.size.toLong)
        denseSeq.foreach(vec.push_back)
        torch.cat(vec, 1L)
      }
      val vec = new TensorVector(2L)
      vec.push_back(sparseFlat)
      vec.push_back(denseCat)
      torch.cat(vec, 1L)
    } else sparseFlat

    // 维度对齐
    val featDim = input.size(1).toInt
    val alignedInput = if (featDim < linkFeatDim) {
      val pad = torch.zeros(Array(batchSize.toLong, linkFeatDim - featDim),
        new TensorOptions().device(new DeviceOptional(targetDevice)).dtype(new ScalarTypeOptional(ScalarType.Float)))
      val vec = new TensorVector(2L)
      vec.push_back(input)
      vec.push_back(pad)
      torch.cat(vec, 1L)
    } else if (featDim > linkFeatDim) {
      input.narrow(1, 0, linkFeatDim)
    } else input

    // 树输出求平均
    val treeOutputs = trees.map(_.forward(alignedInput))
    val vec = new TensorVector()
    treeOutputs.foreach(vec.push_back)
    val stacked = torch.stack(vec, 0L)
    val result = stacked.mean(0L)

    result
  }
}

/**
 * 软决策树 - 最简化版
 */
class SoftDecisionTree(
                        inputDim: Long,
                        depth: Int,
                        numLeaves: Int,
                        device: String = DeviceSupport.backend
                      ) extends Module {

  private val numInternalNodes = numLeaves - 1
  private val targetDevice = new Device(device)

  // 路由MLP
  private val routeMLP = new LinearImpl(inputDim, numInternalNodes)
  routeMLP.to(targetDevice, false)
  register_module("route_mlp", routeMLP)

  // 叶子节点参数（正确注册，不参与前向，避免维度污染）
  private val leafValues = torch.zeros(Array(numLeaves.toLong),
    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  leafValues.to(targetDevice, ScalarType.Float)
  register_parameter("leaf_values", leafValues)

  def forward(x: Tensor): Tensor = {
    // Only transfer if not on target device
    val xOnDev = if (!x.device().equals(targetDevice)) {
      x.to(targetDevice, ScalarType.Float)
    } else x

    // 路由逻辑（纯全连接+平均，完全稳定输出形状 [batch, 1]）
    val routeLogits = routeMLP.forward(xOnDev)
    val avgLogit = routeLogits.mean(Array(1l), true, new ScalarTypeOptional())

    // 确保输出形状永远是 [batch, 1]
    avgLogit.unsqueeze(1).squeeze(2)
  }
}

//package torchrec.models.ranking
//
//import torchrec.basic.features._
//import torchrec.basic.layers._
//import torchrec.Implicits._
//import torchrec.utils.DeviceSupport
//import torchrec.TorchRec
//
//import org.bytedeco.pytorch._
//import org.bytedeco.pytorch.global.torch
//import org.bytedeco.pytorch.global.torch.ScalarType
//
//import scala.collection.mutable
//
///**
// * XGBoost风格模型 - 简化版
// */
//class XGBoostModel(
//  features: List[Feature],
//  numTrees: Int = 64,
//  treeDepth: Int = 6,
//  embedDim: Int = 8,
//  linkFeatDim: Long = 128L,
//  device: String = DeviceSupport.backend
//) extends Module {
//
//  require(features.nonEmpty, "features cannot be empty")
//  require(treeDepth >= 1, s"treeDepth must be >= 1, got $treeDepth")
//  require(linkFeatDim > 0, s"linkFeatDim must be > 0, got $linkFeatDim")
//
//  private val numLeaves = 1 << treeDepth
//
//  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
//  register_module("embedding", embeddingLayer)
//
//  private val trees = (0 until numTrees).map { i =>
//    val tree = new SoftDecisionTree(linkFeatDim, treeDepth, numLeaves, device)
//    register_module(s"tree_$i", tree)
//    tree
//  }
//
//  def forward(
//    sparseFeats: Map[String, Tensor],
//    denseFeats: Map[String, Tensor] = Map.empty
//  ): Tensor = {
//    val batchSize = sparseFeats.values.head.size(0).toInt
//
//    // 嵌入
//    val sparseFeatureNames = features.collect { case f: SparseFeature => f.name }.toSet
//    val validSparseFeats = sparseFeats.filter { case (k, _) => sparseFeatureNames.contains(k) }
//    val embeddings = embeddingLayer.forward(validSparseFeats, Map.empty, squeeze = false)
//    val sparseFlat = embeddings.view(batchSize.toLong, -1)
//
//    // 合并稠密特征
//    val input = if (denseFeats.nonEmpty) {
//      val denseSeq = denseFeats.values.toSeq
//      val denseCat = if (denseSeq.size == 1) denseSeq.head else {
//        val vec = new TensorVector(denseSeq.size.toLong)
//        denseSeq.foreach(vec.push_back)
//        torch.cat(vec, 1L)
//      }
//      val vec = new TensorVector(2L)
//      vec.push_back(sparseFlat)
//      vec.push_back(denseCat)
//      torch.cat(vec, 1L)
//    } else sparseFlat
//
//    // 维度对齐
//    val featDim = input.size(1).toInt
//    val alignedInput = if (featDim < linkFeatDim) {
//      val pad = TorchRec.zeros(batchSize.toLong, linkFeatDim - featDim)
//      val vec = new TensorVector(2L)
//      vec.push_back(input)
//      vec.push_back(pad)
//      torch.cat(vec, 1L)
//    } else if (featDim > linkFeatDim) {
//      input.narrow(1, 0, linkFeatDim)
//    } else input
//
//    // 树输出求平均
//    val treeOutputs = trees.map(_.forward(alignedInput))
//    val vec = new TensorVector(treeOutputs.size.toLong)
//    treeOutputs.foreach(vec.push_back)
//    val stacked = torch.stack(vec, 0L)
//    val result = stacked.mean(0L)
//
//    embeddings.close()
//    sparseFlat.close()
//    alignedInput.close()
//    stacked.close()
//    treeOutputs.foreach(_.close())
//
//    result
//  }
//}
//
///**
// * 软决策树 - 最简化版
// */
//class SoftDecisionTree(
//  inputDim: Long,
//  depth: Int,
//  numLeaves: Int,
//  device: String = DeviceSupport.backend
//) extends Module {
//
//  private val numInternalNodes = numLeaves - 1
//
//  // 路由MLP
//  private val routeMLP = new LinearImpl(inputDim, numInternalNodes)
//  routeMLP.to(new Device(device),false)
//  register_module("route_mlp", routeMLP)
//
//  // 叶子值 - 简化为直接用零
//  private val leafValues = torch.zeros(Array(numLeaves.toLong),
//    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
//  leafValues.to(new Device(device),ScalarType.Float)
//  register_parameter("leaf_values", leafValues)
//
//  def forward(x: Tensor): Tensor = {
//    val batchSize = x.size(0).toInt
//
//    // 确保输入在正确设备上
//    val xDev = x.device()
//
//    // 确保参数在同一设备
//    if (!leafValues.device().equals(xDev)) {
//      leafValues.to(xDev, ScalarType.Float)
//    }
//
//    // 路由分数 -> 直接作为输出
//    val routeLogits = routeMLP.forward(x)
//
//    // 简化：直接用路由logits的平均作为输出
//    val avgLogit = routeLogits.mean(1L)
//
//    routeLogits.close()
//
//    avgLogit.unsqueeze(1)
//  }
//}