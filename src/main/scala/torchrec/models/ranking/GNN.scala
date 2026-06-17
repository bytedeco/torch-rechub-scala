package torchrec.models.ranking

import torchrec.utils.DeviceSupport
import org.bytedeco.pytorch.Module
import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.mutable

/**
 * Graph Neural Network models for risk control and fraud detection.
 * Includes GCN, GAT, and GraphSAGE implementations.
 */

/**
 * Graph Convolutional Network (GCN)
 * Reference: "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017)
 */
class GCN(
  numFeatures: Int,
  hiddenDim: Int = 64,
  numClasses: Int = 2,
  dropout: Float = 0.5f,
  device: String = DeviceSupport.backend
) extends Module {

  private val gc1 = new GraphConvolution(numFeatures, hiddenDim, device)
  private val gc2 = new GraphConvolution(hiddenDim, numClasses, device)
  private val dropoutLayer = new DropoutImpl(dropout)
  private val activation = new ReLUImpl()

  register_module("gc1", gc1)
  register_module("gc2", gc2)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    gc1.to(dev, false)
    gc2.to(dev, false)
  }

  def forward(
    features: Tensor, // (numNodes, numFeatures)
    adj: Tensor // (numNodes, numNodes) adjacency matrix
  ): Tensor = {
    var x = gc1.forward(features, adj)
    x = activation.forward(x)
    x = dropoutLayer.forward(x)
    x = gc2.forward(x, adj)
    x
  }
}

/**
 * Graph Convolution Layer
 */
class GraphConvolution(
  inFeatures: Int,
  outFeatures: Int,
  device: String = DeviceSupport.backend
) extends Module {

  private val weight = new LinearImpl(inFeatures, outFeatures)
  register_module("weight", weight)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    weight.to(dev, false)
  }

  def forward(input: Tensor, adj: Tensor): Tensor = {
    val support = weight.forward(input)
    val output = torch.matmul(adj, support)
    // Add bias - create on same device as output
    val biasTensor = torch.zeros(Array(outFeatures.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(output.device(), ScalarType.Float)
    output.add(biasTensor)
  }
}

/**
 * Graph Attention Network (GAT)
 * Reference: "Graph Attention Networks" (ICLR 2018)
 */
class GAT(
  numFeatures: Int,
  hiddenDim: Int = 64,
  numClasses: Int = 2,
  numHeads: Int = 8,
  dropout: Float = 0.5f,
  device: String = DeviceSupport.backend
) extends Module {

  private val att1 = new GraphAttentionLayer(numFeatures, hiddenDim, numHeads, dropout, device)
  private val att2 = new GraphAttentionLayer(hiddenDim, numClasses, 1, dropout, device)

  register_module("att1", att1)
  register_module("att2", att2)

  def forward(
    features: Tensor,      // (numNodes, numFeatures)
    adj: Tensor             // (numNodes, numNodes) adjacency matrix
  ): Tensor = {
    val x = att1.forward(features, adj)
    val xActivated = torch.elu(x)  // ELU activation between attention layers
    att2.forward(xActivated, adj)
  }
}

/**
 * Graph Attention Layer
 */
class GraphAttentionLayer(
  inFeatures: Int,
  outFeatures: Int,
  numHeads: Int = 8,
  dropout: Float = 0.5f,
  device: String = DeviceSupport.backend
) extends Module {

  private val headDim = outFeatures / numHeads
  private val heads = (0 until numHeads).map { i =>
    val fc = new LinearImpl(inFeatures, outFeatures)
    register_module(s"fc_$i", fc)
    fc
  }
  private val dropoutLayer = new DropoutImpl(dropout)

  // Attention mechanism parameters
  private val a1 = new LinearImpl(inFeatures, 1)
  private val a2 = new LinearImpl(inFeatures, 1)
  register_module("a1_attention", a1)
  register_module("a2_attention", a2)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    heads.foreach(_.to(dev, false))
    a1.to(dev, false)
    a2.to(dev, false)
  }

  def forward(input: Tensor, adj: Tensor): Tensor = {
    val batchSize = input.size(0)
    val headOutputs = heads.map { fc =>
      val h = fc.forward(input)  // (numNodes, outFeatures)
      selfAttention(h, adj)
    }

    // Concatenate heads
    val concatenated = torch.cat(new TensorVector(headOutputs.map(_.contiguous().view(batchSize, -1)): _*), 1)
    dropoutLayer.forward(concatenated)
  }

  private def selfAttention(embeddings: Tensor, adj: Tensor): Tensor = {
    val numNodes = embeddings.size(0)

    // Compute attention coefficients
    val e = a1.forward(embeddings).add(a2.forward(embeddings)).squeeze(2)  // (numNodes,)

    // Masked attention - use adjacency mask with -inf for non-edges
    val negInf = new Scalar(-1e9f)
    val negInfTensor = torch.full(Array(numNodes), negInf, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(embeddings.device(), ScalarType.Float)
    val attentionScores = torch.where(adj.gt(new Scalar(0.5f)), e, negInfTensor)

    val attentionWeights = torch.softmax(attentionScores, 1)

    torch.matmul(attentionWeights.unsqueeze(1), embeddings).squeeze(1)
  }
}

/**
 * GraphSAGE: Graph SAmpling and Aggregation
 * Reference: "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
 */
class GraphSAGE(
  numFeatures: Int,
  hiddenDim: Int = 64,
  numClasses: Int = 2,
  aggregator: String = "mean",  // "mean", "pool", "lstm"
  dropout: Float = 0.5f,
  device: String = DeviceSupport.backend
) extends Module {

  private val agg1 = aggregator match {
    case "mean" => new SAGEAggregator(numFeatures, hiddenDim, device)
    case "pool" => new SAGEAggregator(numFeatures, hiddenDim, device)  // Use same for now
    case "lstm" => new SAGEAggregator(numFeatures, hiddenDim, device)
    case _ => new SAGEAggregator(numFeatures, hiddenDim, device)
  }

  private val agg2 = aggregator match {
    case "mean" => new SAGEAggregator(hiddenDim, numClasses, device)
    case "pool" => new SAGEAggregator(hiddenDim, numClasses, device)
    case "lstm" => new SAGEAggregator(hiddenDim, numClasses, device)
    case _ => new SAGEAggregator(hiddenDim, numClasses, device)
  }

  register_module("agg1", agg1)
  register_module("agg2", agg2)

  private val dropoutLayer = new DropoutImpl(dropout)
  private val activation = new ReLUImpl()

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    agg1.to(dev, false)
    agg2.to(dev, false)
  }

  def forward(
    features: Tensor,      // (numNodes, numFeatures)
    adj: Tensor             // (numNodes, numNodes) adjacency matrix
  ): Tensor = {
    var x = agg1.forward(features, adj)
    x = activation.forward(x)
    x = dropoutLayer.forward(x)
    x = agg2.forward(x, adj)
    x
  }
}

/**
 * SAGE Aggregator
 */
class SAGEAggregator(
  inFeatures: Int,
  outFeatures: Int,
  device: String = DeviceSupport.backend
) extends Module {

  private val weight = new LinearImpl(inFeatures, outFeatures)
  register_module("weight", weight)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    weight.to(dev, false)
  }

  def forward(input: Tensor, adj: Tensor): Tensor = {
    val numNodes = input.size(0)

    // Aggregate neighbor features
    val degree = adj.sum(1).unsqueeze(1)  // (numNodes, 1)
    val neighborSum = torch.matmul(adj, input)  // (numNodes, inFeatures)

    // Mean aggregation
    val epsilon = new Scalar(1e-9f)
    val aggregated = neighborSum.div(degree.add(epsilon))  // Avoid division by zero

    // Combine with self
    val combined = aggregated.add(input)  // Residual connection
    weight.forward(combined)
  }
}

/**
 * FraudGNN: GNN for Fraud Detection
 * Combines multiple GNN layers with batch normalization for fraud detection
 */
class FraudGNN(
  numFeatures: Int,
  hiddenDim: Int = 128,
  numClasses: Int = 2,
  numLayers: Int = 3,
  dropout: Float = 0.3f,
  device: String = DeviceSupport.backend
) extends Module {

  private val layers = (0 until numLayers).map { i =>
    val inDim = if (i == 0) numFeatures else hiddenDim
    val outDim = if (i == numLayers - 1) numClasses else hiddenDim
    val layer = new GraphConvolution(inDim, outDim, device)
    register_module(s"gc_$i", layer)
    layer
  }

  private val dropoutLayer = new DropoutImpl(dropout)
  private val activation = new ReLUImpl()

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    layers.foreach(_.to(dev, false))
  }

  def forward(
    features: Tensor,      // (numNodes, numFeatures)
    adj: Tensor             // (numNodes, numNodes) adjacency matrix
  ): Tensor = {
    var x = features
    layers.dropRight(1).foreach { layer =>
      x = layer.forward(x, adj)
      x = activation.forward(x)
      x = dropoutLayer.forward(x)
    }
    layers.last.forward(x, adj)
  }
}