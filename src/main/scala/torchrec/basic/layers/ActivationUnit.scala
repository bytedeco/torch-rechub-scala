package torchrec.basic.layers

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
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Activation Unit for DIN-style attention
 * Reference: Alibaba DIN, KDD 2018
 *
 * Computes attention weight between target item and historical items.
 * Input: two item embeddings [batch, embed_dim]
 * Output: attention weight [batch, 1]
 *
 * Formula:
 *   cross = item1 * item2^T  (element-wise product, treated as "cross" feature)
 *   concat = [item1, cross, item2]
 *   h = linear1(concat) -> Dice activation
 *   weight = linear2(h) -> [batch, 1]
 */
class ActivationUnit(
  val embedDim: Int,
  val hiddenSize: Int = 36,
  val activation: String = "dice",
  val device: String = DeviceSupport.backend
) extends Module {

  private val linear1 = new LinearImpl(embedDim * 3, hiddenSize)
  register_module("linear1", linear1)

  // Dice activation for hidden layer
  private val diceAct = new DiceActivation(hiddenSize)
  register_module("diceAct", diceAct)

  private val linear2 = new LinearImpl(hiddenSize, 1)
  register_module("linear2", linear2)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    linear1.to(dev, false)
    linear2.to(dev, false)
  }

  override def forward(item1: Tensor, item2: Tensor): Tensor = {
    // item1: [batch, embed_dim], item2: [batch, embed_dim]
    // Element-wise product as "cross" interaction
    val cross = item1.mul(item2)
    // Concatenate: [batch, embed_dim * 3]
    val concat = torch.cat(new TensorVector(item1, cross, item2), 1)
    // Hidden layer
    val h = linear1.forward(concat)
    val activated = diceAct.forward(h)
    // Output weight
    linear2.forward(activated)
  }
}

/**
 * Generalized Attention layer
 * Reference: Alibaba DIN, KDD 2018
 *
 * Dot-product style attention between query and key-value pairs.
 * Input:
 *   query: [batch, query_dim]
 *   keys: [batch, seq_len, key_dim] (historical items)
 *   values: [batch, seq_len, val_dim]
 * Output: [batch, val_dim] (attention-weighted sum)
 */
class Attention(
  val queryDim: Int,
  val keyDim: Int,
  val valueDim: Int,
  val outputDim: Int,
  val device: String = DeviceSupport.backend
) extends Module {

  private val queryProj = new LinearImpl(queryDim, outputDim)
  private val keyProj = new LinearImpl(keyDim, outputDim)
  private val valueProj = new LinearImpl(valueDim, outputDim)
  register_module("queryProj", queryProj)
  register_module("keyProj", keyProj)
  register_module("valueProj", valueProj)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    queryProj.to(dev, false)
    keyProj.to(dev, false)
    valueProj.to(dev, false)
  }

  override def forward(query: Tensor, keys: Tensor, values: Tensor): Tensor = {
    // query: [batch, query_dim]
    // keys: [batch, seq_len, key_dim]
    // values: [batch, seq_len, val_dim]
    val batchSize = query.size(0)
    val seqLen = keys.size(1)

    // Project query and keys to same dimension
    val q = queryProj.forward(query).unsqueeze(1)  // [batch, 1, output_dim]
    val k = keyProj.forward(keys)                   // [batch, seq_len, output_dim]
    val v = valueProj.forward(values)                // [batch, seq_len, output_dim]

    // Scaled dot-product attention
    val scores = q.mul(k).sum(2).unsqueeze(2)       // [batch, seq_len, 1]
    val scale = new Scalar(scala.math.sqrt(outputDim.toDouble).toFloat)
    val scaledScores = scores.div(scale)
    val attnWeights = scaledScores.softmax(1)       // [batch, seq_len, 1]

    // Weighted sum
    val attended = v.mul(attnWeights).sum(1)         // [batch, output_dim]
    attended
  }
}

/**
 * PRelu (Parametric ReLU) activation
 * Reference: "Delving Deep into Rectifiers" - He et al., 2015
 *
 * Formula: PReLU(x) = max(0, x) + alpha * min(0, x)
 * Where alpha is a learnable parameter per channel.
 *
 * Note: This differs from the old torch-recommend implementation which used
 * relu(x) + alpha * (x - abs(x)) * 0.5. The new project uses PyTorch's
 * standard prelu which is equivalent to relu(x) + alpha * min(0, x).
 */
class PRelu private[torchrec](
  private val weight: Tensor
) extends Module {

  def this(numParameters: Int = 1) = {
    this(torch.zeros(Array[Long](numParameters.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))))
  }

  register_parameter("weight", weight)

  override def forward(x: Tensor): Tensor = {
    torch.prelu(x, weight)
  }
}
