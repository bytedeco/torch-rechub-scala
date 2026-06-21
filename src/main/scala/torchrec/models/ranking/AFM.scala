package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Attentional Factorization Machine (AFM)
 *
 * Reference: "Attentional Factorization Machines: Learning the Weight of Feature
 * Interactions via Attention Networks" - Zhejiang University, IJCAI 2017
 *
 * Python原版对照实现，核心差异：
 * 1. 有linear部分处理一阶特征 (y_linear)
 * 2. FM部分使用reduce_sum=False得到(batch, embed_dim)的pairwise interaction
 * 3. Attention网络: attention_liner -> relu -> h -> softmax
 * 4. 最终输出: y_linear + attention_weighted * p
 *
 * @param features       Sparse feature list for FM interaction
 * @param embedDim       Embedding dimension
 * @param attentionDim   Attention hidden dimension (t in paper)
 * @param dropout        Dropout rate
 * @param device         Device to run on
 */
class AFM(
  features: List[SparseFeature],
  embedDim: Int = 8,
  attentionDim: Int = 64,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")

  val numFields = features.size
  require(numFields >= 2, "AFM requires at least 2 sparse features for interaction")

  // Linear part (first-order): LR on flattened embeddings
  private val fmDims = features.map(_.embedDim).sum
  private val linear = new LR(fmDims, sigmoid = false, device = device)
  register_module("linear", linear)

  // FM part (second-order): FM with reduce_sum=False to get (batch, embed_dim)
  private val fm = new FMInteraction(embedDim)
  register_module("fm", fm)

  // Embedding layer
  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  // Attention network: input=embed_dim, hidden=attentionDim, output=1
  // 论文公式: attention_liner
  private val attentionLiner = new LinearImpl(embedDim, attentionDim)
  register_module("attention_liner", attentionLiner)

  // 论文公式中的h: (attentionDim, 1) - 直接使用 torch.tensor 创建并注册为 parameter
  private val h = {
    val std = math.sqrt(6.0 / (attentionDim + 1)).toFloat
    val random = new java.util.Random(42)
    val arr = Array.fill(attentionDim * 1) { (random.nextFloat() * 2 - 1) * std }
    val t = torch.tensor(arr*).view(Array(attentionDim.toLong, 1L)*).to(ScalarType.Float)
    register_parameter("h", t)
    t
  }

  // 论文公式中的p: (embed_dim, 1) - 直接使用 torch.tensor 创建并注册为 parameter
  private val p = {
    val std = math.sqrt(6.0 / (embedDim + 1)).toFloat
    val random = new java.util.Random(42)
    val arr = Array.fill(embedDim * 1) { (random.nextFloat() * 2 - 1) * std }
    val t = torch.tensor(arr*).view(Array(embedDim.toLong, 1L)*).to(ScalarType.Float)
    register_parameter("p", t)
    t
  }

  // Dropout for attention output
  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropout", dropoutLayer)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    this.to(dev, false)
  }

  /**
   * Compute attention scores
   * @param yFm FM output of shape (batch, embed_dim)
   * @return attention scores of shape (batch, 1)
   */
  private def attention(yFm: Tensor): Tensor = {
    // yFm: (batch, embed_dim)
    // attention_liner(yFm): (batch, attentionDim)
    val yAtt = attentionLiner.forward(yFm)
    // relu
    val yRelu = torch.relu(yAtt)
    // matmul(h): (batch, 1)
    val yMatmul = torch.matmul(yRelu, h.to(yFm.device(), ScalarType.Float))
    // softmax(dim=1): (batch, 1)
    val atts = torch.softmax(yMatmul, 1)
    atts
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Get embeddings: (batch, num_fields, embed_dim)
    val embeddings = embeddingLayer.forward3D(sparseFeats, sequenceFeats = Map.empty)
    val batchSize = embeddings.size(0).toInt

    // Linear part (first-order): flatten and apply LR
    val embeddingsFlat = embeddings.view(batchSize, numFields * embedDim)
    val yLinear = linear.forward(embeddingsFlat)  // (batch, 1)

    // FM part (second-order): (batch, embed_dim)
    val yFm = fm.forward(embeddings)  // FMInteraction returns (batch, embed_dim)

    // Attention: (batch, 1)
    val atts = attention(yFm)

    // Apply dropout to attention
    val attsDrop = dropoutLayer.forward(atts)

    // outs = atts * y_fm @ p: (batch, 1)
    val weighted = attsDrop.mul(yFm)  // (batch, embed_dim)
    val outs = torch.matmul(weighted, p.to(yFm.device(), ScalarType.Float))  // (batch, 1)

    // Final output: y_linear + outs - return logits (BCEWithLogitsLoss applies sigmoid)
    yLinear.add(outs)
  }
}
