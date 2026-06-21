package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.basic.layers.MLP
import torchrec.models.knowledge_tracing.layers.CosinePositionalEmbedding
import torchrec.utils.DeviceSupport

/**
 * StableKT: Stable Knowledge Tracing with ALiBi and Penumbral Attention
 */
class StableKT(
  numConcepts: Long,
  embedDim: Int = 64,
  numHeads: Int = 8,
  numBlocks: Int = 2,
  r: Float = 1.0f,
  gamma: Float = 1.0f,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)

  private val qEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim))
  register_module("q_embed", qEmbed)

  private val qaEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2 + 2, embedDim))
  register_module("qa_embed", qaEmbed)

  private val posEmb = new CosinePositionalEmbedding(embedDim, 512, device)
  register_module("pos_emb", posEmb)

  private val blocks = (0 until numBlocks).map { i =>
    val block = new StableTransformerBlock(embedDim, numHeads, r, gamma, dropout, device)
    register_module(s"block_$i", block)
    block
  }

  private val outMLP = new MLP(embedDim * 2, List(embedDim.toLong, embedDim / 2), 1, "relu", dropout, device = device)
  register_module("out_mlp", outMLP)

  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropout", dropoutLayer)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    qEmbed.to(dev, false)
    qaEmbed.to(dev, false)
    outMLP.to(dev, false)
  }

  def forward(
    conceptIds: Tensor,
    responses: Tensor
  ): Tensor = {
    val batchSize = conceptIds.size(0).toInt
    val seqLen = conceptIds.size(1).toInt

    // Convert to Long and clamp
    val cLong = conceptIds.toType(ScalarType.Long)
    val rLong = responses.toType(ScalarType.Long)

    val conceptIdx = cLong.clamp(
      new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(0)),
      new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(numConcepts.toDouble))
    ).toType(ScalarType.Long)
    val responseIdx = rLong.clamp(
      new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(0)),
      new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(1))
    ).toType(ScalarType.Long)

    // QA interaction: concept * 2 + response
    val qaIds = conceptIdx.mul(new Scalar(2)).add(responseIdx)
    val qaEmb = qaEmbed.forward(qaIds)
    val qEmb = qEmbed.forward(conceptIdx)

    // Add positional encoding
    val posEnc = posEmb.forward(qaEmb)
    val qaWithPos = qaEmb.add(posEnc)
    val qWithPos = qEmb.add(posEnc)

    // Process through transformer blocks
    var y = qaWithPos
    var x = qWithPos
    blocks.foreach { block =>
      val result = block.forward(x, y)
      x = result._1
      y = result._2
    }

    // Concatenate for output
    val concatQ = torch.cat(new TensorVector(x, qEmb), 2)
    val logits = outMLP.forward(concatQ)

    logits.sigmoid().squeeze(2)
  }

  def predict(conceptIds: Tensor, responses: Tensor): Tensor = forward(conceptIds, responses)
}

class StableTransformerBlock(
  embedDim: Int,
  numHeads: Int,
  r: Float = 1.0f,
  gamma: Float = 1.0f,
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)
  private val headDim = embedDim / numHeads
  private val halfHeads = numHeads / 2

  private val qLinear = new LinearImpl(embedDim, embedDim)
  private val kLinear = new LinearImpl(embedDim, embedDim)
  private val vLinear = new LinearImpl(embedDim, embedDim)
  register_module("q_linear", qLinear)
  register_module("k_linear", kLinear)
  register_module("v_linear", vLinear)

  private val outLinear = new LinearImpl(embedDim, embedDim)
  register_module("out_linear", outLinear)

  private def layerNormShape(d: Int) = { val v = new LongVector(1); v.put(0, d.toLong); v }
  private val ln1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
  private val ln2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
  register_module("ln1", ln1)
  register_module("ln2", ln2)

  private val ff1 = new LinearImpl(embedDim, embedDim * 4)
  private val ff2 = new LinearImpl(embedDim * 4, embedDim)
  register_module("ff1", ff1)
  register_module("ff2", ff2)

  private val dropoutLayer = new DropoutImpl(dropout)

  // ALiBi slopes
  private val slopes = {
    val s = torch.zeros(Array(numHeads.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    for (h <- 0 until numHeads) {
      val slope = math.pow(2, -8.0 * (h + 1) / numHeads).toFloat
      s.select(0, h).fill_(new Scalar((-slope).toDouble))
    }
    register_parameter("slopes", s)
    s
  }

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    qLinear.to(dev, false); kLinear.to(dev, false); vLinear.to(dev, false)
    outLinear.to(dev, false); ff1.to(dev, false); ff2.to(dev, false)
  }

  def forward(x: Tensor, y: Tensor): (Tensor, Tensor) = {
    val batchSize = x.size(0).toInt
    val seqLen = x.size(1).toInt

    val q = qLinear.forward(x).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)
    val k = kLinear.forward(y).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)
    val v = vLinear.forward(y).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)

    // Standard attention for first half
    val scaleVal = math.sqrt(headDim.toDouble).toFloat
    val scale = new Scalar(scaleVal)
    val stdScores = torch.matmul(q.narrow(1, 0, halfHeads), k.narrow(1, 0, halfHeads).transpose(2, 3)).div(scale)

    // Penumbral attention for second half
    val penumbralScores = penumbralAttention(
      q.narrow(1, halfHeads, halfHeads),
      k.narrow(1, halfHeads, halfHeads),
      r, gamma
    )

    // Combine
    val scores = torch.cat(new TensorVector(stdScores, penumbralScores), 1)

    // ALiBi bias
    val alibiBias = computeAliBi(seqLen, numHeads)
    val biasedScores = scores.add(alibiBias)

    // Causal mask
    if (seqLen > 1) {
      val causalMask = torch.triu(torch.ones(Array(seqLen.toLong, seqLen.toLong),
        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))), 1)
        .mul(new Scalar(-1e9))
        .unsqueeze(0).unsqueeze(0)
      val maskedScores = biasedScores.add(causalMask)
      val attnWeights = maskedScores.softmax(-1)
      val attended = torch.matmul(dropoutLayer.forward(attnWeights), v)

      val reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim)
      val outProj = outLinear.forward(reshaped)

      val withRes = x.add(outProj)
      val normed1 = ln1.forward(withRes)

      val ffOut = torch.relu(ff1.forward(normed1))
      val ffDropped = dropoutLayer.forward(ffOut)
      val ffResult = ff2.forward(ffDropped)

      val finalOut = ln2.forward(normed1.add(ffResult))
      (finalOut, y)
    } else {
      val attnWeights = biasedScores.softmax(-1)
      val attended = torch.matmul(dropoutLayer.forward(attnWeights), v)

      val reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim)
      val outProj = outLinear.forward(reshaped)

      val withRes = x.add(outProj)
      val normed1 = ln1.forward(withRes)

      val ffOut = torch.relu(ff1.forward(normed1))
      val ffDropped = dropoutLayer.forward(ffOut)
      val ffResult = ff2.forward(ffDropped)

      val finalOut = ln2.forward(normed1.add(ffResult))
      (finalOut, y)
    }
  }

  private def computeAliBi(seqLen: Int, numHeads: Int): Tensor = {
    // Build ALiBi bias tensor by constructing a float array and reshaping.
    val total = 1 * numHeads * seqLen * seqLen
    val flat = new Array[Float](total)
    for (h <- 0 until numHeads) {
      val slope = slopes.select(0, h).item().toFloat
      for (i <- 0 until seqLen) {
        for (j <- 0 until seqLen) {
          val distance = j - i
          val bias = slope * distance
          val idx = ((0 * numHeads + h) * seqLen + i) * seqLen + j
          flat(idx) = bias.toFloat
        }
      }
    }
    val t = torch.tensor(flat, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    t.reshape(1L, numHeads.toLong, seqLen.toLong, seqLen.toLong)
  }

  private def penumbralAttention(
    q: Tensor, k: Tensor,
    r: Float, gamma: Float
  ): Tensor = {
    val batchSize = q.size(0).toInt
    val numHeads = q.size(1).toInt
    val seqLen = q.size(2).toInt
    val headDim = q.size(3).toInt

    val qExpand = q.unsqueeze(3)
    val kExpand = k.unsqueeze(2)
    val diff = qExpand.sub(kExpand)
    val pairwiseDist = torch.norm(diff, new ScalarOptional(new Scalar(2)), -1)

    val penumbralScores = torch.exp(pairwiseDist.mul(new Scalar((-gamma).toDouble)))

    val scaleVal = math.sqrt(headDim.toDouble).toFloat
    penumbralScores.div(new Scalar(scaleVal.toDouble))
  }
}
