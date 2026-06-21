package torchrec.models.knowledge_tracing.layers

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.utils.DeviceSupport

/**
 * Multi-head attention with learnable distance bias.
 * Used by AKT, SimpleKT, CSKT.
 */
class DistanceBiasMultiHeadAttention(
  embedDim: Int,
  numHeads: Int,
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0, "embedDim must be divisible by numHeads")
  private val headDim = embedDim / numHeads

  private val query = new LinearImpl(embedDim, embedDim)
  private val key = new LinearImpl(embedDim, embedDim)
  private val value = new LinearImpl(embedDim, embedDim)
  private val output = new LinearImpl(embedDim, embedDim)
  private val dropoutLayer = new DropoutImpl(dropout)

  register_module("query", query)
  register_module("key", key)
  register_module("value", value)
  register_module("output", output)

  // Learnable gamma parameter for distance decay
  private val gammaInit = torch.zeros(Array(1L, 1L, 1L),
    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  gammaInit.fill_(new Scalar(0.9f))
  private val gamma = gammaInit
  register_parameter("gamma", gamma)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    query.to(dev, false); key.to(dev, false); value.to(dev, false)
    output.to(dev, false)
  }

  def forward(x: Tensor, mask: Int = 1): Tensor = {
    val batchSize = x.size(0).toInt
    val seqLen = x.size(1).toInt

    val q = query.forward(x).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)
    val k = key.forward(x).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)
    val v = value.forward(x).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)

    // Scaled dot-product attention
    val scale = new Scalar(scala.math.sqrt(headDim.toDouble).toFloat)
    var scores = torch.matmul(q, k.transpose(2, 3)).div(scale)

    // Apply distance bias: gamma^{|i-j|}
    if (seqLen > 1) {
      val g = gamma.sigmoid()
      val posSeq = torch.arange(new Scalar(0), new Scalar(seqLen.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      if (device != "cpu") {
        posSeq.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
      }
      val posIds = posSeq.view(seqLen.toLong, 1L)
      val posIdsT = posSeq.view(1L, seqLen.toLong)
      val distMat = (posIds.sub(posIdsT)).abs().toType(ScalarType.Float)
      val distBias = torch.pow(g, distMat.view(1L, 1L, seqLen.toLong, seqLen.toLong))
      scores = scores.add(distBias)

      if (mask == 0) {
        // Causal mask: mask out future positions
        val causalMask = torch.triu(torch.ones(Array(seqLen.toLong, seqLen.toLong),
          new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))), 1).unsqueeze(0).unsqueeze(0)
        scores = scores.add(causalMask.mul(new Scalar(1e9)))
      }
    }

    val attnWeights = scores.softmax(-1)
    val attended = torch.matmul(dropoutLayer.forward(attnWeights), v)

    // Reshape back
    val reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim)
    output.forward(reshaped)
  }
}

/**
 * Transformer layer with distance bias attention.
 */
class TransformerLayer(
  embedDim: Int,
  numHeads: Int,
  ffnDim: Int = 256,
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  private val attention = new DistanceBiasMultiHeadAttention(embedDim, numHeads, dropout, device)
  // Helper to create normalized_shape LongVector: [d]
  private def layerNormShape(d: Int) = { val v = new LongVector(1); v.put(0, d.toLong); v }
  // LayerNorm with normalized_shape as last dimension
  private val norm1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
  private val norm2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
  private val linear1 = new LinearImpl(embedDim, ffnDim)
  private val linear2 = new LinearImpl(ffnDim, embedDim)
  private val dropoutLayer = new DropoutImpl(dropout)

  register_module("attention", attention)
  register_module("norm1", norm1)
  register_module("norm2", norm2)
  register_module("linear1", linear1)
  register_module("linear2", linear2)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    linear2.to(dev, false)
  }

  def forward(x: Tensor, mask: Int = 1): Tensor = {
    val attended = attention.forward(x, mask)
    val withResidual1 = x.add(dropoutLayer.forward(attended))
    val normed1 = norm1.forward(withResidual1)

    val ffnOut = dropoutLayer.forward(linear2.forward(torch.relu(linear1.forward(normed1))))
    val withResidual2 = normed1.add(ffnOut)
    norm2.forward(withResidual2)
  }
}

/**
 * Standard Multi-head self-attention.
 */
class MultiHeadAttention(
  embedDim: Int,
  numHeads: Int,
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)
  private val headDim = embedDim / numHeads

  private val qLinear = new LinearImpl(embedDim, embedDim)
  private val kLinear = new LinearImpl(embedDim, embedDim)
  private val vLinear = new LinearImpl(embedDim, embedDim)
  private val outLinear = new LinearImpl(embedDim, embedDim)
  private val dropoutLayer = new DropoutImpl(dropout)

  register_module("q_linear", qLinear)
  register_module("k_linear", kLinear)
  register_module("v_linear", vLinear)
  register_module("out_linear", outLinear)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    qLinear.to(dev, false); kLinear.to(dev, false)
    vLinear.to(dev, false); outLinear.to(dev, false)
  }

  def forward(q: Tensor, k: Tensor, v: Tensor, mask: Tensor = torch.empty()): Tensor = {
    val batchSize = q.size(0).toInt
    val seqLen = q.size(1).toInt

    val qProj = qLinear.forward(q).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)
    val kProj = kLinear.forward(k).view(batchSize, k.size(1).toInt, numHeads, headDim).transpose(1, 2)
    val vProj = vLinear.forward(v).view(batchSize, v.size(1).toInt, numHeads, headDim).transpose(1, 2)

    val scale = new Scalar(scala.math.sqrt(headDim.toDouble).toFloat)
    var scores = torch.matmul(qProj, kProj.transpose(2, 3)).div(scale)

    if (mask != null && mask.numel() > 0) {
      scores = scores.add(mask)
    }

    val attnWeights = scores.softmax(-1)
    val attended = torch.matmul(dropoutLayer.forward(attnWeights), vProj)
    val reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim)
    outLinear.forward(reshaped)
  }
}

/**
 * Encoder block for SAINT.
 */
class SAINTEncoderBlock(
  embedDim: Int,
  numHeads: Int,
  ffnDim: Int = 256,
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  private val multiEn = new MultiHeadAttention(embedDim, numHeads, dropout, device)
  private val ffnEn1 = new LinearImpl(embedDim, ffnDim)
  private val ffnEn2 = new LinearImpl(ffnDim, embedDim)
  private def layerNormShape(d: Int) = { val v = new LongVector(1); v.put(0, d.toLong); v }
  private val ln1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
  private val ln2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
  private val dropoutLayer = new DropoutImpl(dropout)

  register_module("multi_en", multiEn)
  register_module("ffn_en1", ffnEn1)
  register_module("ffn_en2", ffnEn2)
  register_module("ln1", ln1)
  register_module("ln2", ln2)

  def forward(inEx: Tensor, inCat: Tensor, inPos: Tensor): Tensor = {
    // Combine exercise, category, and position embeddings
    val combined = inEx.add(inCat).add(inPos)
    val attended = multiEn.forward(combined, combined, combined)
    val withResidual1 = combined.add(dropoutLayer.forward(attended))
    val normed1 = ln1.forward(withResidual1)

    val ffnOut = dropoutLayer.forward(ffnEn2.forward(torch.relu(ffnEn1.forward(normed1))))
    ln2.forward(normed1.add(ffnOut))
  }
}

/**
 * Decoder block for SAINT.
 */
class SAINTDecoderBlock(
  embedDim: Int,
  numHeads: Int,
  ffnDim: Int = 256,
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  // Cross attention: attend to encoder output
  private val multiDe1 = new MultiHeadAttention(embedDim, numHeads, dropout, device)
  // Self attention
  private val multiDe2 = new MultiHeadAttention(embedDim, numHeads, dropout, device)
  private val ffnDe1 = new LinearImpl(embedDim, ffnDim)
  private val ffnDe2 = new LinearImpl(ffnDim, embedDim)
  private def layerNormShape(d: Int) = { val v = new LongVector(1); v.put(0, d.toLong); v }
  private val ln1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
  private val ln2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
  private val ln3 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)))
  private val dropoutLayer = new DropoutImpl(dropout)

  register_module("multi_de1", multiDe1)
  register_module("multi_de2", multiDe2)
  register_module("ffn_de1", ffnDe1)
  register_module("ffn_de2", ffnDe2)
  register_module("ln1", ln1)
  register_module("ln2", ln2)
  register_module("ln3", ln3)

  def forward(inRes: Tensor, inPos: Tensor, enOut: Tensor): Tensor = {
    val combined = inRes.add(inPos)

    // Cross attention on encoder output
    val crossAttn = multiDe1.forward(combined, enOut, enOut)
    val withResidual1 = combined.add(dropoutLayer.forward(crossAttn))
    val normed1 = ln1.forward(withResidual1)

    // Self attention
    val selfAttn = multiDe2.forward(normed1, normed1, normed1)
    val withResidual2 = normed1.add(dropoutLayer.forward(selfAttn))
    val normed2 = ln2.forward(withResidual2)

    // FFN
    val ffnOut = dropoutLayer.forward(ffnDe2.forward(torch.relu(ffnDe1.forward(normed2))))
    ln3.forward(normed2.add(ffnOut))
  }
}

/**
 * Cone-shaped attention for CSKT.
 * Uses geometric distance penalty in attention scores.
 */
class ConeAttention(
  embedDim: Int,
  numHeads: Int,
  r: Float = 1.0f,
  gamma: Float = 1.0f,
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)
  private val headDim = embedDim / numHeads

  private val qLinear = new LinearImpl(embedDim, embedDim)
  private val kLinear = new LinearImpl(embedDim, embedDim)
  private val vLinear = new LinearImpl(embedDim, embedDim)
  private val outLinear = new LinearImpl(embedDim, embedDim)
  private val dropoutLayer = new DropoutImpl(dropout)

  register_module("q_linear", qLinear)
  register_module("k_linear", kLinear)
  register_module("v_linear", vLinear)
  register_module("out_linear", outLinear)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    qLinear.to(dev, false); kLinear.to(dev, false)
    vLinear.to(dev, false); outLinear.to(dev, false)
  }

  def forward(q: Tensor, k: Tensor, mask: Int = 0): Tensor = {
    val batchSize = q.size(0).toInt
    val seqLen = q.size(1).toInt

    val qProj = qLinear.forward(q).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)
    val kProj = kLinear.forward(k).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)
    val vProj = vLinear.forward(k).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)

    // Cone attention: apply geometric distance penalty
    val scale = new Scalar(scala.math.sqrt(headDim.toDouble).toFloat)
    var scores = torch.matmul(qProj, kProj.transpose(2, 3)).div(scale)

    if (seqLen > 1) {
      // Create position indices
      val posIds = torch.arange(new Scalar(0),new Scalar( seqLen.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      if (device != "cpu") {
        posIds.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
      }
      val posDiff = (posIds.view(seqLen.toLong, 1L).sub(posIds.view(1L, seqLen.toLong))).abs()
      // Cone penalty: tanh(r * d_ij) * gamma
      val conePenalty = torch.tanh(posDiff.mul(new Scalar(r.toDouble.toFloat))).mul(new Scalar(gamma.toDouble.toFloat))
      val coneBias = conePenalty.view(1L, 1L, seqLen.toLong, seqLen.toLong)
      scores = scores.add(coneBias)

      if (mask == 0) {
        // Causal mask
        val causalMask = torch.triu(torch.ones(Array(seqLen.toLong, seqLen.toLong),
          new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))), 1).unsqueeze(0).unsqueeze(0)
        scores = scores.add(causalMask.mul(new Scalar(1e9)))
      }
    }

    val attnWeights = scores.softmax(-1)
    val attended = torch.matmul(dropoutLayer.forward(attnWeights), vProj)
    val reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim)
    outLinear.forward(reshaped)
  }
}
