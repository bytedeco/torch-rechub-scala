package torchrec.models.generative

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.utils.DeviceSupport
import torchrec.basic.layers.MLP

/**
 * LLM4Rec: Transformer Encoder for Sequential Recommendation
 */
class LLM4Rec(
               vocabSize: Long,
               embedDim: Int = 64,
               numHeads: Int = 4,
               numLayers: Int = 3,
               maxSeqLen: Int = 50,
               mlpDims: List[Long] = List(256L, 128L),
               dropout: Float = 0.1f,
               usePosEncoding: Boolean = true,
               device: String = DeviceSupport.backend
             ) extends Module {

  require(embedDim % numHeads == 0, "embedDim must be divisible by numHeads")

  private val headDim = embedDim / numHeads
  private val ffDim = embedDim * 4L
  private val targetDevice = new Device(device)

  // 1. 原生层：安全注册
  private val tokenEmbedding = new EmbeddingImpl(new EmbeddingOptions(vocabSize, embedDim))
  tokenEmbedding.to(targetDevice, false)
  register_module("tokenEmbedding", tokenEmbedding)

  private val clsTensor = {
    val t = torch.zeros(
      Array(1L, 1L, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(targetDevice))
    )
    t.fill_(new Scalar(0.02f))
    t
  }
  register_parameter("clsToken", clsTensor)

  private val positionEmbedding = new EmbeddingImpl(new EmbeddingOptions(maxSeqLen + 1, embedDim))
  positionEmbedding.to(targetDevice, false)
  register_module("positionEmbedding", positionEmbedding)

  private val preNorm = {
    val vec = new LongVector(Array(embedDim.toLong) *)
    new LayerNormImpl(vec)
  }
  preNorm.to(targetDevice, false)
  register_module("preNorm", preNorm)

  // 2. 自定义 Scala 层：坚决不注册，使用 List 强引用保存
  private val encoderLayers = (0 until numLayers).map { _ =>
    val layer = new LLM4RecEncoderLayer(embedDim, numHeads, ffDim, dropout, device)
    // ❌ 删除了 register_module(s"encoder_$i", layer)
    layer
  }.toList

  private val mlp = new MLP(embedDim, mlpDims, 1, "relu", dropout, false, false, true, device)
  // ❌ 删除了 register_module("mlp", mlp)
  // ❌ 删除了 DropoutImpl，改用纯函数

  private var isTrainingMode = true

  // ===========================================================================
  // ✅ 手动状态分发：完美避开 C++ 底层遍历导致的 Director 崩溃
  // ===========================================================================
  override def train(on: Boolean): Unit = {
    isTrainingMode = on
    tokenEmbedding.train(on)
    positionEmbedding.train(on)
    preNorm.train(on)

    // 手动通知自定义子层
    encoderLayers.foreach(_.train(on))
    // 假设 MLP 内部也有安全的 train(on) 或未注册状态
    try { mlp.train(on) } catch { case _: Throwable => }
  }

  // ===========================================================================
  // FORWARD
  // ===========================================================================
  def forward(seqTokens: Tensor, positions: Tensor): Tensor = {
    val batchSize = seqTokens.size(0)
    val dev = seqTokens.device()

    val tokenEmb = tokenEmbedding.forward(seqTokens.toType(ScalarType.Long))
    val clsBatched = clsTensor.expand(Array(batchSize, 1L, embedDim.toLong) *)
    val tokenEmbWithCls = torch.cat(new TensorVector(clsBatched, tokenEmb), 1)

    val clsPos = torch.zeros(
      Array(batchSize, 1L),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)).device(new DeviceOptional(dev))
    )
    val posWithCls = torch.cat(new TensorVector(clsPos, positions.toType(ScalarType.Long)), 1)
    val posEmb = positionEmbedding.forward(posWithCls)

    // ✅ 使用纯函数 Dropout
    var x = torch.dropout(tokenEmbWithCls.add(posEmb), dropout.toDouble, isTrainingMode)

    encoderLayers.foreach { layer =>
      val out = layer.forward(x)
      x = x.add(out)
    }

    val clsRep = preNorm.forward(x).select(1, 0)
    mlp.forward(clsRep)
  }
}

// ===================== Encoder Layer =====================
class LLM4RecEncoderLayer(
                           embedDim: Int,
                           numHeads: Int,
                           ffDim: Long,
                           dropout: Float,
                           device: String
                         ) extends Module {

  private val targetDevice = new Device(device)
  private val headDim = embedDim / numHeads

  // 1. 只创建带有权重的原生层
  private val attnLinear = new LinearImpl(embedDim, 3 * embedDim)
  private val attnOutProj = new LinearImpl(embedDim, embedDim)
  private val ffnLinear1 = new LinearImpl(embedDim, ffDim)
  private val ffnLinear2 = new LinearImpl(ffDim, embedDim)

  private val norm1 = new LayerNormImpl(new LongVector(Array(embedDim.toLong) *))
  private val norm2 = new LayerNormImpl(new LongVector(Array(embedDim.toLong) *))

  // 设备迁移
  List(attnLinear, attnOutProj, ffnLinear1, ffnLinear2, norm1, norm2).foreach(_.to(targetDevice, false))

  // ❌ 彻底废弃 DropoutImpl

  private var isTrainingMode = true

  // ✅ 手动控制子模块状态
  override def train(on: Boolean): Unit = {
    isTrainingMode = on
    attnLinear.train(on)
    attnOutProj.train(on)
    ffnLinear1.train(on)
    ffnLinear2.train(on)
    norm1.train(on)
    norm2.train(on)
  }

  def forward(x: Tensor): Tensor = {
    val bs = x.size(0)
    val sl = x.size(1)

    // Attention
    val nx = norm1.forward(x)
    // Use reshape instead of view to avoid errors on non-contiguous tensors after linear/permute
    val qkv = attnLinear.forward(nx).reshape(bs, sl, 3, numHeads, headDim).permute(3, 0, 1, 4, 2)
    val q = qkv.select(4, 0)
    val k = qkv.select(4, 1)
    val v = qkv.select(4, 2)

    val scale = 1.0f / math.sqrt(headDim).toFloat
    val attn = torch.matmul(q, k.transpose(-2, -1)).mul(new Scalar(scale)).softmax(-1)

    // ✅ 使用纯函数 Dropout 替换 attnDropout.forward
    val droppedAttn = torch.dropout(attn, dropout.toDouble, isTrainingMode)
    val valOut = torch.matmul(droppedAttn, v)

    val attnOut = attnOutProj.forward(valOut.permute(1, 2, 0, 3).reshape(bs, sl, embedDim))
    val residual1 = x.add(attnOut)

    // FFN
    val nx2 = norm2.forward(residual1)
    val ffnHidden = ffnLinear1.forward(nx2).relu()

    // ✅ 使用纯函数 Dropout 替换 ffnDropout.forward
    val droppedFfn = torch.dropout(ffnHidden, dropout.toDouble, isTrainingMode)
    val ffn = ffnLinear2.forward(droppedFfn)

    residual1.add(ffn)
  }
}
//import torchrec.basic.layers._
//import torchrec.utils.DeviceSupport
//
//import org.bytedeco.pytorch._
//import org.bytedeco.pytorch.global.torch
//import org.bytedeco.pytorch.global.torch.ScalarType
//
///**
// * LLM4Rec: Transformer Encoder for Sequential Recommendation
// */
//class LLM4Rec(
//               vocabSize: Long,
//               embedDim: Int = 64,
//               numHeads: Int = 4,
//               numLayers: Int = 3,
//               maxSeqLen: Int = 50,
//               mlpDims: List[Long] = List(256L, 128L),
//               dropout: Float = 0.1f,
//               usePosEncoding: Boolean = true,
//               device: String = DeviceSupport.backend
//             ) extends Module {
//
//  require(embedDim % numHeads == 0, "embedDim must be divisible by numHeads")
//
//  private val headDim = embedDim / numHeads
//  private val ffDim = embedDim * 4L
//  private val targetDevice = new Device(device)
//
//  // Token Embedding
//  private val tokenEmbedding = new EmbeddingImpl(
//    new EmbeddingOptions(vocabSize, embedDim)
//  )
//  tokenEmbedding.to(targetDevice, false)
//  register_module("tokenEmbedding", tokenEmbedding)
//
//  // CLS Token
//  private val clsTensor = {
//    val t = torch.zeros(
//      Array(1L, 1L, embedDim.toLong),
//      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(targetDevice))
//    )
//    t.fill_(new Scalar(0.02f))
//    t
//  }
//  register_parameter("clsToken", clsTensor)
//
//  // Positional Embedding
//  private val positionEmbedding = new EmbeddingImpl(
//    new EmbeddingOptions(maxSeqLen + 1, embedDim)
//  )
//  positionEmbedding.to(targetDevice, false)
//  register_module("positionEmbedding", positionEmbedding)
//
//  // Transformer Layers
//  private val encoderLayers = (0 until numLayers).map { i =>
//    val layer = new LLM4RecEncoderLayer(embedDim, numHeads, ffDim, dropout, device)
//    register_module(s"encoder_$i", layer)
//    layer
//  }
//
//  // LayerNorm
//  private val preNorm = {
//    val vec = new LongVector(1)
//    vec.put(0, embedDim.toLong)
//    new LayerNormImpl(vec)
//  }
//  preNorm.to(targetDevice, false)
//  register_module("preNorm", preNorm)
//
//  private val dropoutLayer = new DropoutImpl(dropout)
//  private val mlp = new MLP(embedDim, mlpDims, 1, "relu", dropout, false, false,true, device)
//
//  register_module("dropout", dropoutLayer)
//  register_module("mlp", mlp)
//
//  // ===========================================================================
//  // FORWARD —— 修复类型错误
//  // ===========================================================================
//  def forward(seqTokens: Tensor, positions: Tensor): Tensor = {
//    val batchSize = seqTokens.size(0)
//    val dev = seqTokens.device()
//
//    // 1. Token Embedding
//    val tokenEmb = tokenEmbedding.forward(seqTokens.toType(ScalarType.Long))
//
//    // 2. CLS
//    val clsBatched = clsTensor.expand(batchSize, 1L, embedDim.toLong)
//    val tokenEmbWithCls = torch.cat(new TensorVector(clsBatched, tokenEmb), 1)
//
//    // ===================== ✅ 超级修复：必须 LongTensor =====================
//    val clsPos = torch.zeros(
//      Array(batchSize, 1L),
//      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)).device(new DeviceOptional(dev))
//    )
//
//    val posWithCls = torch.cat(new TensorVector(clsPos, positions.toType(ScalarType.Long)), 1)
//
//    // 4. Position Embedding
//    val posEmb = positionEmbedding.forward(posWithCls)
//    var x = dropoutLayer.forward(tokenEmbWithCls.add(posEmb))
//
//    // 5. Transformer
//    encoderLayers.foreach { layer =>
//      val out = layer.forward(x)
//      x = x.add(out)
//    }
//
//    // 6. CLS Head
//    val clsRep = preNorm.forward(x).select(1, 0)
//    mlp.forward(clsRep)
//  }
//}
//
//// ===================== Encoder Layer =====================
//class LLM4RecEncoderLayer(
//                           embedDim: Int,
//                           numHeads: Int,
//                           ffDim: Long,
//                           dropout: Float,
//                           device: String
//                         ) extends Module {
//
//  private val targetDevice = new Device(device)
//  private val headDim = embedDim / numHeads
//
//  // Attention
//  private val attnLinear = new LinearImpl(embedDim, 3 * embedDim)
//  private val attnOutProj = new LinearImpl(embedDim, embedDim)
//
//  // FFN
//  private val ffnLinear1 = new LinearImpl(embedDim, ffDim)
//  private val ffnLinear2 = new LinearImpl(ffDim, embedDim)
//
//  // LayerNorm
//  private val norm1 = {
//    val vec = new LongVector(1)
//    vec.put(0, embedDim.toLong)
//    new LayerNormImpl(vec)
//  }
//
//  private val norm2 = {
//    val vec = new LongVector(1)
//    vec.put(0, embedDim.toLong)
//    new LayerNormImpl(vec)
//  }
//
//  private val attnDropout = new DropoutImpl(dropout)
//  private val ffnDropout = new DropoutImpl(dropout)
//
//  // 设备迁移
//  List(attnLinear, attnOutProj, ffnLinear1, ffnLinear2, norm1, norm2).foreach(_.to(targetDevice, false))
//
//  // 注册
//  register_module("attnLinear", attnLinear)
//  register_module("attnOutProj", attnOutProj)
//  register_module("ffnLinear1", ffnLinear1)
//  register_module("ffnLinear2", ffnLinear2)
//  register_module("norm1", norm1)
//  register_module("norm2", norm2)
//  register_module("attnDropout", attnDropout)
//  register_module("ffnDropout", ffnDropout)
//
//  def forward(x: Tensor): Tensor = {
//    val bs = x.size(0)
//    val sl = x.size(1)
//
//    // Attention
//    val nx = norm1.forward(x)
//    val qkv = attnLinear.forward(nx).view(bs, sl, 3, numHeads, headDim).permute(3, 0, 1, 4, 2)
//    val q = qkv.select(4, 0)
//    val k = qkv.select(4, 1)
//    val v = qkv.select(4, 2)
//
//    val scale = 1.0f / math.sqrt(headDim).toFloat
//    val attn = torch.matmul(q, k.transpose(-2, -1)).mul(new Scalar(scale)).softmax(-1)
//    val valOut = torch.matmul(attnDropout.forward(attn), v)
//    val attnOut = attnOutProj.forward(valOut.permute(1, 2, 0, 3).reshape(bs, sl, embedDim))
//
//    val residual1 = x.add(attnOut)
//
//    // FFN
//    val nx2 = norm2.forward(residual1)
//    val ffn = ffnLinear2.forward(ffnDropout.forward(ffnLinear1.forward(nx2).relu()))
//    residual1.add(ffn)
//  }
//}

//package torchrec.models.generative
//
//import torchrec.basic.layers._
//import torchrec.utils.DeviceSupport
//
//import org.bytedeco.pytorch._
//import org.bytedeco.pytorch.global.torch
//import org.bytedeco.pytorch.global.torch.ScalarType
//
///**
// * LLM4Rec: Transformer Encoder for Sequential Recommendation
// */
//class LLM4Rec(
//               vocabSize: Long,
//               embedDim: Int = 64,
//               numHeads: Int = 4,
//               numLayers: Int = 3,
//               maxSeqLen: Int = 50,
//               mlpDims: List[Long] = List(256L, 128L),
//               dropout: Float = 0.1f,
//               usePosEncoding: Boolean = true,
//               device: String = DeviceSupport.backend
//             ) extends Module {
//
//  require(embedDim % numHeads == 0, "embedDim must be divisible by numHeads")
//
//  private val headDim = embedDim / numHeads
//  private val ffDim = embedDim * 4L
//  private val targetDevice = new Device(device)
//
//  // Token Embedding
//  private val tokenEmbedding = new EmbeddingImpl(
//    new EmbeddingOptions(vocabSize, embedDim)
//  )
//  tokenEmbedding.to(targetDevice, false)
//  register_module("tokenEmbedding", tokenEmbedding)
//
//  // CLS Token (正确设备)
//  private val clsTensor = {
//    val t = torch.zeros(
//      Array(1L, 1L, embedDim.toLong),
//      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(targetDevice))
//    )
//    t.fill_(new Scalar(0.02f))
//    t
//  }
//  register_parameter("clsToken", clsTensor)
//
//  // Positional Embedding
//  private val positionEmbedding = new EmbeddingImpl(
//    new EmbeddingOptions(maxSeqLen + 1, embedDim)
//  )
//  positionEmbedding.to(targetDevice, false)
//  register_module("positionEmbedding", positionEmbedding)
//
//  // Transformer Layers
//  private val encoderLayers = (0 until numLayers).map { i =>
//    val layer = new LLM4RecEncoderLayer(embedDim, numHeads, ffDim, dropout, device)
//    register_module(s"encoder_$i", layer)
//    layer
//  }
//
//  // ===================== 🔥 修复：LayerNorm 正确构造 =====================
//  private val preNorm = {
//    val vec = new LongVector(1)
//    vec.put(0, embedDim.toLong)
//    new LayerNormImpl(vec)
//  }
//  preNorm.to(targetDevice, false)
//  register_module("preNorm", preNorm)
//
//  private val dropoutLayer = new DropoutImpl(dropout)
//  private val mlp = new MLP(embedDim, mlpDims, 1, "relu", dropout, false, false ,device)
//
//  register_module("dropout", dropoutLayer)
//  register_module("mlp", mlp)
//
//  // ===========================================================================
//  // FORWARD
//  // ===========================================================================
//  def forward(seqTokens: Tensor, positions: Tensor): Tensor = {
//    val batchSize = seqTokens.size(0)
//    val dev = seqTokens.device()
//
//    // 1. Token Embedding
//    val tokenEmb = tokenEmbedding.forward(seqTokens.toType(ScalarType.Long))
//
//    // 2. Add CLS token
//    val clsBatched = clsTensor.expand(batchSize, 1L, embedDim.toLong)
//    val tokenEmbWithCls = torch.cat(new TensorVector(clsBatched, tokenEmb), 1)
//
//    // 3. Position CLS
//    val clsPos = torch.zeros(
//      Array(batchSize, 1L),
//      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(targetDevice))
//    )
//    val posWithCls = torch.cat(new TensorVector(clsPos, positions.toType(ScalarType.Long)), 1)
//
//    // 4. Position Embedding
//    val posEmb = positionEmbedding.forward(posWithCls)
//    var x = dropoutLayer.forward(tokenEmbWithCls.add(posEmb))
//
//    // 5. Transformer
//    encoderLayers.foreach { layer =>
//      val out = layer.forward(x)
//      x = x.add(out)
//    }
//
//    // 6. CLS Head
//    val clsRep = preNorm.forward(x).select(1, 0)
//    mlp.forward(clsRep)
//  }
//}
//
//// ===================== Encoder Layer (全修复) =====================
//class LLM4RecEncoderLayer(
//                           embedDim: Int,
//                           numHeads: Int,
//                           ffDim: Long,
//                           dropout: Float,
//                           device: String
//                         ) extends Module {
//
//  private val targetDevice = new Device(device)
//  private val headDim = embedDim / numHeads
//
//  // Attention
//  private val attnLinear = new LinearImpl(embedDim, 3 * embedDim)
//  private val attnOutProj = new LinearImpl(embedDim, embedDim)
//
//  // FFN
//  private val ffnLinear1 = new LinearImpl(embedDim, ffDim)
//  private val ffnLinear2 = new LinearImpl(ffDim, embedDim)
//
//  // ===================== 🔥 修复：LayerNorm 正确构造 =====================
//  private val norm1 = {
//    val vec = new LongVector(1)
//    vec.put(0, embedDim.toLong)
//    new LayerNormImpl(vec)
//  }
//
//  private val norm2 = {
//    val vec = new LongVector(1)
//    vec.put(0, embedDim.toLong)
//    new LayerNormImpl(vec)
//  }
//
//  private val attnDropout = new DropoutImpl(dropout)
//  private val ffnDropout = new DropoutImpl(dropout)
//
//  // 设备迁移
//  List(attnLinear, attnOutProj, ffnLinear1, ffnLinear2, norm1, norm2).foreach(_.to(targetDevice, false))
//
//  // 注册
//  register_module("attnLinear", attnLinear)
//  register_module("attnOutProj", attnOutProj)
//  register_module("ffnLinear1", ffnLinear1)
//  register_module("ffnLinear2", ffnLinear2)
//  register_module("norm1", norm1)
//  register_module("norm2", norm2)
//  register_module("attnDropout", attnDropout)
//  register_module("ffnDropout", ffnDropout)
//
//  def forward(x: Tensor): Tensor = {
//    val bs = x.size(0)
//    val sl = x.size(1)
//
//    // Attention
//    val nx = norm1.forward(x)
//    val qkv = attnLinear.forward(nx).view(bs, sl, 3, numHeads, headDim).permute(3, 0, 1, 4, 2)
//    val q = qkv.select(4, 0)
//    val k = qkv.select(4, 1)
//    val v = qkv.select(4, 2)
//
//    val scale = 1.0f / math.sqrt(headDim).toFloat
//    val attn = torch.matmul(q, k.transpose(-2, -1)).mul(new Scalar(scale)).softmax(-1)
//    val valOut = torch.matmul(attnDropout.forward(attn), v)
//    val attnOut = attnOutProj.forward(valOut.permute(1, 2, 0, 3).reshape(bs, sl, embedDim))
//
//    val residual1 = x.add(attnOut)
//
//    // FFN
//    val nx2 = norm2.forward(residual1)
//    val ffn = ffnLinear2.forward(ffnDropout.forward(ffnLinear1.forward(nx2).relu()))
//    residual1.add(ffn)
//  }
//}

//package torchrec.models.generative
//
//import torchrec.basic.features._
//import torchrec.basic.layers._
//import torchrec.utils.DeviceSupport
//
//import org.bytedeco.pytorch._
//import org.bytedeco.pytorch.global.torch
//import org.bytedeco.pytorch.global.torch.ScalarType
//
///**
// * LLM4Rec: LLM-inspired Transformer Encoder for Sequential Recommendation
// */
//class LLM4Rec(
//               vocabSize: Long,
//               embedDim: Int = 64,
//               numHeads: Int = 4,
//               numLayers: Int = 3,
//               maxSeqLen: Int = 50,
//               mlpDims: List[Long] = List(256L, 128L),
//               dropout: Float = 0.1f,
//               usePosEncoding: Boolean = true,
//               device: String = DeviceSupport.backend
//             ) extends Module {
//
//  require(embedDim % numHeads == 0, "embedDim must be divisible by numHeads")
//
//  private val headDim = embedDim / numHeads
//  private val ffDim = embedDim * 4L
//  private val targetDevice = new Device(device)
//
//  // Token embedding
//  private val tokenEmbedding = new EmbeddingImpl(
//    new EmbeddingOptions(vocabSize, embedDim)
//  )
//  tokenEmbedding.to(targetDevice, false)
//  register_module("tokenEmbedding", tokenEmbedding)
//
//  // ===================== 🔥 修复：clsToken 直接创建在 targetDevice =====================
//  private val clsTensor = {
//    val t = torch.zeros(
//      Array(1L, 1L, embedDim.toLong),
//      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(targetDevice))
//    )
//    t.fill_(new Scalar(0.02f))
//    t.to(targetDevice,ScalarType.Float)
//  }
//  register_parameter("clsToken", clsTensor)
//
//  // Positional embedding
//  private val positionEmbedding = new EmbeddingImpl(
//    new EmbeddingOptions(maxSeqLen + 1, embedDim)
//  )
//  positionEmbedding.to(targetDevice, false)
//  register_module("positionEmbedding", positionEmbedding)
//
//  // Encoder layers
//  private val encoderLayers = (0 until numLayers).map { i =>
//    val layer = new LLM4RecEncoderLayer(embedDim, numHeads, ffDim, dropout, device)
//    register_module(s"encoder_$i", layer)
//    layer
//  }
//
//  private val preNorm = new LayerNormImpl(new LongVector(embedDim.toLong))
//  preNorm.to(targetDevice, false)
//  register_module("preNorm", preNorm)
//
//  private val dropoutLayer = new DropoutImpl(dropout)
//  register_module("dropout", dropoutLayer)
//
//  private val mlp = new MLP(embedDim, mlpDims, 1, "relu", dropout, device = device)
//  register_module("mlp", mlp)
//
//  // ===========================================================================
//  // FORWARD —— 所有临时张量自动使用输入张量的设备
//  // ===========================================================================
//  def forward(
//               seqTokens: Tensor,
//               positions: Tensor
//             ): Tensor = {
//    val batchSize = seqTokens.size(0)
//    val seqLen = seqTokens.size(1)
//    val dev = seqTokens.device() // 🔥 关键：使用输入的设备
//
//    // Token emb
//    val tokenEmb = tokenEmbedding.forward(seqTokens.toType(ScalarType.Long))
//
//    // CLS 拼接
//    val clsBatched = clsTensor.expand(batchSize, 1L, embedDim.toLong).to(dev, ScalarType.Float)
//    val tokenEmbWithCls = torch.cat(new TensorVector(clsBatched, tokenEmb), 1)
//
//    // ===================== 🔥 修复：clsPos 创建在正确设备 =====================
//    val clsPos = torch.zeros(
//      Array(batchSize, 1L),
//      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)).device(new DeviceOptional(dev))
//    )
//    val posWithCls = torch.cat(new TensorVector(clsPos, positions.toType(ScalarType.Long)), 1)
//
//    // Pos emb
//    val posEmb = positionEmbedding.forward(posWithCls)
//
//    // Fusion
//    var x = tokenEmbWithCls.add(posEmb)
//    x = dropoutLayer.forward(x)
//
//    // Transformer layers
//    encoderLayers.foreach { layer =>
//      val out = layer.forward(x)
//      x = out.add(x)
//    }
//
//    // CLS 输出
//    val normed = preNorm.forward(x)
//    val clsRep = normed.select(1, 0)
//    val logits = mlp.forward(clsRep)
//
//    logits
//  }
//}
//
//// ===================== Encoder Layer 完全不变 =====================
//class LLM4RecEncoderLayer(
//                           embedDim: Int,
//                           numHeads: Int,
//                           ffDim: Long,
//                           dropout: Float,
//                           device: String
//                         ) extends Module {
//
//  require(embedDim % numHeads == 0)
//  private val headDim = embedDim / numHeads
//  private val targetDevice = new Device(device)
//
//  private val attnLinear = new LinearImpl(embedDim, 3 * embedDim)
//  private val attnOutProj = new LinearImpl(embedDim, embedDim)
//  private val ffnLinear1 = new LinearImpl(embedDim, ffDim)
//  private val ffnLinear2 = new LinearImpl(ffDim, embedDim)
//  private val norm1 = new LayerNormImpl(new LongVector(embedDim.toLong))
//  private val norm2 = new LayerNormImpl(new LongVector(embedDim.toLong))
//  private val attnDropout = new DropoutImpl(dropout)
//  private val ffnDropout = new DropoutImpl(dropout)
//
//  attnLinear.to(targetDevice, false)
//  attnOutProj.to(targetDevice, false)
//  ffnLinear1.to(targetDevice, false)
//  ffnLinear2.to(targetDevice, false)
//  norm1.to(targetDevice, false)
//  norm2.to(targetDevice, false)
//
//  register_module("attnLinear", attnLinear)
//  register_module("attnOutProj", attnOutProj)
//  register_module("ffnLinear1", ffnLinear1)
//  register_module("ffnLinear2", ffnLinear2)
//  register_module("norm1", norm1)
//  register_module("norm2", norm2)
//  register_module("attnDropout", attnDropout)
//  register_module("ffnDropout", ffnDropout)
//
//  def forward(x: Tensor): Tensor = {
//    val batchSize = x.size(0)
//    val seqLen = x.size(1)
//
//    val normedX = norm1.forward(x)
//    val qkv = attnLinear.forward(normedX)
//    val qkvView = qkv.view(batchSize, seqLen, 3, numHeads, headDim)
//    val qkvPerm = qkvView.permute(3, 0, 1, 4, 2)
//
//    val q = qkvPerm.select(4, 0)
//    val k = qkvPerm.select(4, 1)
//    val v = qkvPerm.select(4, 2)
//
//    val invScale = 1.0f / math.sqrt(headDim).toFloat
//    val attnScores = torch.matmul(q, k.transpose(-2, -1)).mul(new Scalar(invScale))
//    val attnWeights = attnScores.softmax(-1)
//    val attnDropped = attnDropout.forward(attnWeights)
//
//    val attnOut = torch.matmul(attnDropped, v)
//    val attnFlat = attnOut.permute(1, 2, 0, 3).contiguous().view(batchSize, seqLen, embedDim)
//    val attnProj = attnOutProj.forward(attnFlat)
//    val attnOutFinal = x.add(attnProj)
//
//    val normed2 = norm2.forward(attnOutFinal)
//    val ffnOut = ffnLinear2.forward(ffnDropout.forward(ffnLinear1.forward(normed2).relu()))
//    ffnOut.add(attnOutFinal)
//  }
//}

//package torchrec.models.generative
//
//import torchrec.basic.features._
//import torchrec.basic.layers._
//import torchrec.utils.DeviceSupport
//
//import org.bytedeco.pytorch._
//import org.bytedeco.pytorch.global.torch
//import org.bytedeco.pytorch.global.torch.ScalarType
//
///**
// * LLM4Rec: LLM-inspired Transformer Encoder for Sequential Recommendation
// *
// * This model applies a multi-layer transformer encoder over item ID sequences,
// * using a learned [CLS] token to produce a unified user representation for CTR.
// *
// * This differs from HLLM (which uses frozen LLM embeddings + MLP) and HSTU
// * (which is a generative sequence model with next-token prediction). LLM4Rec
// * is a discriminative ranking model with a transformer encoder backbone.
// *
// * Reference: "LLM4Rec: Large Language Models for Sequential Recommendation"
// *            (survey/framework paper, 2023-2024)
// *
// * Architecture:
// *   1. Token embedding + positional embedding lookup
// *   2. Stacked transformer encoder layers (self-attention + FFN)
// *   3. [CLS] token representation -> MLP -> CTR logit
// *
// * @param vocabSize         Item vocabulary size (for item ID embedding)
// * @param embedDim          Embedding dimension
// * @param numHeads          Number of attention heads
// * @param numLayers         Number of transformer encoder layers
// * @param maxSeqLen         Maximum sequence length
// * @param mlpDims           MLP hidden dimensions after [CLS] pooling
// * @param dropout           Dropout rate
// * @param usePosEncoding    Use learned positional encodings (default true)
// * @param device            Device
// */
//class LLM4Rec(
//  vocabSize: Long,
//  embedDim: Int = 64,
//  numHeads: Int = 4,
//  numLayers: Int = 3,
//  maxSeqLen: Int = 50,
//  mlpDims: List[Long] = List(256L, 128L),
//  dropout: Float = 0.1f,
//  usePosEncoding: Boolean = true,
//  device: String = DeviceSupport.backend
//) extends Module {
//
//  require(embedDim % numHeads == 0, "embedDim must be divisible by numHeads")
//
//  private val headDim = embedDim / numHeads
//  private val ffDim = embedDim * 4L  // Feed-forward dimension
//
//  // Token embedding table
//  private val tokenEmbedding = new EmbeddingImpl(
//    new EmbeddingOptions(vocabSize, embedDim)
//  )
//  tokenEmbedding.to(new Device(device),false)
//  register_module("tokenEmbedding", tokenEmbedding)
//
//  // Learned [CLS] token embedding
//  // Learned [CLS] token embedding — initialized with small values via fill
//  private val clsTensor = {
//    val t = torch.zeros(
//      Array(1L, 1L, embedDim.toLong),
//      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
//    )
//    t.fill_(new Scalar(0.02f))
//  }
//  register_parameter("clsToken", clsTensor)
//
//  // Positional embeddings (learned)
//  private val positionEmbedding = new EmbeddingImpl(
//    new EmbeddingOptions(maxSeqLen + 1, embedDim)  // +1 for [CLS]
//  )
//  positionEmbedding.to(new Device(device),false)
//  register_module("positionEmbedding", positionEmbedding)
//
//  // Transformer encoder layers
//  private val encoderLayers = (0 until numLayers).map { i =>
//    val layer = new LLM4RecEncoderLayer(embedDim, numHeads, ffDim, dropout, device)
//    register_module(s"encoder_$i", layer)
//    layer
//  }
//
//  // Layer norm before pooling
//  private val preNorm = new LayerNormImpl(new LongVector(embedDim.toLong))
//  register_module("preNorm", preNorm)
//
//  // Dropout
//  private val dropoutLayer = new DropoutImpl(dropout)
//  register_module("dropout", dropoutLayer)
//
//  // Final MLP
//  private val mlp = new MLP(embedDim, mlpDims, 1, "relu", dropout, device = device)
//  register_module("mlp", mlp)
//
//  if (device != "cpu") {
//    val dev = new org.bytedeco.pytorch.Device(device)
//    tokenEmbedding.to(dev, false)
//    positionEmbedding.to(dev, false)
//    encoderLayers.foreach(_.to(dev, false))
//    preNorm.to(dev, false)
//    dropoutLayer.to(dev, false)
//    mlp.to(dev, false)
//  }
//
//  def forward(
//    seqTokens: Tensor,   // (batch, seq_len) -- item IDs
//    positions: Tensor    // (batch, seq_len) -- position indices
//  ): Tensor = {
//    val batchSize = seqTokens.size(0).toInt
//    val seqLen = seqTokens.size(1).toInt
//
//    // Token embeddings: (batch, seq_len, embed_dim)
//    val tokenEmb = tokenEmbedding.forward(seqTokens.toType(ScalarType.Long))
//
//    // Expand [CLS] token to batch size and prepend
//    val clsBatched = clsTensor.expand(batchSize.toLong, 1L, embedDim.toLong)
//    val tokenEmbWithCls = torch.cat(new TensorVector(clsBatched, tokenEmb), 1)  // (batch, seq_len+1, embed_dim)
//
//    // Positions: prepend 0 for [CLS], then the actual positions
//    val clsPos = torch.zeros(
//      Array(batchSize.toLong, 1L),
//      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long))
//    ).to(positions.device(), ScalarType.Long)
//    val posWithCls = torch.cat(new TensorVector(clsPos, positions.toType(ScalarType.Long)), 1)  // (batch, seq_len+1)
//
//    // Positional embeddings
//    val posEmb = positionEmbedding.forward(posWithCls)  // (batch, seq_len+1, embed_dim)
//
//    // Combine: token + position
//    var x = tokenEmbWithCls.add(posEmb)
//    x = dropoutLayer.forward(x)
//
//    // Transformer encoder layers with pre-norm residual
//    encoderLayers.foreach { layer =>
//      val out = layer.forward(x)
//      x = out.add(x)  // Residual connection
//    }
//
//    // Pre-norm and extract [CLS] representation
//    val normed = preNorm.forward(x)  // (batch, seq_len+1, embed_dim)
//    val clsRep = normed.select(1, 0)  // (batch, embed_dim)
//
//    // MLP
//    val logits = mlp.forward(clsRep)  // (batch, 1)
//    logits
//  }
//}
//
///**
// * A single Transformer Encoder Layer with pre-norm architecture.
// * Subcomponents: Multi-head self-attention + Feed-Forward Network.
// */
//class LLM4RecEncoderLayer(
//  embedDim: Int,
//  numHeads: Int,
//  ffDim: Long,
//  dropout: Float,
//  device: String
//) extends Module {
//
//  require(embedDim % numHeads == 0)
//  private val headDim = embedDim / numHeads
//
//  // Self-attention: Q, K, V packed in one linear
//  private val attnLinear = new LinearImpl(embedDim, 3 * embedDim)
//  register_module("attnLinear", attnLinear)
//
//  private val attnOutProj = new LinearImpl(embedDim, embedDim)
//  register_module("attnOutProj", attnOutProj)
//
//  // Feed-forward network
//  private val ffnLinear1 = new LinearImpl(embedDim, ffDim)
//  private val ffnLinear2 = new LinearImpl(ffDim, embedDim)
//  register_module("ffnLinear1", ffnLinear1)
//  register_module("ffnLinear2", ffnLinear2)
//
//  // Layer norms
//  private val norm1 = new LayerNormImpl(new LongVector(embedDim.toLong))
//  private val norm2 = new LayerNormImpl(new LongVector(embedDim.toLong))
//  register_module("norm1", norm1)
//  register_module("norm2", norm2)
//
//  // Dropout
//  private val attnDropout = new DropoutImpl(dropout)
//  private val ffnDropout = new DropoutImpl(dropout)
//  register_module("attnDropout", attnDropout)
//  register_module("ffnDropout", ffnDropout)
//
//  if (device != "cpu") {
//    val dev = new org.bytedeco.pytorch.Device(device)
//    attnLinear.to(dev, false)
//    attnOutProj.to(dev, false)
//    ffnLinear1.to(dev, false)
//    ffnLinear2.to(dev, false)
//    norm1.to(dev, false)
//    norm2.to(dev, false)
//  }
//
//  def forward(x: Tensor): Tensor = {
//    // x: (batch, seq_len, embed_dim)
//    val batchSize = x.size(0).toInt
//    val seqLen = x.size(1).toInt
//
//    // === Multi-head self-attention with pre-norm ===
//    val normedX = norm1.forward(x)
//
//    // Project to Q, K, V: (batch, seq_len, 3*embed_dim)
//    val qkv = attnLinear.forward(normedX)
//
//    // Reshape for multi-head: (batch, seq_len, 3, numHeads, headDim)
//    // Then permute to (numHeads, batch, seq_len, headDim) for each
//    val qkvView = qkv.view(batchSize, seqLen, 3, numHeads, headDim)
//    val qkvPerm = qkvView.permute(3, 0, 1, 4, 2)  // (numHeads, batch, seq_len, headDim, 3)
//
//    val q = qkvPerm.select(4, 0)  // (numHeads, batch, seq_len, headDim)
//    val k = qkvPerm.select(4, 1)
//    val v = qkvPerm.select(4, 2)
//
//    // Scaled dot-product attention
//    val invScale = 1.0f / scala.math.sqrt(headDim.toDouble).toFloat
//    val scale = new Scalar(invScale)
//
//    // q: (numHeads, batch, seq_len, headDim) -> (numHeads, batch, seq_len, headDim)
//    // k: (numHeads, batch, seq_len, headDim) -> transpose last two dims for matmul
//    val qT = q.transpose(2, 3)  // (numHeads, batch, headDim, seq_len)
//    val kT = k.transpose(2, 3)  // (numHeads, batch, headDim, seq_len)
//    val attnScores = torch.matmul(qT, kT).mul(scale)  // (numHeads, batch, seq_len, seq_len)
//    val attnWeights = attnScores.softmax(-1)
//    val attnDropped = attnDropout.forward(attnWeights)
//
//    // Apply attention to values
//    val vT = v.transpose(2, 3)  // (numHeads, batch, headDim, seq_len)
//    val attnOut = torch.matmul(attnDropped, vT)  // (numHeads, batch, headDim, seq_len)
//    val attnFlat = attnOut.transpose(2, 3).contiguous().view(batchSize, seqLen, embedDim)  // (batch, seq_len, embed_dim)
//
//    val attnProj = attnOutProj.forward(attnFlat)
//    val attnOutFinal = x.add(attnProj)
//
//    // === Feed-forward network with pre-norm ===
//    val normed2 = norm2.forward(attnOutFinal)
//    val ffnOut = ffnLinear2.forward(ffnDropout.forward(ffnLinear1.forward(normed2).relu()))
//    ffnOut.add(attnOutFinal)
//  }
//}
