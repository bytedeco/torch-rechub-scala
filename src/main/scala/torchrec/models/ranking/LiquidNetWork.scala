package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

class LiquidNetWork(
                     features: List[Feature],
                     sequenceFeatures: List[SequenceFeature],
                     embedDim: Int = 8,
                     hiddenDim: Int = 16,
                     numOdeSteps: Int = 3,
                     mlpDims: List[Long] = List(64L, 32L),
                     dropout: Float = 0.2f,
                     device: String = DeviceSupport.backend
                   ) extends Module {

  private val targetDevice = new Device(device)

  // 1. 分离 EmbeddingLayer：彻底隔离稀疏特征与序列特征的拼接污染
  private val sparseEmbedding = new EmbeddingLayer(features, embedDim, device)
  private val seqEmbedding = new EmbeddingLayer(sequenceFeatures, embedDim, device)
  register_module("sparseEmbedding", sparseEmbedding)
  register_module("seqEmbedding", seqEmbedding)

  // 2. Liquid ODE 单元
  private val liquidCell = new LiquidCell(embedDim, hiddenDim, device)
  register_module("liquidCell", liquidCell)

  // 3. 投影层
  private val inputProj  = new LinearImpl(embedDim, hiddenDim)
  private val outputProj = new LinearImpl(hiddenDim, embedDim)
  inputProj.to(targetDevice, false)
  outputProj.to(targetDevice, false)
  register_module("inputProj", inputProj)
  register_module("outputProj", outputProj)

  // 4. MLP
  private val sparseDim = features.size * embedDim
  private val mlpInputDim = sparseDim + embedDim
  private val mlp = new MLP(mlpInputDim, mlpDims, 1, "relu", dropout, false, false, true, device)
  register_module("mlp", mlp)

  // ===========================================================================
  // ✅ 完美兼容真实推理与 Benchmark 测试
  // ===========================================================================
  def forward(sparseFeats: Map[String, Tensor], sequenceFeats: Map[String, Tensor]): Tensor = {
    // 动态获取 Batch Size
    val batchSize = if (sparseFeats.nonEmpty) {
      sparseFeats.head._2.size(0)
    } else if (sequenceFeats.nonEmpty) {
      sequenceFeats.head._2.size(0)
    } else {
      128L
    }

    val tensorOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(targetDevice))

    // --- 1. 稀疏特征 Embedding ---
    val sparseEmb = if (sparseFeats.nonEmpty && features.nonEmpty) {
      val emb = sparseEmbedding.forward(sparseFeats, Map.empty, squeeze = true)
      if (emb.dim() == 1) emb.unsqueeze(0) else emb
    } else {
      torch.zeros(Array(batchSize, sparseDim.toLong), tensorOpts)
    }

    // --- 2. 序列特征 Embedding ---
    val seqEmbRaw = if (sequenceFeats.nonEmpty && sequenceFeatures.nonEmpty) {
      val raw = seqEmbedding.forward(Map.empty, sequenceFeats, squeeze = false)
      if (raw.dim() == 4) raw.squeeze(0) else raw
    } else {
      torch.zeros(Array(batchSize, 20L, embedDim.toLong), tensorOpts)
    }

    // 🔥 强制防御：如果 EmbeddingLayer 错误地将序列展平为 2D [batch, seqLen * embedDim]，则强行 view 回 3D
    val seqEmb = if (seqEmbRaw.dim() == 2) {
      val slen = seqEmbRaw.size(1) / embedDim
      seqEmbRaw.view(batchSize, slen, embedDim)
    } else {
      seqEmbRaw
    }

    val seqLen = seqEmb.size(1).toInt

    // --- 3. ODE 初始状态 ---
    val firstStep = seqEmb.select(1, 0) // 现在绝对安全，稳定切出 [batch, embedDim]
    var hidden = inputProj.forward(firstStep)
    val dt = new Scalar(1f / numOdeSteps)

    // --- 4. ODE 积分迭代 (欧拉法) ---
    for (step <- 0 until numOdeSteps) {
      val t = step.toFloat / numOdeSteps
      val f = t * (seqLen - 1)
      val i = math.min(f.toInt, seqLen - 2)
      val a = new Scalar(f - i)

      val x0 = seqEmb.select(1, i)
      val x1 = seqEmb.select(1, i + 1)
      val xt = x0.mul(new Scalar(1f - a.toFloat)).add(x1.mul(a))

      val dh = liquidCell.forward(hidden, xt, t)
      hidden = hidden.add(dh.mul(dt))
    }

    // --- 5. 投影回 Embed 维度并拼接 ---
    val seqOut = outputProj.forward(hidden)
    val combined = torch.cat(new TensorVector(sparseEmb, seqOut), 1)

    // --- 6. MLP 输出 ---
    mlp.forward(combined) //.squeeze(-1)
  }
}

// ===================== LiquidCell =====================
class LiquidCell(inputDim: Int, hiddenDim: Int, device: String) extends Module {
  private val dev = new Device(device)

  private val tc = new LinearImpl(1, hiddenDim)
  private val ip = new LinearImpl(inputDim, hiddenDim)
  private val hp = new LinearImpl(hiddenDim, hiddenDim)
  private val tm = new LinearImpl(1, hiddenDim)
  private val gw = new LinearImpl(inputDim, hiddenDim)
  private val gh = new LinearImpl(hiddenDim, hiddenDim)
  private val gb = new LinearImpl(1, hiddenDim)

  List(tc, ip, hp, tm, gw, gh, gb).foreach(_.to(dev, false))

  register_module("timeConstant", tc)
  register_module("inputProj", ip)
  register_module("hiddenProj", hp)
  register_module("timeMod", tm)
  register_module("gateW", gw)
  register_module("gateH", gh)
  register_module("gateB", gb)

  def forward(hidden: Tensor, input: Tensor, time: Float): Tensor = {
    val batchSize = hidden.size(0)

    val timeTensor = torch.ones(
      Array(batchSize, 1L),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(hidden.device()))
    ).mul(new Scalar(time))

    val ntc = tc.forward(timeTensor).relu().neg()
    val rec = hp.forward(hidden)
    val inp = ip.forward(input)
    val tmd = tm.forward(timeTensor)

    val gate = gw.forward(input).add(gh.forward(hidden)).add(gb.forward(timeTensor)).sigmoid()

    rec.mul(gate).add(inp).add(tmd).add(hidden.mul(ntc))
  }
}

