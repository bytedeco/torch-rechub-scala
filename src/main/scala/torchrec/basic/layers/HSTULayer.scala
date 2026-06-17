package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.utils.DeviceSupport

class HSTULayer(
  dModel: Int = 512,
  nHeads: Int = 8,
  dqk: Int = 64,
  dv: Int = 64,
  dropout: Float = 0.1f,
  maxSeqLen: Int = 200,
  numTimeBuckets: Int = 128,
  timeBucketFn: String = "sqrt",
  timeBucketDivisor: Float = 1.0f,
  timeBucketUnit: String = "minutes",
  device: String = DeviceSupport.backend
) extends Module {

  require(dModel % nHeads == 0, s"d_model ($dModel) must be divisible by n_heads ($nHeads)")

  private val attnAlpha = 1.0f / math.sqrt(dqk).toFloat

  private val normIn = {
    val vec = new LongVector(1)
    vec.put(0, dModel.toLong)
    new LayerNormImpl(vec)
  }
  register_module("norm_in", normIn)
  normIn.to(new org.bytedeco.pytorch.Device(device), false)

  private val proj1 = new LinearImpl(dModel, 2 * nHeads * dqk + 2 * nHeads * dv)
  register_module("proj1", proj1)
  proj1.to(new org.bytedeco.pytorch.Device(device), false)

  private val rab = new RelativeBucketedTimeAndPositionBias(
    nHeads = nHeads,
    maxSeqLen = maxSeqLen,
    numTimeBuckets = numTimeBuckets,
    timeBucketFn = timeBucketFn,
    timeBucketDivisor = timeBucketDivisor,
    timeBucketUnit = timeBucketUnit
  )
  register_module("rab", rab)

  private val normAttn = {
    val vec = new LongVector(1)
    vec.put(0, (nHeads * dv).toLong)
    new LayerNormImpl(vec)
  }
  register_module("norm_attn", normAttn)
  normAttn.to(new org.bytedeco.pytorch.Device(device), false)

  private val proj2 = new LinearImpl(nHeads * dv, dModel)
  register_module("proj2", proj2)
  proj2.to(new org.bytedeco.pytorch.Device(device), false)

  private val dropoutLayer = new DropoutImpl(dropout)

  def forward(
    x: Tensor,
    paddingMask: Option[Tensor] = None,
    timeDiffs: Option[Tensor] = None
  ): Tensor = {
    val batchSize = x.size(0).toInt
    val seqLen = x.size(1).toInt
    val H = nHeads

    val xNormed = normIn.forward(x)

    val projOut = torch.silu(proj1.forward(xNormed))

    val q = projOut.narrow(2, 0, H * dqk).reshape(batchSize, seqLen, H, dqk).transpose(1, 2)
    val k = projOut.narrow(2, H * dqk, H * dqk).reshape(batchSize, seqLen, H, dqk).transpose(1, 2)
    val u = projOut.narrow(2, 2 * H * dqk, H * dv).reshape(batchSize, seqLen, H, dv)
    val v = projOut.narrow(2, 2 * H * dqk + H * dv, H * dv).reshape(batchSize, seqLen, H, dv).transpose(1, 2)

    val scores = torch.matmul(q, k.transpose(-2, -1)).mul(new Scalar(attnAlpha))

    val rabBias = rab.forward(timeDiffs, seqLen)
    val scoresWithBias = scores.add(rabBias)

    val causalMask = torch.tril(torch.ones(Array(seqLen.toLong, seqLen.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(x.device(),ScalarType.Long))
    val validMask = causalMask.unsqueeze(0).unsqueeze(0)

    val finalMask = if (paddingMask.isDefined) {
      val keyMask = paddingMask.get.unsqueeze(1).unsqueeze(1)
      torch.logical_and(validMask, keyMask)
    } else {
      validMask
    }

    val maskedScores = scoresWithBias.masked_fill(torch.eq(finalMask, new Scalar(0)), new Scalar(-1e4f))

    val attnWeights = torch.silu(maskedScores).div(new Scalar(maxSeqLen.toFloat))

    val attnOutput = torch.matmul(attnWeights, v)
    val attnOutputReshaped = attnOutput.transpose(1, 2).reshape(batchSize, seqLen, H * dv)
    val uFlat = u.reshape(batchSize, seqLen, H * dv)

    val gated = normAttn.forward(attnOutputReshaped).mul(uFlat)
    val dropped = dropoutLayer.forward(gated)
    proj2.forward(dropped)
  }
}

object HSTULayer {
  def apply(
    dModel: Int = 512,
    nHeads: Int = 8,
    dqk: Int = 64,
    dv: Int = 64,
    dropout: Float = 0.1f,
    maxSeqLen: Int = 200,
    numTimeBuckets: Int = 128,
    timeBucketFn: String = "sqrt",
    timeBucketDivisor: Float = 1.0f,
    timeBucketUnit: String = "minutes",
    device: String = DeviceSupport.backend
  ): HSTULayer = {
    new HSTULayer(dModel, nHeads, dqk, dv, dropout, maxSeqLen, numTimeBuckets, timeBucketFn, timeBucketDivisor, timeBucketUnit, device)
  }
}
