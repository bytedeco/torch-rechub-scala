package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.utils.DeviceSupport

import scala.math

/**
 * HSTU rab^{p,t}: per-head bias on attention scores from (position-diff, time-diff) pairs,
 * following the HSTU paper (Eq. 3) and Meta's reference.
 * The bias is added to Q K^T before the SiLU/N activation.
 *
 * Parameters
 * ----------
 * nHeads : int
 *   Number of attention heads (per-head bias).
 * maxSeqLen : int
 *   Maximum sequence length. Sizes the position table to 2 * max_seq_len - 1 slots.
 * numTimeBuckets : int, default=128
 *   Number of time-difference buckets; an extra OOB slot is appended.
 * timeBucketFn : {'sqrt', 'log'}, default='sqrt'
 *   Bucketization function applied to |dt| in minutes.
 * timeBucketDivisor : float, default=1.0
 *   Divisor applied after sqrt/log.
 * timeBucketUnit : {'minutes', 'seconds'}, default='minutes'
 *   Unit used before bucketization.
 *
 * Shape
 * -----
 * Output: (1, n_heads, seq_len, seq_len) when timeDiffs is None
 *         (B, n_heads, seq_len, seq_len) when timeDiffs is given
 */
class RelativeBucketedTimeAndPositionBias(
  nHeads: Int,
  maxSeqLen: Int,
  numTimeBuckets: Int = 128,
  timeBucketFn: String = "sqrt",
  timeBucketDivisor: Float = 1.0f,
  timeBucketUnit: String = "minutes"
) extends Module {

  require(timeBucketFn == "sqrt" || timeBucketFn == "log", s"Unsupported time_bucket_fn: $timeBucketFn")
  require(timeBucketUnit == "minutes" || timeBucketUnit == "seconds", s"Unsupported time_bucket_unit: $timeBucketUnit")

  private val boundPos = math.sqrt(1.0 / (2 * maxSeqLen - 1)).toFloat
  private val posW = torch.rand(Array((2 * maxSeqLen - 1).toLong, nHeads.toLong),
    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  posW.uniform_(-boundPos, boundPos,new GeneratorOptional())
  register_parameter("pos_w", posW)

  private val boundTs = math.sqrt(1.0 / (numTimeBuckets + 1)).toFloat
  private val tsW = torch.rand(Array((numTimeBuckets + 1).toLong, nHeads.toLong),
    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  tsW.uniform_(-boundTs, boundTs,new GeneratorOptional())
  register_parameter("ts_w", tsW)

  private def bucketizeTime(dt: Tensor): Tensor = {
    // dt: absolute value of time differences
    val dtAbs = dt.abs()
    val dtScaled = if (timeBucketUnit == "minutes") {
      dtAbs.div(new Scalar(60.0f))
    } else {
      dtAbs
    }
    val dtClamped = dtScaled.clamp(min = new ScalarOptional(new Scalar(1e-6f)))

    val buckets = if (timeBucketFn == "sqrt") {
      dtClamped.sqrt()
    } else {
      dtClamped.log()
    }

    buckets.div(new Scalar(timeBucketDivisor))
      .clamp(min = new ScalarOptional(new Scalar(0)), max = new ScalarOptional(new Scalar(numTimeBuckets)))
      .toType(ScalarType.Long)
  }

  def forward(timeDiffs: Option[Tensor] = None, seqLen: Int = -1): Tensor = {
    val L = if (timeDiffs.isDefined) {
      timeDiffs.get.size(1).toInt
    } else {
      require(seqLen > 0, "Provide either time_diffs or seq_len.")
      require(seqLen <= maxSeqLen, s"seq_len ($seqLen) exceeds max_seq_len ($maxSeqLen)")
      seqLen
    }
    val device = if (timeDiffs.isDefined) timeDiffs.get.device() else posW.device()

    val positions = torch.arange(new  Scalar(L.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)).device(new DeviceOptional(new Device(device))))
    val relPosIdx = positions.unsqueeze(0).sub(positions.unsqueeze(1)).add(new Scalar(maxSeqLen - 1))
    val posBias = posW.index(new TensorIndexVector(new TensorIndex(relPosIdx))).permute(2, 0, 1)

    if (timeDiffs.isEmpty) {
      posBias.unsqueeze(0)
    } else {
      val td = timeDiffs.get
      val dtPairwise = td.unsqueeze(2).sub(td.unsqueeze(1))
      val timeBuckets = bucketizeTime(dtPairwise)
      val timeBias = tsW.index(new TensorIndexVector(new TensorIndex(timeBuckets)))
      val timeBiasPermuted = timeBias.permute(0, 3, 1, 2)
      posBias.unsqueeze(0).add(timeBiasPermuted)
    }
  }
}

/**
 * RelativeBucketedTimeAndPositionBias companion object with factory methods.
 */
object RelativeBucketedTimeAndPositionBias {
  def apply(
    nHeads: Int,
    maxSeqLen: Int,
    numTimeBuckets: Int = 128,
    timeBucketFn: String = "sqrt",
    timeBucketDivisor: Float = 1.0f,
    timeBucketUnit: String = "minutes"
  ): RelativeBucketedTimeAndPositionBias = {
    new RelativeBucketedTimeAndPositionBias(nHeads, maxSeqLen, numTimeBuckets, timeBucketFn, timeBucketDivisor, timeBucketUnit)
  }
}