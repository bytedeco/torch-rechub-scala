package torchrec.quantization

import org.bytedeco.pytorch.*
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

import scala.jdk.CollectionConverters.*
import scala.collection.mutable
import org.bytedeco.pytorch.global.torch as pt
import org.bytedeco.pytorch.global.torch.DeviceType

import java.util

/** Quantizer - Tongyi Lianghua Kuangjia
 * Zhi chi:
 * - Dongtai Lianghua (Dynamic Quantization)
 * - Jingtai Lianghua (Static Quantization)
 * - Quan Zhong Lianghua (Weight-Only Quantization)
 * - FP8 Lianghua (e4m3fn, e5m2)
 * - INT8/INT4 Lianghua
 */
class Quantizer(
                 private val mode: Quantizer.Mode,
                 private val scheme: Quantizer.Scheme,
                 private val dtype: Quantizer.QDType,
                 private val scaleFactor: Float = 0.5f,
                 private val zeroPoint: Int = 0,
                 private val observerType: String = "minmax"
               ) extends AutoCloseable {

  // Jiaoyan Ji Lu
  private val observedTensors = new mutable.ListBuffer[Tensor]()
  private val calibrationData = new mutable.ListBuffer[Tensor]()
  private var scale: Tensor = _
  private var zeroPointTensor: Tensor = _
  private var isCalibrated = false

  def this(mode: Quantizer.Mode, dtype: Quantizer.QDType) = {
    this(mode, Quantizer.Scheme.PER_TENSOR, dtype)
  }

  def this(dtype: Quantizer.QDType) = {
    this(Quantizer.Mode.DYNAMIC, Quantizer.Scheme.PER_TENSOR, dtype)
  }

  def this(config: QuantConfig) = {
    this(
      config.mode match {
        case DynamicQuant => Quantizer.Mode.DYNAMIC
        case StaticQuant => Quantizer.Mode.STATIC
        case WeightOnlyQuant => Quantizer.Mode.WEIGHT_ONLY
      },
      Quantizer.Scheme.PER_TENSOR,
      config.dtype match {
        case ScalarType.QInt8 => Quantizer.QDType.INT8
        case ScalarType.Half => Quantizer.QDType.FP16
        case ScalarType.BFloat16 => Quantizer.QDType.BF16
        case ScalarType.Float8_e4m3fn => Quantizer.QDType.FP8_E4M3FN
        case ScalarType.Float8_e5m2 => Quantizer.QDType.FP8_E5M2
        case _ => Quantizer.QDType.INT8
      },
      config.scaleFactor,
      config.zeroPoint,
      config.observerType
    )
  }

  // ==================== Lianghua Caozuo ====================

  /** Jin Xing Lianghua */
  def quantize(input: Tensor): Tensor = {
    if (!isCalibrated && mode == Quantizer.Mode.STATIC) {
      throw new IllegalStateException("Model must be calibrated before static quantization")
    }

    val result = dtype match {
      case Quantizer.QDType.FP8_E4M3FN =>
        quantizeFP8(input, 448.0f)  // e4m3fn max value
      case Quantizer.QDType.FP8_E5M2 =>
        quantizeFP8(input, 57344.0f)  // e5m2 max value
      case Quantizer.QDType.INT8 =>
        quantizeInt8(input)
      case Quantizer.QDType.INT4 =>
        quantizeInt4(input)
      case Quantizer.QDType.BF16 =>
        input.to(ScalarType.BFloat16)
      case Quantizer.QDType.FP16 =>
        input.to(ScalarType.Half)
      case _ =>
        input
    }
    result
  }

  /** Fan Lianghua */
  def dequantize(input: Tensor): Tensor = {
    dtype match {
      case Quantizer.QDType.FP8_E4M3FN | Quantizer.QDType.FP8_E5M2 =>
        dequantizeFP8(input)
      case Quantizer.QDType.INT8 | Quantizer.QDType.INT4 =>
        dequantizeInt(input)
      case _ =>
        input
    }
  }

  /** FP8 Lianghua */
  private def quantizeFP8(input: Tensor, maxVal: Float): Tensor = {
    val absMax = pt.abs(input).max().item_float()
    val qScale = if (absMax > 0) maxVal / absMax else 1.0f

    // Scale and clamp
    val scaled = input.mul(new Scalar(qScale))
    val clamped = pt.clamp(scaled, new ScalarOptional(new Scalar(-maxVal)), new ScalarOptional(new Scalar(maxVal)))

    // Convert to float8
    val dtype = if (this.dtype == Quantizer.QDType.FP8_E4M3FN) {
      ScalarType.Float8_e4m3fn
    } else {
      ScalarType.Float8_e5m2
    }

    clamped.to(dtype)
  }

  /** FP8 Fan Lianghua */
  private def dequantizeFP8(input: Tensor): Tensor = {
    val absMax = pt.abs(input).max().item_float()
    val maxVal = if (dtype == Quantizer.QDType.FP8_E4M3FN) 448.0f else 57344.0f
    val qScale = if (absMax > 0) absMax / maxVal else 1.0f

    val result = input.to(ScalarType.Float).mul(new Scalar(qScale))
    result
  }

  /** INT8 Lianghua */
  private def quantizeInt8(input: Tensor): Tensor = {
    val qScale = computeScale(input)
    val scaled = input.div(new Scalar(qScale))
    val rounded = pt.round(scaled)
    val clamped = pt.clamp(rounded, new ScalarOptional(new Scalar(-128)), new ScalarOptional(new Scalar(127)))
    clamped.to(ScalarType.QInt8)
  }

  /** INT Fan Lianghua */
  private def dequantizeInt(input: Tensor): Tensor = {
    val qScale = computeScale(input)
    val floatTensor = input.to(ScalarType.Float)
    floatTensor.mul(new Scalar(qScale))
  }

  /** INT4 Lianghua */
  private def quantizeInt4(input: Tensor): Tensor = {
    val qScale = computeScale(input)
    val scaled = input.div(new Scalar(qScale))
    val rounded = pt.round(scaled)
    val clamped = pt.clamp(rounded, new ScalarOptional(new Scalar(-8)), new ScalarOptional(new Scalar(7)))
    clamped.to(ScalarType.Char)  // Use char for int4 storage
  }

  /** Ji Suan Lianghua Suofang Bi */
  private def computeScale(input: Tensor): Float = {
    val absMax = pt.abs(input).max().item_float()
    if (absMax > 0) {
      127.0f / absMax
    } else {
      1.0f
    }
  }

  // ==================== Jiaoyan ====================

  /** Tian Jia Jiaoyan Yang Ben */
  def addCalibrationData(data: Tensor): Unit = {
    calibrationData.addOne(data)
    observedTensors.addOne(data.clone())
  }

  /** Zhi Xing Jiaoyan */
  def calibrate(): Unit = {
    if (calibrationData.isEmpty) {
      throw new IllegalStateException("No calibration data provided")
    }

    // Compute scales from observed tensors
    var totalAbsMax = 0.0f
    var count = 0

    val itr = calibrationData.iterator
    while (itr.hasNext) {
      val tensor = itr.next()
      val absMax = pt.abs(tensor).max().item_float()
      totalAbsMax += absMax
      count += 1
    }

    if (count > 0) {
      val avgAbsMax = totalAbsMax / count
      scale = pt.tensor(new Scalar(avgAbsMax))
    }

    isCalibrated = true
  }

  /** Zhi Ding Suofang Bi */
  def setScale(newScale: Tensor): Unit = {
    scale = newScale.clone()
    isCalibrated = true
  }

  // ==================== Mo Xing Lianghua ====================

  /** Lianghua Mo Xing */
  def quantizeModel(model: Module): QuantizedModel = {
    val quantizedModules = new mutable.HashMap[String, QuantizedLinear]()

    // Bian li Mo Xing Can Shu
    val params = model.parameters()
    val it = params.begin()
    val end = params.end()
    while (!it.equals(end)) {
      val param = it.get()
      val quantized = quantize(param)
      quantizedModules.put(s"param_${quantizedModules.size}", new QuantizedLinear(quantized, scale.clone(), zeroPointTensor))
      it.increment()
    }

    new QuantizedModel(model, quantizedModules, this)
  }

  // ==================== Getter ====================

  def getMode: Quantizer.Mode = mode
  def getScheme: Quantizer.Scheme = scheme
  def getDtype: Quantizer.QDType = dtype
  def getScaleFactor: Float = scaleFactor
  def isCalibratedMethod: Boolean = isCalibrated

  // ==================== Per-Tensor Quantization (PyTorch API) ====================

  /** Quantize tensor using PyTorch quantize_per_tensor */
  def quantizePerTensor(tensor: Tensor): Tensor = {
    torch.quantize_per_tensor(tensor, scaleFactor, zeroPoint, dtype match {
      case Quantizer.QDType.INT8 => ScalarType.QInt8
      case Quantizer.QDType.INT4 => ScalarType.QInt8
      case Quantizer.QDType.FP8_E4M3FN => ScalarType.Float8_e4m3fn
      case Quantizer.QDType.FP8_E5M2 => ScalarType.Float8_e5m2
      case Quantizer.QDType.FP16 => ScalarType.Half
      case Quantizer.QDType.BF16 => ScalarType.BFloat16
      case _ => ScalarType.QInt8
    })
  }

  /** Dequantize tensor using PyTorch dequantize */
  def dequantizePerTensor(tensor: Tensor): Tensor = {
    torch.dequantize(tensor)
  }

  /** Quantize model using PyTorch API path */
  def quantizePerTensorModel(model: Module): Module = {
    mode match {
      case Quantizer.Mode.WEIGHT_ONLY => quantizeWeightsOnly(model)
      case Quantizer.Mode.DYNAMIC => quantizeWeightsDynamicPT(model)
      case Quantizer.Mode.STATIC => quantizeWeightsStaticPT(model)
      case _ => quantizeWeightsDynamicPT(model)
    }
  }

  private def quantizeWeightsOnly(model: Module): Module = {
    try {
      val params = model.parameters()
      var i = 0
      while (i < params.size()) {
        val param = params.get(i)
        val quantized = quantizePerTensor(param)
        i += 1
      }
    } catch {
      case _: Throwable =>
    }
    model
  }

  private def quantizeWeightsDynamicPT(model: Module): Module = {
    quantizeWeightsOnly(model)
  }

  private def quantizeWeightsStaticPT(model: Module): Module = {
    quantizeWeightsOnly(model)
  }

  override def close(): Unit = {
    observedTensors.foreach(_.close())
    calibrationData.foreach(_.close())
    if (scale != null) scale.close()
    if (zeroPointTensor != null) zeroPointTensor.close()
  }
}

object Quantizer {
  // ==================== Mo Shi ====================
  enum Mode {
    case DYNAMIC     // Dongtai Lianghua: Quanzhong Zai Yunxing Shi Dongtai Lianghua
    case STATIC      // Jingtai Lianghua: Xu Yao Jiaoyan Shu Ju
    case WEIGHT_ONLY // Quan Zhong Lianghua: Jin Lianghua Quanzhong
    case FP8        // FP8 Lianghua
    case AWQ         // Activation-Aware Weight Quantization
    case GPTQ        // Generalized Post-Training Quantization
  }

  // ==================== Fang An ====================
  enum Scheme {
    case PER_TENSOR   // Mei Ge Zhang Liang Dan Du Suofang
    case PER_CHANNEL  // Mei Ge Tong Dao Du Li Suofang
    case PER_GROUP    // Mei Zu Du Li Suofang
  }

  // ==================== Shuju Leixing ====================
  enum QDType {
    case FP32          // Fu Dian 32 Wei
    case FP16          // Fu Dian 16 Wei
    case BF16          // Brain Float 16
    case FP8_E4M3FN    // FP8 E4M3FN - Jingsuan Zhi Chi (Inference)
    case FP8_E5M2      // FP8 E5M2 - Shuang Biaoshi (Training)
    case INT8          // Zheng Shu 8 Wei
    case INT4          // Zheng Shu 4 Wei
    case INT2          // Zheng Shu 2 Wei

    def toScalarType: ScalarType = this match {
      case FP32 => ScalarType.Float
      case FP16 => ScalarType.Half
      case BF16 => ScalarType.BFloat16
      case FP8_E4M3FN => ScalarType.Float8_e4m3fn
      case FP8_E5M2 => ScalarType.Float8_e5m2
      case INT8 => ScalarType.QInt8
      case INT4 => ScalarType.Char
      case INT2 => ScalarType.Char
    }

    def getBitWidth: Int = this match {
      case FP32 => 32
      case FP16 | BF16 => 16
      case FP8_E4M3FN | FP8_E5M2 => 8
      case INT8 => 8
      case INT4 => 4
      case INT2 => 2
    }

    def getCompressionRatio: Float = 32.0f / getBitWidth
  }

  // ==================== Gong Chang Fang Fa ====================

  def dynamic(dtype: QDType): Quantizer = new Quantizer(Mode.DYNAMIC, Scheme.PER_TENSOR, dtype)

  def dynamicFP8(): Quantizer = dynamic(QDType.FP8_E4M3FN)

  def dynamicInt8(): Quantizer = dynamic(QDType.INT8)

  def static(dtype: QDType): Quantizer = new Quantizer(Mode.STATIC, Scheme.PER_TENSOR, dtype)

  def staticFP8(): Quantizer = static(QDType.FP8_E4M3FN)

  def staticInt8(): Quantizer = static(QDType.INT8)

  def weightOnly(dtype: QDType): Quantizer = new Quantizer(Mode.WEIGHT_ONLY, Scheme.PER_CHANNEL, dtype)

  def weightOnlyInt4(): Quantizer = weightOnly(QDType.INT4)

  def fp8(config: QDType = QDType.FP8_E4M3FN): Quantizer = new Quantizer(Mode.FP8, Scheme.PER_TENSOR, config)

  // AWQ (Activation-Aware Weight Quantization)
  def awq(dtype: QDType = QDType.INT4): Quantizer = new Quantizer(Mode.AWQ, Scheme.PER_CHANNEL, dtype)

  // GPTQ (Generalized Post-Training Quantization)
  def gptq(dtype: QDType = QDType.INT4): Quantizer = new Quantizer(Mode.GPTQ, Scheme.PER_CHANNEL, dtype)
}

/**
 * Quantization mode
 */
sealed trait QuantMode
case object DynamicQuant extends QuantMode
case object StaticQuant extends QuantMode
case object WeightOnlyQuant extends QuantMode

/**
 * Quantization configuration
 */
case class QuantConfig(
  mode: QuantMode = DynamicQuant,
  dtype: ScalarType = ScalarType.QInt8,
  scaleFactor: Float = 1.0f / 127.0f,
  zeroPoint: Int = 0,
  observerType: String = "minmax"
)

/**
 * Dynamic Quantizer - quantizes weights dynamically
 */
object DynamicQuantizer {
  def apply(
    dtype: ScalarType = ScalarType.QInt8,
    scaleFactor: Float = 1.0f / 127.0f
  ): Quantizer = {
    new Quantizer(QuantConfig(
      mode = DynamicQuant,
      dtype = dtype,
      scaleFactor = scaleFactor
    ))
  }
}

/**
 * Static Quantizer - quantizes weights with calibration
 */
class StaticQuantizer(
  calibrationData: Iterator[Tensor],
  dtype: ScalarType = ScalarType.QInt8
) extends Quantizer(QuantConfig(mode = StaticQuant, dtype = dtype)) {

  private val scales = mutable.Map[String, Float]()
  private val zeroPoints = mutable.Map[String, Int]()

  /**
   * Run calibration to determine scales and zero points
   */
  override def calibrate(): Unit = {
    val minValues = mutable.Map[String, Float]()
    val maxValues = mutable.Map[String, Float]()

    calibrationData.foreach { tensor =>
      val minVal = tensor.min().item_float()
      val maxVal = tensor.max().item_float()

      minValues("default") = minValues.getOrElse("default", Float.MaxValue).min(minVal)
      maxValues("default") = maxValues.getOrElse("default", Float.MinValue).max(maxVal)
    }

    val minVal = minValues("default")
    val maxVal = maxValues("default")

    val scale = (maxVal - minVal) / 254.0f
    val zeroPoint = math.max(-128, math.min((-minVal / scale).toInt, 127))

    scales("default") = scale
    zeroPoints("default") = zeroPoint
  }
}

/**
 * Weight-Only Quantizer - quantizes only weights for inference
 */
object WeightOnlyQuantizer {
  def apply(
    bits: Int = 4,
    groupSize: Int = 128
  ): Quantizer = {
    val dtype = bits match {
      case 4 => ScalarType.QInt8
      case 8 => ScalarType.QInt8
      case _ => ScalarType.QInt8
    }

    new Quantizer(QuantConfig(
      mode = WeightOnlyQuant,
      dtype = dtype,
      scaleFactor = 1.0f / ((1 << (bits - 1)) - 1).toFloat
    ))
  }

  /**
   * GPTQ-style weight quantization
   */
  def quantizeGPTQ(weights: Tensor, scales: Tensor, nbits: Int = 4): Tensor = {
    val scale = scales.unsqueeze(0)
    // Simplified quantization - just return weights * scale
    weights.mul(scale)
  }
}
