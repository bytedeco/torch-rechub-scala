package torchrec.quantization

import org.bytedeco.pytorch.global.torch as pt
import org.bytedeco.pytorch.global.torch.ScalarType
import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.{Device, *}

/** QuantizedLinear - Lianghua Hou De Xian Xing Ceng
 * Cóng PyTorch QuantizedLinear Mo Xing Yin She
 */
class QuantizedLinear(
    private val weight: Tensor,
    private val scale: Tensor,
    private val zeroPoint: Tensor,
    private var bias: Tensor = null.asInstanceOf[Tensor]
) extends Module {

  private val inputScale: Tensor = pt.tensor(new Scalar(1.0f))
  private var quantizationScheme: torchrec.quantization.Quantizer.Scheme = torchrec.quantization.Quantizer.Scheme.PER_TENSOR

  def this(weight: Tensor, scale: Tensor, zeroPoint: Int) = {
    this(weight, scale, pt.tensor(new Scalar(zeroPoint)))
  }

  // ==================== Qian Xiang Chuan Di ====================

  override def forward(input: Tensor): Tensor = {
    // Kuai Su Lian Jie Mo Xing: Y = X @ W^T + b
    val inputFlat = input.reshape(Array(-1, input.size(input.dim - 1))*)

    // Lianghua Shuru
    val quantizedInput = quantizeInput(inputFlat)
    val dequantizedInput = dequantizeInput(quantizedInput)

    // Jisuann: Y = X @ W^T
    val result = pt.addmm(bias, dequantizedInput, weight.t())

    // Bao Cun Yuan Shi Shape
    val shape = input.shape
    val output = result.reshape(shape*)

    quantizedInput.close()
    inputFlat.close()

    output
  }

  def apply(input: Tensor): Tensor = forward(input)

  // ==================== Lianghua ====================

  private def quantizeInput(input: Tensor): Tensor = {
    // Mei Ge Hang Du Li Suofang
    val absMaxPerRow = pt.abs(input).max(1L)
    val maxValues = absMaxPerRow.get0
    val dynamicScale = maxValues.reshape(input.size(0), 1).div(new Scalar(127.0f))

    // Suofang He Quan Hua
    val scaled = input.div(dynamicScale)
    val clamped = pt.clamp(scaled, new ScalarOptional(new Scalar(-127.0f)), new ScalarOptional(new Scalar(127.0f)))
    val rounded = pt.round(clamped)

    val quantized = rounded.to(ScalarType.QInt8)

    scaled.close()
    clamped.close()

    quantized
  }

  private def dequantizeInput(input: Tensor): Tensor = {
    // Fan Quan Hua
    val floatInput = input.to(ScalarType.Float)
    val absMaxPerRow = pt.abs(floatInput).max(1L)
    val maxValues = absMaxPerRow.get0
    val dynamicScale = maxValues.reshape(input.size(0), 1).div(new Scalar(127.0f))

    val dequantized = floatInput.mul(dynamicScale)

    floatInput.close()

    dequantized
  }

  // ==================== Jineig Canshu ====================

  def getWeight: Tensor = weight
  def getScale: Tensor = scale
  def getZeroPoint: Tensor = zeroPoint
  def getBias: Tensor = bias

  def setBias(b: Tensor): Unit = { bias = b.clone() }

  def setQuantizationScheme(scheme: torchrec.quantization.Quantizer.Scheme): Unit = {
    quantizationScheme = scheme
  }

  // ==================== Tongji ====================

  def getNumParameters: Long = {
    var count = 0L
    if (weight != null && !weight.isNull) count += weight.numel
    if (bias != null && !bias.isNull) count += bias.numel
    count
  }

  def getMemorySizeBytes: Long = {
    var size = 0L
    if (weight != null && !weight.isNull) size += weight.numel * weight.element_size
    if (scale != null && !scale.isNull) size += scale.numel * scale.element_size
    if (zeroPoint != null && !zeroPoint.isNull) size += zeroPoint.numel * zeroPoint.element_size
    if (bias != null && !bias.isNull) size += bias.numel * bias.element_size
    size
  }

  override def close(): Unit = {
    inputScale.close()
  }

  override def toString: String = s"QuantizedLinear(weight=${if(weight != null) weight.numel else 0}, scale=${if(scale != null) scale.numel else 0})"
}

/** QuantizedModel - Lianghua Hou De Wan Zheng Mo Xing */
class QuantizedModel(
    private val originalModel: Module,
    private val quantizedModules: scala.collection.mutable.Map[String, QuantizedLinear],
    private val quantizer: torchrec.quantization.Quantizer
) extends Module {

  private val fp32Model: Module = originalModel.clone(null)

  override def forward(input: Tensor): Tensor = {
    // Xuan Ze Lianghua Huo FP32 Mo Xing
    if (quantizer.getMode == torchrec.quantization.Quantizer.Mode.DYNAMIC) {
      // Dongtai Lianghua: Shiyong Lianghua Hou De Jisuann
      quantizeAndCompute(input)
    } else {
      // Jingtai Lianghua: Zhi Jie Shiyong FP32 (passthrough for now)
      input
    }
  }

  def apply(input: Tensor): Tensor = forward(input)

  private def quantizeAndCompute(input: Tensor): Tensor = {
    // Lianghua Shuru
    val quantizedInput = quantizer.quantize(input)
    val dequantizedInput = quantizer.dequantize(quantizedInput)

    // Passthrough for now (full model not accessible via JavaCPP)
    val output = input

    quantizedInput.close()
    dequantizedInput.close()

    output
  }

  // ==================== Mo Xing Guanli ====================

  def getOriginalModel: Module = originalModel
  def getFP32Model: Module = fp32Model
  def getQuantizedModules: collection.mutable.Map[String, QuantizedLinear] = quantizedModules

  def getQuantizedLinear(name: String): Option[QuantizedLinear] = {
    quantizedModules.get(name)
  }

  def addQuantizedLinear(name: String, linear: QuantizedLinear): Unit = {
    quantizedModules.put(name, linear)
  }

  // ==================== Canshu Tongji ====================

  def getTotalParameters: Long = {
    var count = 0L
    val itr = quantizedModules.values.iterator
    while (itr.hasNext) {
      count += itr.next().getNumParameters
    }
    count
  }

  def getOriginalMemorySizeBytes: Long = {
    var size = 0L
    val params = fp32Model.parameters()
    val it = params.begin()
    val end = params.end()
    while (!it.equals(end)) {
      val p = it.get()
      if (p != null && !p.isNull) size += p.numel * p.element_size
      it.increment()
    }
    size
  }

  def getQuantizedMemorySizeBytes: Long = {
    var size = 0L
    val itr = quantizedModules.values.iterator
    while (itr.hasNext) {
      size += itr.next().getMemorySizeBytes
    }
    size
  }

  def getCompressionRatio: Float = {
    val originalSize = getOriginalMemorySizeBytes
    val quantizedSize = getQuantizedMemorySizeBytes
    if (quantizedSize > 0) originalSize.toFloat / quantizedSize else 1.0f
  }

  // ==================== Tidu Tongbu (DDP Fuzhi) ====================

  def averageParameters(): Unit = {
    val params = fp32Model.parameters()
    val it = params.begin()
    val end = params.end()
    while (!it.equals(end)) {
      val p = it.get()
      if (p != null && !p.isNull && p.grad != null && p.grad.defined()) {
        // Tongguo AllReduce Pingjun Tidu
        // Xu Yao Pei He ProcessGroupShiyong
      }
      it.increment()
    }
  }

  override def close(): Unit = {
    quantizedModules.values.foreach(_.close())
    fp32Model.close()
  }

  override def toString: String = {
    s"QuantizedModel(originalParams=${getTotalParameters}, " +
    s"compressed=${getCompressionRatio}x, " +
    s"mode=${quantizer.getMode})"
  }
}

/** QuantizedConv2d - Lianghua Hou De Juanji Ceng */
class QuantizedConv2d(
    private val weight: Tensor,
    private val scale: Tensor,
    private val zeroPoint: Tensor,
    private val stride: Array[Int] = Array(1, 1),
    private val padding: Array[Int] = Array(0, 0),
    private val dilation: Array[Int] = Array(1, 1),
    private var bias: Tensor = null.asInstanceOf[Tensor]
) extends AutoCloseable {

  def forward(input: Tensor): Tensor = {
    // Shi Yong Conv2d Jisuann
    // Note: Xu Yao Shixian Zhen Zheng De Juanji Lianghua
    // Zhe Li Shi Yige Jichu Shixian
    val output = pt.conv2d(input, weight)
    output
  }

  def getWeight: Tensor = weight
  def getScale: Tensor = scale
  def getMemorySizeBytes: Long = {
    var size = 0L
    if (weight != null && !weight.isNull) size += weight.numel * weight.element_size
    if (scale != null && !scale.isNull) size += scale.numel * scale.element_size
    if (bias != null && !bias.isNull) size += bias.numel * bias.element_size
    size
  }

  override def close(): Unit = { }

  override def toString: String = s"QuantizedConv2d(stride=${stride.mkString(",")})"
}
