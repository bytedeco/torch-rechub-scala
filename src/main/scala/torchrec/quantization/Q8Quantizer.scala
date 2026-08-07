package torchrec.quantization

import org.bytedeco.javacpp.FloatPointer
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
import org.bytedeco.pytorch.global.torch.*


/**
 * GGML-style Q8_0 quantization utilities.
 *
 * Q8_0 format: each block of Q8_BLOCK_SIZE elements is quantized as:
 *   - 32 x int8 values (quantized from float)
 *   - 1 x float32 scale factor
 *
 * Memory: 32 * 1 byte + 4 bytes = 36 bytes per 32 elements
 *   vs 32 * 4 bytes = 128 bytes in FP32
 *   Compression ratio: ~3.6x
 */
object Q8Quantizer {
  val Q8_BLOCK_SIZE = 32

  /**
   * Quantize a float array to Q8_0 format.
   * Returns (quantized_bytes, scales), where:
   *   - quantized_bytes: flat array of int8 values (already converted to signed byte)
   *   - scales: array of scale factors (one per block)
   */
  def quantize(data: Array[Float]): (Array[Byte], Array[Float]) = {
    val numBlocks = (data.length + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE
    val qbytes = Array.ofDim[Byte](numBlocks * Q8_BLOCK_SIZE)
    val scales = Array.ofDim[Float](numBlocks)

    var block = 0
    while (block < numBlocks) {
      val start = block * Q8_BLOCK_SIZE
      val end = Math.min(start + Q8_BLOCK_SIZE, data.length)
      val blockLen = end - start

      // Find max absolute value in block for scale
      var maxAbs = 0.0f
      var i = start
      while (i < end) {
        val abs = Math.abs(data(i))
        if (abs > maxAbs) maxAbs = abs
        i += 1
      }

      val scale = if (maxAbs > 0f) maxAbs / 127f else 1f
      scales(block) = scale
      val invScale = 127f / scale

      i = start
      while (i < end) {
        val qval = (data(i) * invScale).round.toInt.max(-128).min(127)
        qbytes(i) = qval.toByte
        i += 1
      }

      block += 1
    }

    (qbytes, scales)
  }

  /**
   * Dequantize Q8_0 data back to float array.
   */
  def dequantize(qbytes: Array[Byte], scales: Array[Float], outLen: Int): Array[Float] = {
    val result = Array.ofDim[Float](outLen)
    var block = 0
    while (block < scales.length) {
      val scale = scales(block)
      val start = block * Q8_BLOCK_SIZE
      val end = Math.min(start + Q8_BLOCK_SIZE, outLen)
      var i = start
      while (i < end) {
        result(i) = qbytes(i) * scale
        i += 1
      }
      block += 1
    }
    result
  }

  /**
   * Quantize a PyTorch tensor to Q8_0.
   * Returns (qbytes, scales, shape).
   */
  def quantizeTensor(t: Tensor): (Array[Byte], Array[Float], Array[Long]) = {
    val dims = t.dim().toInt
    val numel = t.numel().toInt
    val shape = (0 until dims).map(d => t.size(d)).toArray.map(_.toLong)

    // Move to CPU (Float) and clone to get an independent copy
    val cpu = t.to(new Device("cpu"), ScalarType.Float).clone()
    val data = Array.ofDim[Float](numel)
    try {
      // Use FloatPointer for batch read — single native memory copy, no per-element JNI
      val fp = new FloatPointer(cpu.storage().data())
      fp.get(data, 0, numel)
    } finally {
      cpu.close()
    }

    val (qbytes, scales) = quantize(data)
    (qbytes, scales, shape)
  }

  /**
   * Dequantize a Q8 weight to the specified dtype/device.
   */
  def dequantizeToTensor(qbytes: Array[Byte], scales: Array[Float], shape: Array[Long],
                         opts: TensorOptions): Tensor = {
    val floatData = dequantize(qbytes, scales, (shape.product).toInt)
    val flat = tensor(floatData, opts)
    flat.reshape(shape*)
  }
}

/**
 * Stores a weight matrix in Q8_0 format.
 * Memory-efficient: ~1 byte per parameter (vs 2 for FP16, 4 for FP32).
 */
class Q8Weight(
  val qbytes: Array[Byte],
  val scales: Array[Float],
  val shape: Array[Long],
) {
  def numRows: Long = shape(0)
  def numCols: Long = shape(1)

  def dequantize(opts: TensorOptions): Tensor = {
    Q8Quantizer.dequantizeToTensor(qbytes, scales, shape, opts)
  }
}
