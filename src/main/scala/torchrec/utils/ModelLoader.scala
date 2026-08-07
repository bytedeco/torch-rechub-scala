package torchrec.utils

import org.bytedeco.javacpp.FloatPointer
import org.bytedeco.pytorch.global.torch.*
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
import torchrec.quantization.{Q8Quantizer, Q8Weight}

import java.io.*
import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import java.nio.{ByteBuffer, ByteOrder, FloatBuffer}

object ModelLoader {

  case class TensorMeta(dtype: String, shape: Array[Long], dataOffsets: Array[Long])

  /** Detect the compute dtype from a safetensors file header. */
  def detectComputeDtype(modelPath: String): ScalarType = {
    val dir = new File(modelPath)
    val file = dir.listFiles((_, n) => n.endsWith(".safetensors")).sortBy(_.getName).headOption
    file match {
      case Some(f) =>
        val raf = new RandomAccessFile(f, "r")
        val ch = raf.getChannel
        val headerSizeBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
        readFully(ch, headerSizeBuf)
        headerSizeBuf.flip()
        val headerSize = headerSizeBuf.getLong().toInt
        val headerBuf = ByteBuffer.allocate(headerSize).order(ByteOrder.LITTLE_ENDIAN)
        readFully(ch, headerBuf)
        headerBuf.flip()
        val headerJson = StandardCharsets.UTF_8.decode(headerBuf).toString
        ch.close()
        raf.close()
        val dominant = dominantDtype(headerJson)
        val st = dominant match {
          case "F16" | "float16" => ScalarType.Half
          case "BF16" | "bfloat16" => ScalarType.BFloat16
          case _ => ScalarType.Float
        }
        println(s"  ModelLoader: detected dtype=$dominant -> ScalarType=$st")
        st
      case None =>
        ScalarType.Float
    }
  }

  private def dominantDtype(headerJson: String): String = {
    val parsed = SimpleJson.parse(headerJson).asInstanceOf[Map[String, Any]]
    val counts = scala.collection.mutable.Map[String, Long]()
    for ((_, value) <- parsed) {
      value match {
        case m: Map[_, _] @unchecked =>
          val m2 = m.asInstanceOf[Map[String, Any]]
          m2.get("dtype") match {
            case Some(d: String) if m2.get("shape").exists(_.isInstanceOf[List[_]]) =>
              val shape = m2("shape").asInstanceOf[List[_]]
              val params = shape.map(_.asInstanceOf[Number].longValue()).product
              counts(d) = counts.getOrElse(d, 0L) + params
            case _ =>
          }
        case _ =>
      }
    }
    if (counts.isEmpty) "F32"
    else counts.maxBy(_._2)._1
  }

  def loadSafetensors(modelPath: String)(implicit config: Config): Map[String, Tensor] = {
    val dir = new File(modelPath)
    val result = scala.collection.mutable.Map[String, Tensor]()

    for (file <- dir.listFiles((_, n) => n.endsWith(".safetensors")).sortBy(_.getName)) {
      val tensors = loadFile(file)
      for ((name, tensor) <- tensors) {
        // Strip "thinker." prefix from Qwen3-ASR model keys so that
        // standard weight names (model.layers.X....) work throughout.
        val normalizedName = if (name.startsWith("thinker.")) {
          name.stripPrefix("thinker.")
        } else name
        result(normalizedName) = tensor
      }
    }

    result.toMap
  }

  private def loadFile(file: File)(implicit config: Config): Map[String, Tensor] = {
    val raf = new RandomAccessFile(file, "r")
    val ch = raf.getChannel

    val headerSizeBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
    readFully(ch, headerSizeBuf)
    headerSizeBuf.flip()
    val headerSize = headerSizeBuf.getLong().toInt

    val headerBuf = ByteBuffer.allocate(headerSize).order(ByteOrder.LITTLE_ENDIAN)
    readFully(ch, headerBuf)
    headerBuf.flip()
    val headerJson = StandardCharsets.UTF_8.decode(headerBuf).toString

    val meta = parseHeader(headerJson)
    val dataOffset = 8 + headerSize
    val result = scala.collection.mutable.Map[String, Tensor]()

    for ((name, tm) <- meta) {
      val start = dataOffset + tm.dataOffsets(0)
      val end = dataOffset + tm.dataOffsets(1)
      val size = (end - start).toInt
      val shape = tm.shape

      ch.position(start)
      val dataBuf = ByteBuffer.allocateDirect(size).order(ByteOrder.LITTLE_ENDIAN)
      readFully(ch, dataBuf)
      dataBuf.flip()
      require(dataBuf.remaining() == size, s"Short read for $name: got ${dataBuf.remaining()} of $size bytes")

      val numElements = shape.product.toInt
      // Create float tensor on CPU, then move to GPU as Half
      val cpuOpts = new TensorOptions()
        .dtype(new ScalarTypeOptional(ScalarType.Float))
        .device(new DeviceOptional(new Device("cpu")))
      val t = tm.dtype match {
        case "F32" | "float32" =>
          val arr = Array.ofDim[Float](numElements)
          for (i <- 0 until numElements) arr(i) = dataBuf.getFloat()
          tensor(arr, cpuOpts).clone().to(DeviceSupport.deviceOf(), config.computeDtype)

        case "F16" | "float16" =>
          val halfBuf = ByteBuffer.allocateDirect(size).order(ByteOrder.LITTLE_ENDIAN)
          dataBuf.reset()
          readFully(ch, halfBuf)
          halfBuf.flip()
          val arr = Array.ofDim[Float](numElements)
          for (i <- 0 until numElements) arr(i) = float16ToFloat(halfBuf.getShort())
          tensor(arr, cpuOpts).clone().to(DeviceSupport.deviceOf(), config.computeDtype)

        case "BF16" | "bfloat16" =>
          val arr = Array.ofDim[Float](numElements)
          val shortBuf = dataBuf.duplicate().order(ByteOrder.LITTLE_ENDIAN)
          for (i <- 0 until numElements) arr(i) = bfloat16ToFloat(shortBuf.getShort())
          tensor(arr, cpuOpts).clone().to(DeviceSupport.deviceOf(), config.computeDtype)

        case _ =>
          val arr = Array.ofDim[Float](numElements)
          for (i <- 0 until numElements) arr(i) = dataBuf.getFloat()
          tensor(arr, cpuOpts).clone().to(DeviceSupport.deviceOf(), config.computeDtype)
      }

      result(name) = t.reshape(shape*)
    }

    ch.close()
    raf.close()
    result.toMap
  }

  /** Fully fill the buffer, looping until no more bytes can be read (FileChannel.read may return partial counts). */
  private def readFully(ch: FileChannel, buf: ByteBuffer): Unit = {
    while (buf.hasRemaining) {
      val n = ch.read(buf)
      if (n < 0) return
    }
  }

  // Correct BF16 to Float32: BF16 is the high 16 bits of IEEE-754 float32
  private def bfloat16ToFloat(b: Short): Float = {
    java.lang.Float.intBitsToFloat((b & 0xFFFF) << 16)
  }

  // Standard IEEE-754 float16 to float32
  private def float16ToFloat(b: Short): Float = {
    val sign = (b >> 15) & 0x1
    val exp = (b >> 10) & 0x1f
    val frac = b & 0x3ff
    val signF = if (sign == 1) -1.0f else 1.0f
    if (exp == 0) signF * Math.scalb(frac / 1024.0f, -14).toFloat
    else if (exp == 31) if (frac == 0) Float.PositiveInfinity else Float.NaN
    else signF * Math.scalb(1.0f + frac / 1024.0f, exp - 15).toFloat
  }

  private def parseHeader(json: String): Map[String, TensorMeta] = {
    try {
      val parsed = SimpleJson.parse(json).asInstanceOf[Map[String, Any]]
      val result = scala.collection.mutable.Map[String, TensorMeta]()
      for ((key, value) <- parsed) {
        value match {
          case m: Map[_, _] @unchecked =>
            val m2 = m.asInstanceOf[Map[String, Any]]
            val dtype = m2.get("dtype") match {
              case Some(s: String) => s
              case _ => "F32"
            }
            val shape = m2.get("shape") match {
              case Some(lst: List[_] @unchecked) => lst.map(_.asInstanceOf[Number].longValue()).toArray
              case _ => Array.emptyLongArray
            }
            val offsets = m2.get("data_offsets") match {
              case Some(lst: List[_] @unchecked) => lst.map(_.asInstanceOf[Number].longValue()).toArray
              case _ => Array.emptyLongArray
            }
            if (shape.nonEmpty && offsets.length == 2) {
              result(key) = TensorMeta(dtype, shape, offsets)
            }
          case _ =>
        }
      }
      result.toMap
    } catch {
      case _: Throwable => Map.empty
    }
  }

  def loadModel(modelPath: String)(implicit config: Config): Map[String, Tensor] = {
    loadSafetensors(modelPath)
  }

  /**
   * Load model weights and quantize to Q8_0 format.
   * Returns a map of weight name -> Q8Weight (quantized in JVM memory).
   *
   * This reduces memory from ~2.65GB (BF16) to ~1.3GB for Q8.
   */
  def loadQ8Weights(modelPath: String)(implicit config: Config): Map[String, Q8Weight] = {
    val floatWeights = loadSafetensors(modelPath)
    val result = scala.collection.mutable.Map[String, Q8Weight]()

    var totalQ8Bytes = 0L
    var totalFloatBytes = 0L

    for ((name, t) <- floatWeights) {
      val dims = t.dim().toInt
      val numel = t.numel().toInt
      val shape = (0 until dims).map(d => t.size(d)).toArray.map(_.toLong)
      totalFloatBytes += numel * 4L

      // Read tensor data: move to CPU, clone, then batch-read with FloatPointer
      val cpu = t.to(new Device("cpu"), ScalarType.Float).clone()
      val data = Array.ofDim[Float](numel)
      try {
        val fp = new FloatPointer(cpu.storage().data())
        fp.get(data, 0, numel)
      } finally {
        cpu.close()
      }
      t.close()

      val (qbytes, scales) = Q8Quantizer.quantize(data)
      val qw = new Q8Weight(qbytes, scales, shape)
      result(name) = qw

      totalQ8Bytes += qbytes.length + scales.length * 4L

      // Clear sensitive data
      java.util.Arrays.fill(data, 0f)
    }

    val ratio = if (totalFloatBytes > 0) totalFloatBytes.toDouble / totalQ8Bytes else 1.0
    println(f"  ModelLoader Q8: ${result.size} tensors, " +
      f"FP32=${totalFloatBytes / 1e9}%.2fGB, Q8=${totalQ8Bytes / 1e9}%.2fGB, ratio=$ratio%.1f:1")

    result.toMap
  }
}
