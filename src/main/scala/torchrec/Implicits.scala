package torchrec

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import org.bytedeco.javacpp.{FloatPointer, LongPointer}

import scala.language.implicitConversions

object Implicits {
  def tensor(data: Array[Float], sizes: Array[Long]): Tensor = {
    val flat = torch.tensor(data, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    if (sizes.length == 1 && sizes(0) == data.length) flat
    else flat.reshape(sizes: _*)
  }

  def zeros(sizes: Array[Long]): Tensor =
    torch.zeros(sizes, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  def ones(sizes: Array[Long]): Tensor =
    torch.ones(sizes, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  def randn(sizes: Array[Long]): Tensor =
    torch.randn(sizes, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  def rand(sizes: Array[Long]): Tensor =
    torch.rand(sizes, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))

  def longTensor(data: Array[Long]): Tensor = {
    // Create a Long tensor directly to avoid dtype upcasting during batching/stacking.
    // Constructing as Long prevents intermittent CPUFloatType indices seen in some
    // embedding/index_select calls when tensors are mixed-typed during stacking.
    val t = torch.tensor(data, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
    // Ensure contiguous layout
    if (t.is_contiguous()) t else t.contiguous()
  }

  def floatTensor(data: Array[Float]): Tensor =
    torch.tensor(data, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))

  def toTensorVector(tensors: Seq[Tensor]): TensorVector = {
    val vec = new TensorVector(tensors.size.toLong)
    tensors.foreach(vec.push_back)
    vec
  }

  def toParameterVector(params: Seq[Tensor]): TensorVector = {
    val vec = new TensorVector(params.size.toLong)
    params.foreach(vec.push_back)
    vec
  }

  def cpu(): String = "cpu"
  def mps(): String = "mps"
  def cuda(device: Int = 0): String = s"cuda:$device"

  implicit class RichTensor(val tensor: Tensor) extends AnyVal {
    def shape: Array[Long] = tensor.shape()
    def ndim: Int = tensor.dim().toInt
    def numel: Long = tensor.numel()

    def clone(): Tensor = tensor.clone()
    def detach(): Tensor = tensor.detach()

    def to(device: String): Tensor = tensor.to(new Device(device), tensor.dtype())

    def toType(dtype: ScalarType): Tensor = tensor.toType(dtype)

    def mean(): Tensor = tensor.mean()
    def sum(): Tensor = tensor.sum()

    /**
     * Safely extract a scalar value from a tensor, handling GPU tensors properly.
     * Automatically moves the tensor to CPU if needed before calling .item()
     */
    def itemSafe(): Double = {
      val cpuTensor = try {
        // Try to move to CPU if on CUDA
        if (tensor.is_cuda()) {
          tensor.cpu()
        } else {
          tensor
        }
      } catch {
        case _: Exception =>
          // If that fails, try using .to() with explicit device
          try {
            tensor.to(new Device("cpu"), tensor.dtype())
          } catch {
            case _: Exception => tensor
          }
      }
      cpuTensor.item().toDouble
    }

    def toFloat: Float = {
      val cpuTensor = try {
        if (tensor.is_cuda()) {
          tensor.cpu()
        } else {
          tensor
        }
      } catch {
        case _: Exception =>
          try {
            tensor.to(new Device("cpu"), tensor.dtype())
          } catch {
            case _: Exception => tensor
          }
      }
      cpuTensor.item().toFloat
    }

    def toFloatArray: Array[Float] = {
      // Move to CPU if on a different device (e.g., CUDA)
      val cpuTensor = try {
        if (tensor.is_cuda()) {
          tensor.cpu()
        } else {
          tensor
        }
      } catch {
        case _: Exception =>
          try {
            tensor.to(new Device("cpu"), tensor.dtype())
          } catch {
            case _: Exception => tensor
          }
      }

      val contig = if (cpuTensor.is_contiguous()) cpuTensor else cpuTensor.contiguous()
      val size = contig.numel().toInt
      if (size == 0) return Array.empty[Float]
      try {
        val ptr = contig.data_ptr()
        val buf = ptr.asByteBuffer()
        buf.limit(size * 4)
        val floatBuf = buf.asFloatBuffer()
        val result = new Array[Float](size)
        var i = 0
        while (i < size) {
          result(i) = floatBuf.get(i)
          i += 1
        }
        result
      } catch {
        case _: Exception =>
          // Fallback: reshape to 1-D and read scalars
          val flat = contig.reshape(size.toLong)
          val result = new Array[Float](size)
          var i = 0
          while (i < size) {
            result(i) = flat.select(0, i).itemSafe().toFloat
            i += 1
          }
          result
      }
    }

    def toLongArray: Array[Long] = {
      // Move to CPU if on a different device (e.g., CUDA)
      val cpuTensor = try {
        if (tensor.is_cuda()) {
          tensor.cpu()
        } else {
          tensor
        }
      } catch {
        case _: Exception =>
          try {
            tensor.to(new Device("cpu"), tensor.dtype())
          } catch {
            case _: Exception => tensor
          }
      }

      val contig = if (cpuTensor.is_contiguous()) cpuTensor else cpuTensor.contiguous()
      val size = contig.numel().toInt
      if (size == 0) return Array.empty[Long]
      try {
        val ptr = contig.data_ptr()
        val buf = ptr.asByteBuffer()
        buf.limit(size * 8)
        val longBuf = buf.asLongBuffer()
        val result = new Array[Long](size)
        var i = 0
        while (i < size) {
          result(i) = longBuf.get(i)
          i += 1
        }
        result
      } catch {
        case _: Exception =>
          // Fallback: reshape to 1-D and read scalars
          val flat = contig.reshape(size.toLong)
          val result = new Array[Long](size)
          var i = 0
          while (i < size) {
            result(i) = flat.select(0, i).itemSafe().toLong
            i += 1
          }
          result
      }
    }
  }

  implicit class SeqTensorRichSeq(val tensors: Seq[Tensor]) extends AnyVal {
    def toTensorVector: TensorVector = {
      val vec = new TensorVector()
      tensors.foreach(vec.push_back)
      vec
    }

    def cat(dim: Int = 0): Tensor = torch.cat(toTensorVector, dim)
    def stack(dim: Int = 0): Tensor = torch.stack(toTensorVector, dim)
  }
}

type TRTensor = Tensor
type TRModule = Module
type TRScalarType = ScalarType
