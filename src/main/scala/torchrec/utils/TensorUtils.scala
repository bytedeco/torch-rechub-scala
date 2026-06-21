package torchrec.utils

import org.bytedeco.pytorch.global.torch.*
import org.bytedeco.pytorch.{Device, Tensor}
import torchrec.Implicits.tensor
object TensorUtils {
  private val cpuDevice = new Device("cpu")
  private val cudaDevice = new Device("cuda")
  private val mpsDevice = new Device("mps")
  
  def toCpu(t: Tensor): Tensor =
    if (t == null) null.asInstanceOf[Tensor] else t.to(cpuDevice, t.scalar_type())

  def toCuda(t: Tensor): Tensor =
    if (t == null) null.asInstanceOf[Tensor] else t.to(cudaDevice, t.scalar_type())

  def toMps(t: Tensor): Tensor =
    if (t == null) null.asInstanceOf[Tensor] else t.to(mpsDevice, t.scalar_type())

  def readLongs(t: Tensor): Array[Long] = {
    if (t == null) return Array.emptyLongArray
    val flat = t.reshape(-1L)
    val cpu = flat.to(cpuDevice, ScalarType.Long)
    try {
      Array.tabulate(cpu.numel().toInt)(i => cpu.data().get(i).item().toLong())
    } finally {
      cpu.close()
      flat.close()
    }
  }

  def readInts(t: Tensor): Array[Int] = readLongs(t).map(_.toInt)

  def readFloats(t: Tensor): Array[Float] = {
    if (t == null) return Array.emptyFloatArray
    val flat = t.reshape(-1L)
    val cpu = flat.to(cpuDevice, ScalarType.Float)
    try {
      Array.tabulate(cpu.numel().toInt)(i => cpu.data().get(i).item_float())
    } finally {
      cpu.close()
      flat.close()
    }
  }

  /** Summarize a tensor for debugging: shape, nan/inf counts, min/max/mean, and a few samples. */
  def stats(label: String, t: Tensor): String = {
    if (t == null) return s"$label=null"
    val shape = (0 until t.dim().toInt).map(d => t.size(d)).mkString("[", ",", "]")
    val values = readFloats(t)
    if (values.isEmpty) return s"$label shape=$shape EMPTY"
    var nan = 0
    var inf = 0
    var minV = Float.PositiveInfinity
    var maxV = Float.NegativeInfinity
    var sum = 0.0
    var i = 0
    while (i < values.length) {
      val v = values(i)
      if (v.isNaN) nan += 1
      else if (v.isInfinite) inf += 1
      else {
        if (v < minV) minV = v
        if (v > maxV) maxV = v
        sum += v
      }
      i += 1
    }
    val finite = values.length - nan - inf
    val mean = if (finite > 0) sum / finite else Double.NaN
    val sample = values.take(5).map(v => f"$v%.4f").mkString(",")
    s"$label shape=$shape n=${values.length} nan=$nan inf=$inf min=$minV max=$maxV mean=$mean sample=[$sample]"
  }

  def hasNaN(t: Tensor): Boolean = {
    if (t == null) return false
    val values = readFloats(t)
    var i = 0
    while (i < values.length) {
      if (values(i).isNaN) return true
      i += 1
    }
    false
  }

  // Helper to create 2D tensor
  def tensor2d(arr: Array[Array[Float]]): Tensor = {
    val flat = arr.flatten
    tensor(flat, Array(arr.length.toLong, arr(0).length.toLong))
  }

  // Helper to create 3D tensor from Array[Array[Array[Float]]]
  def tensor3d(arr: Array[Array[Array[Float]]]): Tensor = {
    val b = arr.length
    val s = arr(0).length
    val d = arr(0)(0).length
    val flat = arr.flatten.flatten
    tensor(flat, Array(b.toLong, s.toLong, d.toLong))
  }


  // Helper to create 4D tensor from Array[Array[Array[Array[Float]]]]
  def tensor4d(arr: Array[Array[Array[Array[Float]]]]): Tensor = {
    val b = arr.length
    val f1 = arr(0).length
    val f2 = arr(0)(0).length
    val d = arr(0)(0)(0).length
    val flat = arr.flatten.flatten.flatten
    tensor(flat, Array(b.toLong, f1.toLong, f2.toLong, d.toLong))
  }

}

