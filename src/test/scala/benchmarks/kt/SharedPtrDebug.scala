package benchmarks.kt

import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl
import org.bytedeco.pytorch.nn.options.EmbeddingOptions
import org.bytedeco.pytorch.nn.modules.LinearImpl
import torchrec.utils.DeviceSupport

object SharedPtrDebug {
  def main(args: Array[String]): Unit = {
    DeviceSupport.setDevice("cpu")
    // The PyTorch @SharedPtr uses a separate underlying pointer.
    // After allocate(), Python's `addr = self.address` would normally be 0
    // because allocate() returns into shared_ptr, not into address.
    // Train method works around this by storing peer obj differently.
    val m = new Module()
    println(s"ctor address=0x${m.address().toLong.toHexString}")

    // Many JavaCPP @SharedPtr classes use a `javacpp_module_object_id()` helper
    // to inspect the inner shared object ID — let's see if that's set.
    try { val id = m.javacpp_module_object_id(); println(s"javacpp_module_object_id = $id") }
    catch { case e: Throwable => println(s"javacpp_module_object_id FAILED: ${e.getMessage}") }

    // Try cloning to see if Module can produce a working clone
    try {
      val cl = m.clone(new org.bytedeco.pytorch.DeviceOptional)
      println(s"clone address=0x${cl.address().toLong.toHexString}")
      try { cl.eval(); println("clone eval() OK") }
      catch { case e: Throwable => println(s"clone eval() FAILED: ${e.getMessage}") }
    } catch { case e: Throwable => println(s"clone FAILED: ${e.getMessage}") }
  }
}
