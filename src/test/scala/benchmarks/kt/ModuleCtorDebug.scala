package benchmarks.kt

import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl
import org.bytedeco.pytorch.nn.options.EmbeddingOptions
import torchrec.utils.DeviceSupport

object ModuleCtorDebug {
  def main(args: Array[String]): Unit = {
    DeviceSupport.setDevice("cpu")
    // Test 1: bare Module()
    try {
      val m = new Module()
      println(s"[Test 1] bare new Module() address=0x${m.address().toHexString} isNull=${m.isNull}")
      try { m.eval(); println("[Test 1] eval OK") }
      catch { case e: Throwable => println(s"[Test 1] eval FAILED: ${e.getClass.getName}: ${e.getMessage}") }
    } catch { case e: Throwable => println(s"[Test 1] CTOR THREW: ${e.getMessage}"); e.printStackTrace(System.out) }

    // Test 2: extends Module without parens — emulating a trivial subclass
    class Sub extends Module {
      val e = new EmbeddingImpl(new EmbeddingOptions(100L, 16))
      register_module("e", e)
    }
    try {
      val s = new Sub()
      println(s"[Test 2] extends Module address=0x${s.address().toHexString} isNull=${s.isNull}")
      try { s.eval(); println("[Test 2] eval OK") }
      catch { case e: Throwable => println(s"[Test 2] eval FAILED: ${e.getClass.getName}: ${e.getMessage}") }
    } catch { case e: Throwable => println(s"[Test 2] CTOR THREW: ${e.getMessage}"); e.printStackTrace(System.out) }

    // Test 3: extends Module() with parens
    class SubParen extends Module() {
      val e = new EmbeddingImpl(new EmbeddingOptions(100L, 16))
      register_module("e", e)
    }
    try {
      val s = new SubParen()
      println(s"[Test 3] extends Module() address=0x${s.address().toHexString} isNull=${s.isNull}")
      try { s.eval(); println("[Test 3] eval OK") }
      catch { case e: Throwable => println(s"[Test 3] eval FAILED: ${e.getClass.getName}: ${e.getMessage}") }
    } catch { case e: Throwable => println(s"[Test 3] CTOR THREW: ${e.getMessage}"); e.printStackTrace(System.out) }
  }
}
