package benchmarks.kt

import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl
import org.bytedeco.pytorch.nn.options.EmbeddingOptions
import torchrec.utils.DeviceSupport

object ModuleMethodDebug {
  def main(args: Array[String]): Unit = {
    DeviceSupport.setDevice("cpu")
    val m = new Module()
    println(s"ctor address=0x${m.address().toHexString}")

    println("--- probe methods ---")
    try { println(s"name() = ${m.name()} (empty=" + m.name().toString().isEmpty + ")"); println("name() OK") }
    catch { case e: Throwable => println(s"name() FAILED: ${e.getClass.getSimpleName}: ${e.getMessage}") }

    try { val t = m.is_training(); println(s"is_training() = $t OK") }
    catch { case e: Throwable => println(s"is_training() FAILED: ${e.getClass.getSimpleName}: ${e.getMessage}") }

    try { m.train(true); println("train(true) OK") }
    catch { case e: Throwable => println(s"train(true) FAILED: ${e.getClass.getSimpleName}: ${e.getMessage}") }

    try { m.eval(); println("eval() OK") }
    catch { case e: Throwable => println(s"eval() FAILED: ${e.getClass.getSimpleName}: ${e.getMessage}") }

    // Now try adding a child module and registering it
    println("--- with child module ---")
    val child = new EmbeddingImpl(new EmbeddingOptions(100L, 16))
    println(s"child ctor address=0x${child.address().toHexString}")
    try { m.register_module("e", child); println("register_module OK") }
    catch { case e: Throwable => println(s"register_module FAILED: ${e.getClass.getSimpleName}: ${e.getMessage}") }

    try { child.eval(); println("child eval() OK") }
    catch { case e: Throwable => println(s"child eval() FAILED: ${e.getClass.getSimpleName}: ${e.getMessage}") }

    try { m.eval(); println("m eval() after register OK") }
    catch { case e: Throwable => println(s"m eval() after register FAILED: ${e.getClass.getSimpleName}: ${e.getMessage}") }
  }
}
