package benchmarks.kt

import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl
import org.bytedeco.pytorch.nn.options.EmbeddingOptions
import torchrec.utils.DeviceSupport

object ModuleBoolArgDebug {
  def main(args: Array[String]): Unit = {
    DeviceSupport.setDevice("mps")
    val m = new Module()
    println(s"ctor address=0x${m.address().toHexString}")

    try { m.train(false); println("train(false) OK") }
    catch { case e: Throwable => println(s"train(false) FAILED: ${e.getClass.getSimpleName}: ${e.getMessage}") }

    try { m.eval(); println("eval() OK") }
    catch { case e: Throwable => println(s"eval() FAILED: ${e.getClass.getSimpleName}: ${e.getMessage}") }

    // Try via different mechanism — the JNI/JVM might encode false differently.
    // Use require_grad(false) or similar to see if it's a bool issue.
    try { val b = !m.is_training(); println(s"is_training() = $b") }
    catch { case e: Throwable => println(s"is_training() FAILED: ${e.getMessage}") }

    // Check all bool-arg methods
    try { m.eval(); println("eval() 2nd try OK") }
    catch { case e: Throwable => println(s"eval() 2nd try FAILED: ${e.getMessage}") }
  }
}
