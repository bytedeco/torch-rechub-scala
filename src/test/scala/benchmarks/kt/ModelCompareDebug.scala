package benchmarks.kt

import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import org.bytedeco.pytorch.nn.Module
import torchrec.Implicits.*
import torchrec.models.knowledge_tracing.*
import torchrec.utils.DeviceSupport
import org.bytedeco.javacpp.Pointer

object ModelCompareDebug {
  def main(args: Array[String]): Unit = {
    DeviceSupport.setDevice("cpu")

    val models: Seq[(String, () => Module)] = Seq(
      "DKT" -> (() => new DKT(50L)),
      "DKTForget" -> (() => new DKTForget(50L)),
      "SAKT" -> (() => new SAKT(50L)),
      "AKT" -> (() => new AKT(50L)),
      "IEKT" -> (() => new IEKT(50L)),
      "MTKT" -> (() => new MTKT(50L)),
      "PromptKT" -> (() => new PromptKT(50L)),
      "QDKT" -> (() => new QDKT(numQuestions = 50, numConcepts = 50)),
      "RKT" -> (() => new RKT(50L)),
      "RobustKT" -> (() => new RobustKT(50L)),
      "SAINT" -> (() => new SAINT(numExercises = 50, numCategories = 50)),
      "SAINTPlusPlus" -> (() => new SAINTPlusPlus(numExercises = 50, numCategories = 50)),
      "UKT" -> (() => new UKT(50L)),
      "ATKT" -> (() => new ATKT(50L)),
      "LPKT" -> (() => new LPKT(numExercises = 50, numConcepts = 50, numActionTypes = 2))
    )

    models.foreach { case (name, mk) =>
      try {
        val m = mk()
        val addr: Long = m.address()
        println(s"[$name] ctor ok, address=0x${addr.toHexString}, isNull=${m.isNull}")
        try {
          m.eval()
          println(f"[$name%-15s] eval OK")
        } catch { case e: Throwable =>
          println(f"[$name%-15s] eval FAILED: ${e.getClass.getSimpleName}: ${e.getMessage}")
        }
      } catch { case e: Throwable =>
        println(f"[$name%-15s] CTOR THREW: ${e.getClass.getName}: ${e.getMessage}")
        e.printStackTrace(System.out)
      }
    }
  }
}
