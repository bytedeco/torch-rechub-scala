package benchmarks.kt

import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.models.knowledge_tracing.*
import torchrec.utils.DeviceSupport

object ModelNpeDebug {
  def main(args: Array[String]): Unit = {
    val batch = 2
    val seq = 8
    val numConcepts = 50
    val embed = 64

    DeviceSupport.setDevice("cpu")
    val device = DeviceSupport.backend

    val concepts = longTensor((0 until (batch * seq)).map(_.toLong).toArray)
      .view(batch, seq)
      .to(device)
      .toType(ScalarType.Long)
    val responses = longTensor(Array.fill(batch * seq)(0L))
      .view(batch, seq)
      .to(device)
      .toType(ScalarType.Long)

    val factories: Seq[() => org.bytedeco.pytorch.nn.Module] = Seq(
      () => new AKT(numConcepts, embedDim = embed),
      () => new IEKT(numConcepts, embedDim = embed),
      () => new MTKT(numConcepts, embedDim = embed),
      () => new PromptKT(numConcepts, embedDim = embed),
      () => new QDKT(numQuestions = numConcepts, numConcepts = numConcepts, embedDim = embed),
      () => new RKT(numConcepts, embedDim = embed),
      () => new RobustKT(numConcepts, embedDim = embed),
      () => new SAINT(numExercises = numConcepts, numCategories = numConcepts, embedDim = embed),
      () => new SAINTPlusPlus(numExercises = numConcepts, numCategories = numConcepts, embedDim = embed),
      () => new UKT(numConcepts, embedDim = embed)
    )

    factories.foreach { f =>
      val m = try f() catch { case e: Throwable =>
        println(s"CTOR FAILED: ${e.getClass.getName}: ${e.getMessage}"); return
      }
      try {
        m.eval()
        val out = m match {
          case a: AKT => a.forward(concepts, responses)
          case b: IEKT => b.forward(concepts, responses)
          case c: MTKT => c.forward(concepts, responses)
          case d: PromptKT => d.forward(concepts, responses)
          case e0: QDKT => e0.forward(concepts, concepts, responses)
          case f2: RKT => f2.forward(concepts, responses)
          case g: RobustKT => g.forward(concepts, responses)
          case h: SAINT => h.forward(concepts, concepts, responses)
          case i: SAINTPlusPlus => i.forward(concepts, concepts, responses)
          case l: UKT => l.forward(concepts, responses)
        }
        println(s"Model ${m.getClass.getSimpleName} forward ok -> shape: ${out.shape().mkString("x")}")
      } catch { case e: Throwable =>
        println(s"\n==== ${m.getClass.getSimpleName} FAILED (full trace) ====")
        e.printStackTrace(System.out)
      }
    }
  }
}
