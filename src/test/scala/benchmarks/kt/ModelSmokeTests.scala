package benchmarks.kt

import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.models.knowledge_tracing.*
import torchrec.utils.DeviceSupport

object ModelSmokeTests {
  def main(args: Array[String]): Unit = {
    val batch = 2
    val seq = 8
    val numConcepts = 50
    val embed = 64

    // Force CPU for the quick model smoke-tests to avoid CUDA/device mismatches
    // (call must happen before any model is constructed)
    DeviceSupport.setDevice("cpu")
    val device = DeviceSupport.backend

    // Ensure these are Long tensors on the chosen device
    val concepts = longTensor((0 until (batch * seq)).map(_.toLong).toArray)
      .view(batch, seq)
      .to(device)
      .toType(ScalarType.Long)

    val responses = longTensor(Array.fill(batch * seq)(0L))
      .view(batch, seq)
      .to(device)
      .toType(ScalarType.Long)

    // Model factories wrapped to avoid crashing the whole test on heavy initializations.
    val factories: Seq[() => org.bytedeco.pytorch.Module] = Seq(
      () => new AKT(numConcepts, embedDim = embed),
      () => new IEKT(numConcepts, embedDim = embed),
      () => new MTKT(numConcepts, embedDim = embed),
      () => new PromptKT(numConcepts, embedDim = embed),
      () => new QDKT(numQuestions = numConcepts, numConcepts = numConcepts, embedDim = embed),
      () => new RKT(numConcepts, embedDim = embed),
      // robust/saint models can be heavy on initialization; wrap in try to continue
      () => new RobustKT(numConcepts, embedDim = embed),
      () => new SAINT(numExercises = numConcepts, numCategories = numConcepts, embedDim = embed),
      () => new SAINTPlusPlus(numExercises = numConcepts, numCategories = numConcepts, embedDim = embed),
      //      () => new SKVMN(numConcepts, embedDim = embed),
      //      () => new StableKT(numConcepts, embedDim = embed),
      () => new UKT(numConcepts, embedDim = embed)
    )

    factories.foreach { f =>
      try {
        val m = try { f() } catch { case e: Throwable =>
          println(s"Model factory failed during construction: ${e.getMessage}"); null
        }
        if (m != null) {
          try {
            m.eval()
            val out = m match {
              case a: AKT => a.forward(concepts, responses)
              case b: IEKT => b.forward(concepts, responses)
              case c: MTKT => c.forward(concepts, responses)
              case d: PromptKT => d.forward(concepts, responses)
              case e: QDKT => e.forward(concepts, concepts, responses)
              case f2: RKT => f2.forward(concepts, responses)
              case g: RobustKT => g.forward(concepts, responses)
              case h: SAINT => h.forward(concepts, concepts, responses)
              case i: SAINTPlusPlus => i.forward(concepts, concepts, responses)
              case j: SKVMN => j.forward(concepts, responses)
              case k: StableKT => k.forward(concepts, responses)
              case l: UKT => l.forward(concepts, responses)
              case other => throw new IllegalArgumentException(s"Unsupported model: ${other.getClass}")
            }
            println(s"Model ${m.getClass.getSimpleName} forward ok -> shape: ${out.shape().mkString("x")}")
          } catch { case e: Throwable => println(s"Model ${m.getClass.getSimpleName} forward failed: ${e.getMessage}") }
        }
      } catch { case e: Throwable => println(s"Unexpected error: ${e.getMessage}") }
    }
  }
}


