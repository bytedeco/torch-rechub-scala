package benchmarks.kt

import org.bytedeco.pytorch.global.torch
/*
* [PASS] assist0910 / DKT -> AUC=-0.8577, Accuracy=0.7474, LogLoss=0.5244
[W621 09:22:24.303132200 rnn.cpp:68] Warning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1 (function reset)
[PASS] assist0910 / DKTForget -> AUC=-0.8603, Accuracy=0.7463, LogLoss=0.5206
[PASS] assist0910 / DKVMN -> AUC=-0.6553, Accuracy=0.6601, LogLoss=0.6291
[PASS] assist0910 / DeepIRT -> AUC=-0.6225, Accuracy=0.6504, LogLoss=0.6410
[W621 09:26:32.024638134 rnn.cpp:68] Warning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.5 and num_layers=1 (function reset)
[PASS] assist0910 / GKT -> AUC=-0.8156, Accuracy=0.7350, LogLoss=0.5478
[PASS] assist0910 / ATDKT -> AUC=-0.8703, Accuracy=0.7578, LogLoss=0.5027
[PASS] assist0910 / ATDKT -> AUC=-0.8703, Accuracy=0.7578, LogLoss=0.5027
[PASS] assist0910 / DIMKT -> AUC=-0.8523, Accuracy=0.7632, LogLoss=0.5152
[PASS] assist0910 / SKVMN -> AUC=-0.7074, Accuracy=0.6725, LogLoss=0.6172
[PASS] assist0910 / QDKT -> AUC=-0.8279, Accuracy=0.7383, LogLoss=0.5410
[PASS] assist0910 / RKT -> AUC=-0.8568, Accuracy=0.7470, LogLoss=0.5224


[FAIL] assist0910 / AKT -> index out of range in self
[FAIL] assist0910 / IEKT -> index_select(): Expected dtype int32 or int64 for index
[FAIL] assist0910 / MTKT -> Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got CPUFloatType instead (while checking arguments for embedding)
[FAIL] assist0910 / PromptKT -> Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got CPUFloatType instead (while checking arguments for embedding)

* [FAIL] assist0910 / RKT -> Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got CPUFloatType instead (while checking arguments for embedding)
[FAIL] assist0910 / RobustKT -> [enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 2071210686004019200 bytes. Error code 12 (无法分配内存)
[FAIL] assist0910 / SAINT -> index out of range in self
[FAIL] assist0910 / SAINTPlusPlus -> Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got CPUFloatType instead (while checking arguments for embedding)
[FAIL] assist0910 / StableKT -> Dimension out of range (expected to be in range of [-2, 1], but got 2)
[FAIL] assist0910 / UKT -> Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got CPUFloatType instead (while checking arguments for embedding)
[FAIL] assist15 / StableKT -> Dimension out of range (expected to be in range of [-2, 1], but got 2)
[FAIL] assist15 / UKT -> Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got CPUFloatType instead (while checking arguments for embedding)
[FAIL] KDD / StableKT -> Dimension out of range (expected to be in range of [-2, 1], but got 2)
[FAIL] KDD / UKT -> Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got CPUFloatType instead (while checking arguments for embedding)
[FAIL] statics11 / StableKT -> Dimension out of range (expected to be in range of [-2, 1], but got 2)
[FAIL] statics11 / UKT -> Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got CPUFloatType instead (while checking arguments for embedding)
  - assist0910 / StableKT -> Dimension out of range (expected to be in range of [-2, 1], but got 2)
  - statics11 / StableKT -> Dimension out of range (expected to be in range of [-2, 1], but got 2)
  - statics11 / UKT -> Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got CPUFloatType instead (while checking arguments for embedding)

*
* */
object RunAllKnowledgeTracingBenchmarks {
  private val models = Vector(
    "DKT", "DKTForget", "DKVMN", "DeepIRT", "GKT", "ATDKT", "ATKT",   "DIMKT",  "SKVMN" ,"QDKT",   "RKT",
    "CSKT",   "LPKT",  "SAKT","StableKT", "MTKT", "PromptKT", "UKT",  "IEKT","SAINT","SAINTPlusPlus",  "AKT",



   "RobustKT",



  )
  private val datasets = Vector("assist0910")//, "assist15", "KDD", "statics11")
  def main(args: Array[String]): Unit = {
    val p = KnowledgeTracingHarness.Args.parse(args)
    val datasetRoot = p.getOrElse("dataset_root", "/home/muller/IdeaProjects/Knowledge-Tracing-Datasets")
    val epochs = p.getOrElse("epochs", "1").toInt
    val batchSize = p.getOrElse("batch_size", "32").toInt
    val maxSeqLen = p.getOrElse("max_seq_len", "200").toInt
    val stride = p.getOrElse("window_stride", p.getOrElse("max_seq_len", "200")).toInt
    val lr = p.getOrElse("lr", "0.001").toFloat
    val embedDim = p.getOrElse("embed_dim", "64").toInt
    val device = p.getOrElse("device", "cpu")
    val seed = p.getOrElse("seed", "2026").toInt
    val limitTrain = p.getOrElse("limit_train_sequences", "0").toInt
    val limitTest = p.getOrElse("limit_test_sequences", "0").toInt
    var failures = Vector.empty[String]
    for (dataset <- datasets; model <- models) {
      try {
        val result = KnowledgeTracingHarness.run(
          KnowledgeTracingHarness.KTRunConfig(
            datasetRoot = datasetRoot,
            datasetName = dataset,
            modelName = model,
            epochs = epochs,
            batchSize = batchSize,
            maxSeqLen = maxSeqLen,
            windowStride = stride,
            learningRate = lr,
            embedDim = embedDim,
            device = device,
            seed = seed,
            limitTrainSequences = limitTrain,
            limitTestSequences = limitTest
          )
        )
        val metricStr = result.metrics.toSeq.sortBy(_._1).map { case (k, v) => f"$k=$v%.4f" }.mkString(", ")
        System.gc()
        torch.emptyCache()
        println(s"[PASS] $dataset / $model -> $metricStr")
      } catch {
        case e: Throwable =>
          val msg = Option(e.getMessage).getOrElse(e.getClass.getSimpleName)
          failures :+= s"$dataset / $model -> $msg"
          println(s"[FAIL] $dataset / $model -> $msg")
      }
    }
    if (failures.nonEmpty) {
      println("\nFailures:")
      failures.foreach(f => println(s"  - $f"))
      throw new RuntimeException(s"Knowledge tracing suite failed: ${failures.size} case(s)")
    }
  }
}
