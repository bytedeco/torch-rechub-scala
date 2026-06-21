package benchmarks.kt
object RunSKVMNKtBenchmark {
  def main(args: Array[String]): Unit = {
    val p = KnowledgeTracingHarness.Args.parse(args)
    val dataset = p.getOrElse("dataset", "assist0910")
    val result = KnowledgeTracingHarness.run(
      KnowledgeTracingHarness.KTRunConfig(
        datasetName = dataset,
        modelName = "SKVMN",
        datasetRoot = p.getOrElse("dataset_root", "/home/muller/IdeaProjects/Knowledge-Tracing-Datasets"),
        epochs = p.getOrElse("epochs", "1").toInt,
        batchSize = p.getOrElse("batch_size", "32").toInt,
        maxSeqLen = p.getOrElse("max_seq_len", "200").toInt,
        windowStride = p.getOrElse("window_stride", p.getOrElse("max_seq_len", "200")).toInt,
        learningRate = p.getOrElse("lr", "0.001").toFloat,
        embedDim = p.getOrElse("embed_dim", "64").toInt,
        device = p.getOrElse("device", "cpu"),
        seed = p.getOrElse("seed", "2026").toInt,
        limitTrainSequences = p.getOrElse("limit_train_sequences", "0").toInt,
        limitTestSequences = p.getOrElse("limit_test_sequences", "0").toInt
      )
    )
    println(s"[PASS] SKVMN on $dataset -> " + result.metrics.toSeq.sortBy(_._1).map { case (k, v) => f"$k=$v%.4f" }.mkString(", "))
  }
}
