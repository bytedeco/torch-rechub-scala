package tutorials

import benchmarks.RunAvazuReplicaBenchmark

object TutorialAvazuWideDeepReplica {
  def main(args: Array[String]): Unit = {
    RunAvazuReplicaBenchmark.main(Array("--model_name", "widedeep", "--epoch", "2", "--batch_size", "512", "--device", "cpu"))
  }
}

