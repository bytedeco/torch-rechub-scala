package tutorials

import benchmarks.RunCriteoReplicaBenchmark

object Tutorial00QuickStartCTRDeepFMReplica {
  def main(args: Array[String]): Unit = {
    RunCriteoReplicaBenchmark.main(Array("--model_name", "deepfm", "--epoch", "1", "--batch_size", "512", "--device", "cpu"))
  }
}

