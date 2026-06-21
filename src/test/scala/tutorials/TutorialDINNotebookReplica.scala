package tutorials

import benchmarks.RunAvazuReplicaBenchmark

object TutorialDINNotebookReplica {
  def main(args: Array[String]): Unit = {
    RunAvazuReplicaBenchmark.main(Array("--model_name", "dcn", "--epoch", "1", "--batch_size", "512", "--device", "cpu"))
  }
}

