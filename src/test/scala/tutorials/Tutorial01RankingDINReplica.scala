package tutorials

import benchmarks.RunAmazonElectronicsReplicaBenchmark

object Tutorial01RankingDINReplica {
  def main(args: Array[String]): Unit = {
    RunAmazonElectronicsReplicaBenchmark.main(Array("--epoch", "1", "--batch_size", "256", "--device", "cpu"))
  }
}

