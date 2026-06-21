package tutorials

import benchmarks.RunAvazuReplicaBenchmark

object Tutorial04ExperimentTrackingLightReplica {
  def main(args: Array[String]): Unit = {
    val t0 = System.currentTimeMillis()
    RunAvazuReplicaBenchmark.main(Array("--model_name", "widedeep", "--epoch", "1", "--batch_size", "512", "--device", "cpu"))
    val sec = (System.currentTimeMillis() - t0) / 1000.0
    println(f"[TRACK] tutorial=04 elapsed_sec=$sec%.2f status=ok")
  }
}

