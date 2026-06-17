package benchmarks

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

object ModelBenchmark {

  def measureMemoryUsage(): Float = {
    val runtime = Runtime.getRuntime()
    val usedMemory = runtime.totalMemory() - runtime.freeMemory()
    usedMemory.toFloat / (1024 * 1024)
  }

  def profileModel(model: Module): Map[String, Any] = {
    var totalParams = 0L
    var trainableParams = 0L

    try {
      val params = model.parameters()
      val size = params.size().toInt
      var i = 0
      while (i < size) {
        val param = params.get(i)
        val numParams = param.numel().longValue()
        totalParams += numParams
        if (param.requires_grad()) {
          trainableParams += numParams
        }
        i += 1
      }
    } catch {
      case _: Throwable => totalParams = 0L
    }

    Map(
      "total_parameters" -> totalParams.toInt,
      "trainable_parameters" -> trainableParams.toInt,
      "model_size_mb" -> (totalParams * 4 / (1024 * 1024)).toFloat
    )
  }

  def profileModule(model: Module): Map[String, Any] = {
    val profile = profileModel(model)
    Map(
      "total_parameters" -> profile("total_parameters"),
      "trainable_parameters" -> profile("trainable_parameters"),
      "model_size_mb" -> profile("model_size_mb"),
      "memory_used_mb" -> measureMemoryUsage()
    )
  }

  def printBenchmarkResults(results: Map[String, Any]): Unit = {
    println("\n" + "=" * 60)
    println("Benchmark Results")
    println("=" * 60)

    results.foreach { case (key, value) =>
      val formattedKey = key.replace("_", " ").replaceFirst("(?<=.)([A-Z])", " $1").capitalize
      val formattedValue = value match {
        case v: Float if key.contains("time") => f"$v%.3f ms"
        case v: Float if key.contains("throughput") => f"$v%.2f samples/s"
        case v: Float if key.contains("size") || key.contains("memory") => f"$v%.2f MB"
        case v: Float if key.contains("parameters") => f"$v%.0f"
        case v: Float => f"$v%.4f"
        case v: Int => f"$v%,d"
        case v => v.toString
      }
      println(s"$formattedKey: $formattedValue")
    }

    println("=" * 60)
  }

  def printModelComparison(
                            comparison: Map[String, Map[String, Any]]
                          ): Unit = {
    println("\n" + "=" * 80)
    println("Model Comparison")
    println("=" * 80)

    printf("%-20s %-15s %-20s %-15s %-15s%n", "Model": Any, "Time (ms)": Any, "Throughput": Any, "Size (MB)": Any, "Params": Any)
    println("-" * 80)

    comparison.foreach { case (name, results) =>
      val time = results.get("avg_inference_time_ms").map(_.asInstanceOf[Float]).getOrElse(0f)
      val throughput = results.get("throughput_samples_per_sec").map(_.asInstanceOf[Float]).getOrElse(0f)
      val size = results.get("model_size_mb").map(_.asInstanceOf[Float]).getOrElse(0f)
      val params = results.get("total_parameters").map(_.asInstanceOf[Int]).getOrElse(0)

      println(f"$name%-20s${time}%-15.3f${throughput}%-20.2f${size}%-15.2f${params}%-15.0f")
    }

    println("=" * 80)
  }
}
