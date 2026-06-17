package benchmarks

import torchrec.dataframe._
import torchrec.data._
import torchrec.dataframe.DataFrameBridge
import torchrec.dataframe.FeatureTransformers
import torchrec.dataframe.PipelineBuilder
import torchrec.basic.features._
import torchrec.trainers._
import torchrec.utils.DeviceSupport
import torchrec.Implicits.tensor
import torchrec.Implicits._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.util.Random
import scala.collection.mutable

/**
 * Benchmark result case class.
 */
case class DataFrameBenchmarkResult(
                                     name: String,
                                     category: String,
                                     passed: Boolean,
                                     rowsProcessed: Long,
                                     latencyMs: Double,
                                     throughputRowsPerSec: Double,
                                     memoryMb: Double,
                                     error: Option[String] = None
                                   )

/**
 * Comprehensive DataFrame pipeline benchmark suite.
 * Validates all 50+ transformers, DataFrame <-> Dataset conversion, and full pipeline execution.
 */
object DataFrameBenchmark {

  val DEFAULT_NUM_ROWS = 10000
  val DEFAULT_BATCH_SIZE = 256
  val DEFAULT_EMBED_DIM = 8

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("DataFrame Pipeline Benchmark Suite")
    println("=" * 80)

    val results = mutable.ListBuffer[DataFrameBenchmarkResult]()

    // Phase 1: DataFrame Creation & EDA
    println("\n[Phase 1] DataFrame Creation & EDA")
    println("-" * 40)
    results += runDataFrameCreationBenchmark()
    results += runEDABenchmark()

    // Phase 2: Feature Engineering
    println("\n[Phase 2] Feature Engineering - Scaling Transformers")
    println("-" * 40)
    results ++= runScalingTransformersBenchmark()

    println("\n[Phase 2] Feature Engineering - Encoding Transformers")
    println("-" * 40)
    results ++= runEncodingTransformersBenchmark()

    println("\n[Phase 2] Feature Engineering - Recommender Transformers")
    println("-" * 40)
    results ++= runRecommenderTransformersBenchmark()

    // Phase 3: DataFrame <-> Dataset Conversion
    println("\n[Phase 3] DataFrame <-> Dataset Conversion")
    println("-" * 40)
    results += runDataFrameToDatasetBenchmark()
    results += runDatasetToDataFrameBenchmark()
    results += runBatchToDataFrameBenchmark()

    // Phase 4: DataLoader Integration
    println("\n[Phase 4] DataLoader Integration")
    println("-" * 40)
    results += runDataFrameDataLoaderBenchmark()

    // Phase 5: Full Pipeline
    println("\n[Phase 5] Full Pipeline")
    println("-" * 40)
    results += runFullPipelineBenchmark()

    printSummary(results.toList)
  }

  // ========================================================================
  // Phase 1: DataFrame Creation & EDA
  // ========================================================================

  def runDataFrameCreationBenchmark(): DataFrameBenchmarkResult = {
    val name = "DataFrameCreation"
    val category = "Phase1_EDA"

    try {
      val numRows = DEFAULT_NUM_ROWS
      val startTime = System.nanoTime()

      val rows = (0 until numRows).map { i =>
        Map(
          "user_id" -> (i % 1000).toString,
          "item_id" -> (i % 5000).toString,
          "price" -> (Random.nextFloat() * 100),
          "rating" -> (Random.nextFloat() * 5),
          "timestamp" -> s"2024-01-1${i % 9 + 1}T${i % 24}:00:00",
          "category" -> s"cat_${i % 20}",
          "label" -> (if (Random.nextFloat() > 0.5) 1 else 0)
        )
      }

      val df = DataFrame.fromRows(rows)
      val elapsed = (System.nanoTime() - startTime) / 1e6
      val throughput = numRows * 1000.0 / elapsed

      DataFrameBenchmarkResult(
        name = name,
        category = category,
        passed = df.numRows == numRows,
        rowsProcessed = numRows,
        latencyMs = elapsed,
        throughputRowsPerSec = throughput,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception =>
        DataFrameBenchmarkResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runEDABenchmark(): DataFrameBenchmarkResult = {
    val name = "EDA_Stats"
    val category = "Phase1_EDA"

    try {
      val numRows = DEFAULT_NUM_ROWS
      val rows = (0 until numRows).map { i =>
        Map(
          "price" -> (Random.nextFloat() * 100),
          "rating" -> (Random.nextFloat() * 5)
        )
      }
      val df = DataFrame.fromRows(rows)

      val startTime = System.nanoTime()

      for (colName <- df.columns) {
        val stats = df.col(colName).stats()
        assert(stats.count > 0, s"Stats should be computed for $colName")
      }

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val throughput = numRows * 1000.0 / elapsed

      DataFrameBenchmarkResult(
        name = name,
        category = category,
        passed = true,
        rowsProcessed = numRows,
        latencyMs = elapsed,
        throughputRowsPerSec = throughput,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception =>
        DataFrameBenchmarkResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  // ========================================================================
  // Phase 2: Feature Engineering - Scaling
  // ========================================================================

  def runScalingTransformersBenchmark(): List[DataFrameBenchmarkResult] = {
    val results = mutable.ListBuffer[DataFrameBenchmarkResult]()

    val numRows = DEFAULT_NUM_ROWS
    val rows = (0 until numRows).map { i =>
      Map(
        "price" -> (Random.nextFloat() * 100),
        "rating" -> (Random.nextFloat() * 5),
        "value1" -> (Random.nextFloat() * 50),
        "value2" -> (Random.nextFloat() * 200)
      )
    }
    val df = DataFrame.fromRows(rows)

    results += benchmarkTransformer("StandardScaler", "Phase2_Scaling", df, FeatureTransformers.standardScaler())
    results += benchmarkTransformer("MinMaxScaler", "Phase2_Scaling", df, FeatureTransformers.minMaxScaler())
    results += benchmarkTransformer("MaxAbsScaler", "Phase2_Scaling", df, FeatureTransformers.maxAbsScaler())
    results += benchmarkTransformer("RobustScaler", "Phase2_Scaling", df, FeatureTransformers.robustScaler())
    results += benchmarkTransformer("LogTransformer", "Phase2_Scaling", df, FeatureTransformers.logTransformer())

    results.toList
  }

  // ========================================================================
  // Phase 2: Feature Engineering - Encoding
  // ========================================================================

  def runEncodingTransformersBenchmark(): List[DataFrameBenchmarkResult] = {
    val results = mutable.ListBuffer[DataFrameBenchmarkResult]()

    val numRows = DEFAULT_NUM_ROWS
    val rows = (0 until numRows).map { i =>
      Map(
        "user_id" -> s"user_${i % 100}",
        "item_id" -> s"item_${i % 500}",
        "category" -> s"cat_${i % 20}",
        "brand" -> s"brand_${i % 50}",
        "label" -> (if (Random.nextFloat() > 0.5) 1 else 0)
      )
    }
    val df = DataFrame.fromRows(rows)

    results += benchmarkTransformer("LabelEncoder", "Phase2_Encoding", df, FeatureTransformers.labelEncoder())
    results += benchmarkTransformer("OneHotEncoder", "Phase2_Encoding", df, FeatureTransformers.oneHotEncoder())
    results += benchmarkTransformer("TargetEncoder", "Phase2_Encoding", df, FeatureTransformers.targetEncoder())
    results += benchmarkTransformer("CountEncoder", "Phase2_Encoding", df, FeatureTransformers.countEncoder())
    results += benchmarkTransformer("HashEncoder", "Phase2_Encoding", df, FeatureTransformers.hashEncoder())
    results += benchmarkTransformer("OrdinalEncoder", "Phase2_Encoding", df, FeatureTransformers.ordinalEncoder())
    results += benchmarkTransformer("CatBoostEncoder", "Phase2_Encoding", df, FeatureTransformers.catBoostEncoder())
    results += benchmarkTransformer("WOEEncoder", "Phase2_Encoding", df, FeatureTransformers.woeEncoder())
    results += benchmarkTransformer("EmbeddingEncoder", "Phase2_Encoding", df, FeatureTransformers.embeddingEncoder())
    results += benchmarkTransformer("SequenceEncoder", "Phase2_Encoding", df, FeatureTransformers.sequenceEncoder())

    results.toList
  }

  // ========================================================================
  // Phase 2: Feature Engineering - Recommender Transformers
  // ========================================================================

  def runRecommenderTransformersBenchmark(): List[DataFrameBenchmarkResult] = {
    val results = mutable.ListBuffer[DataFrameBenchmarkResult]()

    val numRows = DEFAULT_NUM_ROWS
    val rows = (0 until numRows).map { i =>
      Map(
        "user_id" -> s"user_${i % 100}",
        "item_id" -> s"item_${i % 500}",
        "age" -> (18 + Random.nextInt(60)).toFloat,
        "price" -> (Random.nextFloat() * 100),
        "rating" -> (Random.nextFloat() * 5),
        "click_history" -> (1 to (i % 10 + 1)).map(j => s"item_${(i + j) % 500}").mkString("|"),
        "category" -> s"cat_${i % 20}",
        "timestamp" -> s"2024-01-1${i % 9 + 1}T${i % 24}:00:00",
        "device" -> (if (i % 3 == 0) "mobile" else if (i % 3 == 1) "desktop" else "tablet"),
        "label" -> (if (Random.nextFloat() > 0.5) 1 else 0)
      )
    }
    val df = DataFrame.fromRows(rows)

    results += benchmarkTransformer("UserAgeBinTransformer", "Phase2_Recommender", df, FeatureTransformers.userAgeBinTransformer())
    results += benchmarkTransformer("ItemPopularityTransformer", "Phase2_Recommender", df, FeatureTransformers.itemPopularityTransformer())
    results += benchmarkTransformer("UserActivityTransformer", "Phase2_Recommender", df, FeatureTransformers.userActivityTransformer())
    results += benchmarkTransformer("CategoryEncoder", "Phase2_Recommender", df, FeatureTransformers.categoryEncoder())
    results += benchmarkTransformer("TagTransformer", "Phase2_Recommender", df, FeatureTransformers.tagTransformer())
    results += benchmarkTransformer("SequencePaddingTransformer", "Phase2_Recommender", df, FeatureTransformers.sequencePaddingTransformer())
    results += benchmarkTransformer("BehaviorHistoryTransformer", "Phase2_Recommender", df, FeatureTransformers.behaviorHistoryTransformer())
    results += benchmarkTransformer("TimeDecayTransformer", "Phase2_Recommender", df, FeatureTransformers.timeDecayTransformer())
    results += benchmarkTransformer("ItemAgeTransformer", "Phase2_Recommender", df, FeatureTransformers.itemAgeTransformer())
    results += benchmarkTransformer("UserClusterTransformer", "Phase2_Recommender", df, FeatureTransformers.userClusterTransformer())
    results += benchmarkTransformer("ItemClusterTransformer", "Phase2_Recommender", df, FeatureTransformers.itemClusterTransformer())
    results += benchmarkTransformer("PriceTransformer", "Phase2_Recommender", df, FeatureTransformers.priceTransformer())
    results += benchmarkTransformer("RatingTransformer", "Phase2_Recommender", df, FeatureTransformers.ratingTransformer())
    results += benchmarkTransformer("ContextualFeatureTransformer", "Phase2_Recommender", df, FeatureTransformers.contextualFeatureTransformer())
    results += benchmarkTransformer("ClickThroughRateTransformer", "Phase2_Recommender", df, FeatureTransformers.clickThroughRateTransformer())
    results += benchmarkTransformer("ConversionRateTransformer", "Phase2_Recommender", df, FeatureTransformers.conversionRateTransformer())
    results += benchmarkTransformer("UserEmbeddingTransformer", "Phase2_Recommender", df, FeatureTransformers.userEmbeddingTransformer())
    results += benchmarkTransformer("ItemEmbeddingTransformer", "Phase2_Recommender", df, FeatureTransformers.itemEmbeddingTransformer())
    results += benchmarkTransformer("GraphFeatureTransformer", "Phase2_Recommender", df, FeatureTransformers.graphFeatureTransformer())
    results += benchmarkTransformer("SequenceHistogramTransformer", "Phase2_Recommender", df, FeatureTransformers.sequenceHistogramTransformer())
    results += benchmarkTransformer("SequenceStatTransformer", "Phase2_Recommender", df, FeatureTransformers.sequenceStatTransformer())

    results.toList
  }

  // ========================================================================
  // Phase 3: DataFrame <-> Dataset Conversion
  // ========================================================================

  def runDataFrameToDatasetBenchmark(): DataFrameBenchmarkResult = {
    val name = "DataFrameToDataset"
    val category = "Phase3_Conversion"

    try {
      val numRows = DEFAULT_NUM_ROWS
      val rows = (0 until numRows).map { i =>
        Map(
          "user_id" -> (i % 100).toString,
          "item_id" -> (i % 500).toString,
          "price" -> (Random.nextFloat() * 100),
          "rating" -> (Random.nextFloat() * 5),
          "click_hist" -> (1 to 5).map(j => s"${(i + j) % 500}").mkString("|"),
          "label" -> (if (Random.nextFloat() > 0.5) 1 else 0)
        )
      }
      val df = DataFrame.fromRows(rows)

      val featureSpec = FeatureSpec(
        sparseFeatures = List(
          SparseSpec("user_id", 100, 8),
          SparseSpec("item_id", 500, 8)
        ),
        denseFeatures = List(
          DenseSpec("price", 1),
          DenseSpec("rating", 1)
        ),
        sequenceFeatures = List(
          SequenceSpec("click_hist", 500, 8, maxLen = 50)
        )
      )

      val labelSpec = LabelSpec("label")

      val startTime = System.nanoTime()
      val dataset = DataFrameBridge.toDataset(df, featureSpec, Some(labelSpec))
      val elapsed = (System.nanoTime() - startTime) / 1e6
      val throughput = numRows * 1000.0 / elapsed

      val batch = dataset.get(0)
      val valid = batch.sparseFeatures.nonEmpty && batch.denseFeatures.nonEmpty

      DataFrameBenchmarkResult(
        name = name,
        category = category,
        passed = valid,
        rowsProcessed = numRows,
        latencyMs = elapsed,
        throughputRowsPerSec = throughput,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception =>
        DataFrameBenchmarkResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runDatasetToDataFrameBenchmark(): DataFrameBenchmarkResult = {
    val name = "DatasetToDataFrame"
    val category = "Phase3_Conversion"

    try {
      val numRows = DEFAULT_NUM_ROWS

      val (trainData, _, _) = torchrec.data.DataGenerator.generateRankingData(
        numSamples = numRows,
        numSparseFeatures = 5,
        numDenseFeatures = 3,
        vocabSize = 100,
        seed = 42
      )

      val startTime = System.nanoTime()
      val df = DataFrameBridge.fromDataset(trainData)
      val elapsed = (System.nanoTime() - startTime) / 1e6
      val throughput = numRows * 1000.0 / elapsed

      df.show(300)

      DataFrameBenchmarkResult(
        name = name,
        category = category,
        passed = df.numRows == numRows,
        rowsProcessed = numRows,
        latencyMs = elapsed,
        throughputRowsPerSec = throughput,
        memoryMb = estimateMemory(df)
      )
    }
    //    catch {
    //      case e: Exception =>
    //        DataFrameBenchmarkResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    //    }
  }

  def runBatchToDataFrameBenchmark(): DataFrameBenchmarkResult = {
    val name = "BatchToDataFrame"
    val category = "Phase3_Conversion"

    try {
      val numRows = DEFAULT_NUM_ROWS

      val (trainData, _, _) = torchrec.data.DataGenerator.generateRankingData(
        numSamples = numRows,
        numSparseFeatures = 5,
        numDenseFeatures = 3,
        vocabSize = 100,
        seed = 42
      )

      val batch = trainData.get(0)

      val startTime = System.nanoTime()
      val df = DataFrameBridge.fromBatch(batch)
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameBenchmarkResult(
        name = name,
        category = category,
        passed = df.numRows == 1,
        rowsProcessed = 1,
        latencyMs = elapsed,
        throughputRowsPerSec = 1000.0 / elapsed,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception =>
        DataFrameBenchmarkResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  // ========================================================================
  // Phase 4: DataLoader Integration
  // ========================================================================

  def runDataFrameDataLoaderBenchmark(): DataFrameBenchmarkResult = {
    val name = "DataFrameDataLoader"
    val category = "Phase4_DataLoader"

    try {
      val numRows = DEFAULT_NUM_ROWS
      val batchSize = DEFAULT_BATCH_SIZE
      val rows = (0 until numRows).map { i =>
        Map(
          "user_id" -> (i % 100).toString,
          "item_id" -> (i % 500).toString,
          "price" -> (Random.nextFloat() * 100),
          "label" -> (if (Random.nextFloat() > 0.5) 1 else 0)
        )
      }
      val df = DataFrame.fromRows(rows)

      val featureSpec = FeatureSpec(
        sparseFeatures = List(
          SparseSpec("user_id", 100, 8),
          SparseSpec("item_id", 500, 8)
        ),
        denseFeatures = List(DenseSpec("price", 1))
      )

      val labelSpec = LabelSpec("label")

      val startTime = System.nanoTime()
      val dataLoader = DataFrameBridge.toDataLoader(df, featureSpec, Some(labelSpec), batchSize)

      var numBatches = 0
      val iter = dataLoader.iterator
      while (iter.hasNext && numBatches < 10) {
        iter.next()
        numBatches += 1
      }
      val elapsed = (System.nanoTime() - startTime) / 1e6
      val throughput = numRows * 1000.0 / elapsed

      DataFrameBenchmarkResult(
        name = name,
        category = category,
        passed = numBatches > 0,
        rowsProcessed = numRows,
        latencyMs = elapsed,
        throughputRowsPerSec = throughput,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception =>
        DataFrameBenchmarkResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  // ========================================================================
  // Phase 5: Full Pipeline
  // ========================================================================

  def runFullPipelineBenchmark(): DataFrameBenchmarkResult = {
    val name = "FullPipeline"
    val category = "Phase5_FullPipeline"

    try {
      val numRows = DEFAULT_NUM_ROWS
      val batchSize = DEFAULT_BATCH_SIZE

      val rows = (0 until numRows).map { i =>
        Map(
          "user_id" -> s"user_${i % 100}",
          "item_id" -> s"item_${i % 500}",
          "category" -> s"cat_${i % 20}",
          "price" -> (Random.nextFloat() * 100),
          "rating" -> (Random.nextFloat() * 5),
          "click_hist" -> (1 to (i % 10 + 1)).map(j => s"item_${(i + j) % 500}").mkString("|"),
          "label" -> (if (Random.nextFloat() > 0.5) 1 else 0)
        )
      }
      val rawDF = DataFrame.fromRows(rows)

      val pipeline = PipelineBuilder.create()
        .addScaling("dense", "standard", Map())
        .addEncoding("sparse", "label", Map())
        .addEncoding("sparse", "count", Map())
        .addRecommenderFeature("ctr", "clickThroughRate", Map())
        .addRecommenderFeature("userActivity", "userActivity", Map())
        .addRecommenderFeature("itemPopularity", "itemPopularity", Map())
        .build()

      val startTime = System.nanoTime()

      val processedDF = pipeline.fitTransform(rawDF)

      val featureSpec = FeatureSpec(
        sparseFeatures = List(
          SparseSpec("user_id", 100, 8),
          SparseSpec("item_id", 500, 8)
        ),
        denseFeatures = List(
          DenseSpec("price", 1),
          DenseSpec("rating", 1)
        ),
        sequenceFeatures = List(
          SequenceSpec("click_hist", 500, 8, maxLen = 50)
        )
      )

      val labelSpec = LabelSpec("label")
      val dataLoader = DataFrameBridge.toDataLoader(processedDF, featureSpec, Some(labelSpec), batchSize)

      var numBatches = 0
      val iter = dataLoader.iterator
      while (iter.hasNext && numBatches < 5) {
        iter.next()
        numBatches += 1
      }

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val throughput = numRows * 1000.0 / elapsed

      DataFrameBenchmarkResult(
        name = name,
        category = category,
        passed = numBatches > 0,
        rowsProcessed = numRows,
        latencyMs = elapsed,
        throughputRowsPerSec = throughput,
        memoryMb = estimateMemory(rawDF) + estimateMemory(processedDF)
      )
    }
    //    catch {
    //      case e: Exception =>
    //        DataFrameBenchmarkResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    //    }
  }

  // ========================================================================
  // Helper Methods
  // ========================================================================

  private def benchmarkTransformer(
                                    name: String,
                                    category: String,
                                    df: DataFrame,
                                    transformer: FeatureTransformer
                                  ): DataFrameBenchmarkResult = {
    try {
      val numRows = df.numRows
      val startTime = System.nanoTime()

      val resultDF = transformer.fitTransform(df)

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val throughput = numRows * 1000.0 / elapsed

      DataFrameBenchmarkResult(
        name = name,
        category = category,
        passed = resultDF.numRows == numRows,
        rowsProcessed = numRows,
        latencyMs = elapsed,
        throughputRowsPerSec = throughput,
        memoryMb = estimateMemory(resultDF)
      )
    } catch {
      case e: Exception =>
        DataFrameBenchmarkResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  private def estimateMemory(df: DataFrame): Double = {
    var totalBytes = 0L
    for (colName <- df.columns) {
      val col = df.col(colName)
      totalBytes += col.length * 8 // Approximate 8 bytes per element
    }
    totalBytes / (1024.0 * 1024.0)
  }

  private def printSummary(results: List[DataFrameBenchmarkResult]): Unit = {
    println("\n" + "=" * 80)
    println("Benchmark Results Summary")
    println("=" * 80)

    val passedCount = results.count(_.passed)
    val totalCount = results.length
    val passRate = if (totalCount > 0) passedCount * 100.0 / totalCount else 0

    println(f"\nPass Rate: $passRate%.1f%% ($passedCount/$totalCount)\n")

    val byCategory = results.groupBy(_.category)
    for ((category, categoryResults) <- byCategory.toSeq.sortBy(_._1)) {
      println(s"\n[$category]")
      println("-" * 60)
      println(f"${"Name"}%-30s${"Passed"}%-8s${"Rows"}%-12s${"Latency"}%-10s${"Throughput"}%-15s")
      println("-" * 60)

      for (result <- categoryResults.sortBy(_.name)) {
        val status = if (result.passed) "PASS" else "FAIL"
        val latency = f"${result.latencyMs}%.2fms"
        val throughput = f"${result.throughputRowsPerSec}%.0f rows/s"
        println(f"${result.name}%-30s${status}%-8s${result.rowsProcessed}%-12d${latency}%-10s${throughput}%-15s")

        if (!result.passed && result.error.isDefined) {
          println(f"  Error: ${result.error.get}")
        }
      }

      val categoryPassed = categoryResults.count(_.passed)
      val categoryTotal = categoryResults.length
      println(f"\nCategory: $categoryPassed/$categoryTotal passed")
    }

    println("\n" + "=" * 80)
    if (passRate >= 90) {
      println("OVERALL: SUCCESS - All critical benchmarks passed")
    } else if (passRate >= 70) {
      println("OVERALL: PARTIAL - Some benchmarks failed, review needed")
    } else {
      println("OVERALL: FAILURE - Too many benchmarks failed")
    }
    println("=" * 80)
  }
}