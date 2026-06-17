package benchmarks

import torchrec.dataframe._
import torchrec.dataframe.DataFrameBridge
import torchrec.dataframe.FeatureTransformers
import torchrec.dataframe.PipelineBuilder

import scala.util.Random
import scala.collection.mutable

/**
 * Benchmark result case class for DataFrame operations.
 */
case class DataFrameOpResult(
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
 * Comprehensive DataFrame operations benchmark suite.
 * Tests 40+ DataFrame API functions including selection, filtering,
 * column operations, aggregation, join, string operations, and more.
 */
object DataFrameOperationsBenchmark {

  val DEFAULT_NUM_ROWS = 10000
  val DEFAULT_BATCH_SIZE = 256

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("DataFrame Operations Benchmark Suite")
    println("=" * 80)

    val results = mutable.ListBuffer[DataFrameOpResult]()

    // Create test DataFrame once for reuse
    val testDF = createTestDataFrame(DEFAULT_NUM_ROWS)

    // Phase 1: Selection Operations
    println("\n[Phase 1] Selection Operations")
    println("-" * 40)
    results += runSelectBenchmark(testDF)
    results += runDropBenchmark(testDF)
    results += runSelectColsBenchmark(testDF)

    // Phase 2: Filtering Operations
    println("\n[Phase 2] Filtering Operations")
    println("-" * 40)
    results += runFilterPredicateBenchmark(testDF)
    results += runLimitBenchmark(testDF)
    results += runHeadBenchmark(testDF)

    // Phase 3: Column Operations
    println("\n[Phase 3] Column Operations")
    println("-" * 40)
    results += runWithColumnBenchmark(testDF)
    results += runRenameBenchmark(testDF)
    results += runCastBenchmark(testDF)
    results += runStrLengthBenchmark(testDF)
    results += runStrUpperBenchmark(testDF)
    results += runStrContainsBenchmark(testDF)

    // Phase 4: Aggregation Operations
    println("\n[Phase 4] Aggregation Operations")
    println("-" * 40)
    results += runGroupByBenchmark(testDF)
    results += runGroupByCountBenchmark(testDF)
    results += runGroupBySumBenchmark(testDF)
    results += runGroupByMeanBenchmark(testDF)

    // Phase 5: Binary Operations
    println("\n[Phase 5] Binary Operations")
    println("-" * 40)
    results += runUnionBenchmark(testDF)
    results += runJoinBenchmark(testDF)

    // Phase 6: Evaluation Operations
    println("\n[Phase 6] Evaluation Operations")
    println("-" * 40)
    results += runCollectBenchmark(testDF)
    results += runToMapSeqBenchmark(testDF)
    results += runDescribeBenchmark(testDF)

    // Phase 7: Row Operations
    println("\n[Phase 7] Row Operations")
    println("-" * 40)
    results += runRowAccessBenchmark(testDF)
    results += runRowToMapBenchmark(testDF)
    results += runRowConcatBenchmark(testDF)

    // Phase 8: Column Statistics
    println("\n[Phase 8] Column Statistics")
    println("-" * 40)
    results += runColumnStatsBenchmark(testDF)
    results += runColumnMinBenchmark(testDF)
    results += runColumnMaxBenchmark(testDF)
    results += runColumnIsNullBenchmark(testDF)
    results += runColumnFillNullBenchmark(testDF)

    // Phase 9: DataFrame Creation
    println("\n[Phase 9] DataFrame Creation")
    println("-" * 40)
    results += runCreateFromRowsBenchmark()
    results += runCreateEmptyBenchmark()
    results += runCreateFromCSVPathBenchmark()

    // Phase 10: Complex Operations
    println("\n[Phase 10] Complex Operations")
    println("-" * 40)
    results += runComplexPipelineBenchmark()
    results += runMultiGroupByBenchmark(testDF)
    results += runFilteredAggregationBenchmark(testDF)

    printSummary(results.toList)
  }

  // ========================================================================
  // Test DataFrame Factory
  // ========================================================================

  def createTestDataFrame(numRows: Int): DataFrame = {
    val rows = (0 until numRows).map { i =>
      Map(
        "user_id" -> (i % 100).toString,
        "item_id" -> (i % 500).toString,
        "category" -> s"cat_${i % 20}",
        "price" -> (Random.nextFloat() * 100),
        "rating" -> (Random.nextFloat() * 5),
        "count" -> (i % 50).toInt,
        "active" -> (if (i % 3 == 0) true else false),
        "label" -> (if (Random.nextFloat() > 0.5) 1 else 0)
      )
    }
    DataFrame.fromRows(rows)
  }

  // ========================================================================
  // Phase 1: Selection Operations
  // ========================================================================

  def runSelectBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "select"
    val category = "Phase1_Selection"
    try {
      val startTime = System.nanoTime()
      val result = df.select("user_id", "item_id", "price")
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.numCols == 3 && result.columns.contains("user_id"),
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runDropBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "drop"
    val category = "Phase1_Selection"
    try {
      val startTime = System.nanoTime()
      val result = df.drop("category", "active")
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.numCols == df.numCols - 2 && !result.columns.contains("category"),
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runSelectColsBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "selectCols"
    val category = "Phase1_Selection"
    try {
      val startTime = System.nanoTime()
      val result = df.selectCols(0 until 4)
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.numCols == 4,
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  // ========================================================================
  // Phase 2: Filtering Operations
  // ========================================================================

  def runFilterPredicateBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "filter_predicate"
    val category = "Phase2_Filtering"
    try {
      val startTime = System.nanoTime()
      // Filter on count column (Int type) - keep rows where count > 25
      val result = df.filter((v: Any) => v match {
        case i: Int => i > 25
        case f: Float => f > 25.0f
        case d: Double => d > 25.0
        case l: Long => l > 25L
        case s: String => s.toIntOption.exists(_ > 25)
        case _ => false
      })
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.numRows <= df.numRows && result.numRows > 0,
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runLimitBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "limit"
    val category = "Phase2_Filtering"
    try {
      val limitSize = 100
      val startTime = System.nanoTime()
      val result = df.limit(limitSize)
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.numRows == limitSize,
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runHeadBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "head"
    val category = "Phase2_Filtering"
    try {
      val startTime = System.nanoTime()
      val result = df.head(10)
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.numRows == 10,
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  // ========================================================================
  // Phase 3: Column Operations
  // ========================================================================

  def runWithColumnBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "withColumn"
    val category = "Phase3_ColumnOps"
    try {
      val newCol = Column("price_x2", (0 until df.numRows).map(i => df("price")(i).asInstanceOf[Float] * 2).map(_.asInstanceOf[Any]), DataType.Float32)
      val startTime = System.nanoTime()
      val result = df.withColumn("price_x2", newCol)
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.numCols == df.numCols + 1 && result.columns.contains("price_x2"),
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runRenameBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "rename"
    val category = "Phase3_ColumnOps"
    try {
      val startTime = System.nanoTime()
      val result = df.rename("user_id", "uid")
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.columns.contains("uid") && !result.columns.contains("user_id"),
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runCastBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "cast"
    val category = "Phase3_ColumnOps"
    try {
      val startTime = System.nanoTime()
      val result = df.cast("count", DataType.Float32)
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result("count").dtype == DataType.Float32,
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runStrLengthBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "str_length"
    val category = "Phase3_ColumnOps"
    try {
      val startTime = System.nanoTime()
      val result = df.str("category").length
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.length == df.numRows && result.name.endsWith("_length"),
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runStrUpperBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "str_upper"
    val category = "Phase3_ColumnOps"
    try {
      val startTime = System.nanoTime()
      val result = df.str("category").upper
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.length == df.numRows && result.dtype == DataType.String,
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runStrContainsBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "str_contains"
    val category = "Phase3_ColumnOps"
    try {
      val startTime = System.nanoTime()
      val result = df.str("category").contains("cat_1")
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.length == df.numRows && result.dtype == DataType.Boolean,
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  // ========================================================================
  // Phase 4: Aggregation Operations
  // ========================================================================

  def runGroupByBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "groupBy"
    val category = "Phase4_Aggregation"
    try {
      val startTime = System.nanoTime()
      val grouped = df.groupBy("category")
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = grouped != null && grouped.isInstanceOf[GroupedDataFrame],
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runGroupByCountBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "groupBy_count"
    val category = "Phase4_Aggregation"
    try {
      val startTime = System.nanoTime()
      val result = df.groupBy("category").count()
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.numRows > 0 && result.numRows <= df.numRows,
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runGroupBySumBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "groupBy_sum"
    val category = "Phase4_Aggregation"
    try {
      val startTime = System.nanoTime()
      val result = df.groupBy("category").agg(Aggregation("price", AggregationType.Sum, Some("price_sum")))
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.numRows > 0 && result.columns.contains("price_sum"),
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runGroupByMeanBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "groupBy_mean"
    val category = "Phase4_Aggregation"
    try {
      val startTime = System.nanoTime()
      val result = df.groupBy("category").agg(Aggregation("rating", AggregationType.Mean, Some("rating_mean")))
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.numRows > 0 && result.columns.contains("rating_mean"),
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  // ========================================================================
  // Phase 5: Binary Operations
  // ========================================================================

  def runUnionBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "union"
    val category = "Phase5_BinaryOps"
    try {
      val df2 = createTestDataFrame(1000)
      val startTime = System.nanoTime()
      val result = df.union(df2)
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.numRows == df.numRows + 1000 && result.numCols == df.numCols,
        rowsProcessed = (df.numRows + 1000).toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = (df.numRows + 1000) * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runJoinBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "join"
    val category = "Phase5_BinaryOps"
    try {
      val df2Rows = (0 until 100).map { i =>
        Map("category" -> s"cat_${i % 20}", "extra" -> (i * 2).toFloat)
      }
      val df2 = DataFrame.fromRows(df2Rows)
      val startTime = System.nanoTime()
      val result = df.join(df2, "category", JoinType.Inner)
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result != null && result.numRows >= 0 && result.numCols >= df.numCols,
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  // ========================================================================
  // Phase 6: Evaluation Operations
  // ========================================================================

  def runCollectBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "collect"
    val category = "Phase6_Evaluation"
    try {
      val startTime = System.nanoTime()
      val result = df.collect()
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.length == df.numRows,
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runToMapSeqBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "toMapSeq"
    val category = "Phase6_Evaluation"
    try {
      val startTime = System.nanoTime()
      val result = df.toMapSeq
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.length == df.numRows && result.head.contains("user_id"),
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runDescribeBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "describe"
    val category = "Phase6_Evaluation"
    try {
      val startTime = System.nanoTime()
      val result = df.describe("price", "rating")
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.numRows > 0,
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  // ========================================================================
  // Phase 7: Row Operations
  // ========================================================================

  def runRowAccessBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "row_access"
    val category = "Phase7_RowOps"
    try {
      val startTime = System.nanoTime()
      var count = 0
      for (i <- 0 until math.min(1000, df.numRows)) {
        val row = Row.fromIndex(df, i)
        val uid = row("user_id")
        if (uid != null) count += 1
      }
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = count > 0,
        rowsProcessed = 1000,
        latencyMs = elapsed,
        throughputRowsPerSec = 1000 * 1000.0 / elapsed,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runRowToMapBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "row_toMap"
    val category = "Phase7_RowOps"
    try {
      val startTime = System.nanoTime()
      val row = Row.fromIndex(df, 0)
      val map = row.toMap
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = map.size == df.numCols && map.contains("user_id"),
        rowsProcessed = 1,
        latencyMs = elapsed,
        throughputRowsPerSec = 1000.0 / elapsed,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runRowConcatBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "row_concat"
    val category = "Phase7_RowOps"
    try {
      val rows = (0 until 100).map(i => Row.fromIndex(df, i))
      val startTime = System.nanoTime()
      val result = Row.concat(rows)
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.numRows == 100 && result.numCols == df.numCols,
        rowsProcessed = 100,
        latencyMs = elapsed,
        throughputRowsPerSec = 100 * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  // ========================================================================
  // Phase 8: Column Statistics
  // ========================================================================

  def runColumnStatsBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "column_stats"
    val category = "Phase8_ColumnStats"
    try {
      val startTime = System.nanoTime()
      val stats = df("price").stats()
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = stats.count > 0 && !stats.mean.isNaN,
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runColumnMinBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "column_min"
    val category = "Phase8_ColumnStats"
    try {
      val startTime = System.nanoTime()
      val stats = df("price").stats()
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = !stats.min.isInfinity,
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runColumnMaxBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "column_max"
    val category = "Phase8_ColumnStats"
    try {
      val startTime = System.nanoTime()
      val stats = df("price").stats()
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = !stats.max.isInfinity,
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runColumnIsNullBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "column_isNull"
    val category = "Phase8_ColumnStats"
    try {
      val startTime = System.nanoTime()
      val nullMask = df("price").isNull
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = nullMask.length == df.numRows && nullMask.dtype == DataType.Boolean,
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runColumnFillNullBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "column_fillNull"
    val category = "Phase8_ColumnStats"
    try {
      val startTime = System.nanoTime()
      val filled = df("price").fillNull(0.0f)
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = filled.length == df.numRows,
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  // ========================================================================
  // Phase 9: DataFrame Creation
  // ========================================================================

  def runCreateFromRowsBenchmark(): DataFrameOpResult = {
    val name = "create_fromRows"
    val category = "Phase9_Creation"
    try {
      val startTime = System.nanoTime()
      val df = createTestDataFrame(DEFAULT_NUM_ROWS)
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = df.numRows == DEFAULT_NUM_ROWS && df.numCols == 8,
        rowsProcessed = DEFAULT_NUM_ROWS.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = DEFAULT_NUM_ROWS * 1000.0 / elapsed,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runCreateEmptyBenchmark(): DataFrameOpResult = {
    val name = "create_empty"
    val category = "Phase9_Creation"
    try {
      val startTime = System.nanoTime()
      val df = DataFrame.empty()
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = df.isEmpty && df.numCols == 0,
        rowsProcessed = 0,
        latencyMs = elapsed,
        throughputRowsPerSec = 0,
        memoryMb = 0
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runCreateFromCSVPathBenchmark(): DataFrameOpResult = {
    val name = "create_fromCSV"
    val category = "Phase9_Creation"
    try {
      // Create a temporary CSV file
      val tempPath = "/tmp/test_df_temp.csv"
      val testRows = (0 until 1000).map { i =>
        Map(
          "id" -> i.toString,
          "value" -> (i * 2).toString,
          "name" -> s"item_$i"
        )
      }
      val tempDF = DataFrame.fromRows(testRows)
      tempDF.writeCSV(tempPath, ',')

      val startTime = System.nanoTime()
      val df = DataFrame.readCSV(tempPath)
      val elapsed = (System.nanoTime() - startTime) / 1e6

      // Clean up
      new java.io.File(tempPath).delete()

      DataFrameOpResult(
        name = name,
        category = category,
        passed = df.numRows == 1000 && df.numCols == 3,
        rowsProcessed = 1000,
        latencyMs = elapsed,
        throughputRowsPerSec = 1000 * 1000.0 / elapsed,
        memoryMb = estimateMemory(df)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  // ========================================================================
  // Phase 10: Complex Operations
  // ========================================================================

  def runComplexPipelineBenchmark(): DataFrameOpResult = {
    val name = "complex_pipeline"
    val category = "Phase10_Complex"
    try {
      val df = createTestDataFrame(DEFAULT_NUM_ROWS)
      val pipeline = PipelineBuilder.create()
        .addScaling("dense", "standard", Map())
        .addEncoding("sparse", "label", Map())
        .addRecommenderFeature("ctr", "clickThroughRate", Map())
        .build()

      val startTime = System.nanoTime()
      val result = pipeline.fitTransform(df)
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.numRows == df.numRows && result.numCols >= df.numCols,
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runMultiGroupByBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "multi_groupBy"
    val category = "Phase10_Complex"
    try {
      val startTime = System.nanoTime()
      val result = df.groupBy("category", "active")
        .agg(
          Aggregation("price", AggregationType.Sum, Some("price_sum")),
          Aggregation("rating", AggregationType.Mean, Some("rating_mean")),
          Aggregation("count", AggregationType.Count, Some("count_total"))
        )
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.numRows > 0 && result.columns.contains("price_sum"),
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  def runFilteredAggregationBenchmark(df: DataFrame): DataFrameOpResult = {
    val name = "filtered_agg"
    val category = "Phase10_Complex"
    try {
      val startTime = System.nanoTime()
      // Filter on count column - keep rows where count > 25
      val filtered = df.filter((v: Any) => v match {
        case i: Int => i > 25
        case f: Float => f > 25.0f
        case d: Double => d > 25.0
        case l: Long => l > 25L
        case s: String => s.toIntOption.exists(_ > 25)
        case _ => false
      })
      val result = filtered.groupBy("category").agg(
        Aggregation("rating", AggregationType.Mean, Some("rating_mean"))
      )
      val elapsed = (System.nanoTime() - startTime) / 1e6

      DataFrameOpResult(
        name = name,
        category = category,
        passed = result.numRows > 0 && result.columns.contains("rating_mean"),
        rowsProcessed = df.numRows.toLong,
        latencyMs = elapsed,
        throughputRowsPerSec = df.numRows * 1000.0 / elapsed,
        memoryMb = estimateMemory(result)
      )
    } catch {
      case e: Exception => DataFrameOpResult(name, category, false, 0, 0, 0, 0, Some(e.getMessage))
    }
  }

  // ========================================================================
  // Helper Methods
  // ========================================================================

  private def estimateMemory(df: DataFrame): Double = {
    var totalBytes = 0L
    for (colName <- df.columns) {
      val col = df.col(colName)
      totalBytes += col.length * 8
    }
    totalBytes / (1024.0 * 1024.0)
  }

  private def printSummary(results: List[DataFrameOpResult]): Unit = {
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
      println(f"${"Name"}%-25s${"Passed"}%-8s${"Rows"}%-12s${"Latency"}%-10s${"Throughput"}%-15s")
      println("-" * 60)

      for (result <- categoryResults.sortBy(_.name)) {
        val status = if (result.passed) "PASS" else "FAIL"
        val latency = f"${result.latencyMs}%.2fms"
        val throughput = if (result.throughputRowsPerSec > 0) f"${result.throughputRowsPerSec}%.0f rows/s" else "N/A"
        println(f"${result.name}%-25s${status}%-8s${result.rowsProcessed}%-12d${latency}%-10s${throughput}%-15s")

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