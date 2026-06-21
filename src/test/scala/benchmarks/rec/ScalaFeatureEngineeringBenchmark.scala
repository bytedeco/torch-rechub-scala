package benchmarks.rec

import torchrec.dataframe.*

import java.io.*
import scala.collection.mutable
import scala.util.Try

object ScalaFeatureEngineeringBenchmark {

  val ADULT = "src/main/resources/adult.csv"
  val FRAUD = "src/main/resources/fraud_data.csv"
  val OUT_DIR = "target/feature_benchmark_outputs"

  def ensureOutDir(): Unit = {
    val f = new java.io.File(OUT_DIR)
    if (!f.exists()) f.mkdirs()
  }

  case class StepResult(name: String, timeMs: Long, colsBefore: Int, colsAfter: Int)

  def time[T](label: String)(fn: => T): (T, Long) = {
    val t0 = System.nanoTime()
    val r = fn
    val t1 = System.nanoTime()
    (r, (t1 - t0) / 1000000)
  }

  def writeCSV(df: DataFrame, path: String): Unit = {
    val cols = df.columns
    val n = df.numRows

    // Diagnostic: print small preview and per-column stats when many constant columns
    try {
      println(s"[writeCSV] Writing $n rows x ${cols.length} cols to $path")
      val sampleCols = cols.take(10)
      sampleCols.foreach { c =>
        val col = df.col(c)
        val sampleVals = (0 until math.min(5, col.length)).map(i => Option(col(i)).getOrElse("null").toString)
        println(s"  col='$c' dtype=${col.dtype} sample=${sampleVals.mkString("[", ",", "]")}")
      }
    } catch { case _: Throwable => () }

    def quoteIfNeeded(s: String): String = {
      if (s.contains(",") || s.contains("\n") || s.contains("\"") || s.contains("\r")) {
        "\"" + s.replace("\"", "\"\"") + "\""
      } else s
    }

    val sb = new StringBuilder
    // write header with safe quoting
    sb.append(cols.map(quoteIfNeeded).mkString(","))
    sb.append('\n')

    for (i <- 0 until n) {
      val row = cols.map { c =>
        try {
          val v = df.col(c)(i)
          if (v == null) "" else quoteIfNeeded(v.toString)
        } catch {
          case e: Throwable =>
            // On error reading cell, write empty and log once
            try { System.err.println(s"[writeCSV] error reading cell col=$c row=$i: ${e.getMessage}") } catch { case _: Throwable => () }
            ""
        }
      }
      sb.append(row.mkString(","))
      sb.append('\n')
    }
    val pw = new PrintWriter(new File(path))
    try {
      pw.write(sb.toString())
    } finally pw.close()
  }

  def runOn(path: String, name: String, limit: Option[Int] = None): Unit = {
    ensureOutDir()
    println(s"\n=== Running feature pipeline on $path ===")
    val (rawDf, tload) = time("readCSV") { DataFrame.readCSV(path) }
    val df = limit.map(rawDf.limit).getOrElse(rawDf)
    println(s"Loaded: rows=${df.numRows} cols=${df.numCols} (t=${tload}ms)")
    // Diagnostic: show first rows and per-column quick stats to detect all-zero columns
    try {
      println("[RAW_DATA_PREVIEW]")
      df.show(5, truncate = true)
      val checkCols = df.columns.take(12)
      checkCols.foreach { c =>
        val col = df.col(c)
        val vals = (0 until math.min(50, col.length)).map(i => Option(col(i)).getOrElse("null")).toSeq
        val distinct = vals.distinct
        val allZero = distinct.forall(_.toString == "0") || (distinct.size == 1 && distinct.head.toString == "0.0")
        val statsStr = try { col.stats().toString } catch { case _: Throwable => "n/a" }
        println(s"  col=$c dtype=${col.dtype} distinctSample=${distinct.take(5)} allZeroSample=$allZero stats=$statsStr")
      }
    } catch { case _: Throwable => () }

    // Determine if this is categorical data (has String columns) or continuous data (all numeric)
    val hasStringColumns = df.columns.exists(c => df.col(c).dtype == DataType.String)
    println(s"[INFO] Data has String columns: $hasStringColumns")

    // Prepare transformers based on data type
    val transformers = mutable.ArrayBuffer[FeatureTransformer]()

    if (hasStringColumns) {
      // Pipeline for categorical data (like adult.csv)
      println("[INFO] Using categorical pipeline")

      // Scaling
      transformers += new StandardScaler()
      transformers += new MinMaxScaler()
      transformers += new MaxAbsScaler()
      transformers += new RobustScaler()

      // Numeric transforms
      transformers += new LogTransformer(1.0f)
      transformers += new LogTransformer(0.5f)

      // Categorical encoders
      transformers += new LabelEncoder()
      transformers += new OneHotEncoder(sparseOutput = false, maxCategories = Some(10))
      transformers += new TargetEncoder(0.5f)
      transformers += new CountEncoder()
      transformers += new HashEncoder(128, "murmur3")
      transformers += new OrdinalEncoder()
      transformers += new CatBoostEncoder(1.0f, 1.0f)
      transformers += new WOEEncoder(10)

      // Sequence / embedding
      transformers += new EmbeddingEncoder(8)
      transformers += new SequenceEncoder(maxLen = 20)

      // Polynomial / interactions / binned
      transformers += new PolynomialFeatures(2, interactionOnly = false)
      transformers += new InteractionFeatures(factorize = false)
      transformers += new BinnedFeatures(10, "quantile")

      // Date / text / cross / quantile
      transformers += new DateTimeFeatures()
      transformers += new TextFeatures(List(1,2))
      transformers += new CrossFeatures(50)
      transformers += new QuantileFeatures(10)

      // Dimensionality
      transformers += new SVDFeatures(16)

      // Feature selection
      transformers += new VarianceThreshold(0.0f)
      transformers += new CorrelationSelector(0.95f)
      transformers += new Chi2Selector(10)

      // More transforms
      transformers += new BinnedFeatures(5, "uniform")
      transformers += new BinnedFeatures(20, "quantile")
      transformers += new PolynomialFeatures(3, interactionOnly = false)
      transformers += new TextFeatures(List(1))
      transformers += new TextFeatures(List(1,2,3))
      transformers += new VarianceThreshold(0.01f)
      transformers += new CorrelationSelector(0.9f)
      transformers += new QuantileFeatures(5)
      transformers += new QuantileFeatures(50)
      transformers += new InteractionFeatures(factorize = true)
      transformers += new HashEncoder(32)
      transformers += new HashEncoder(256)
      transformers += new EmbeddingEncoder(4)
      transformers += new EmbeddingEncoder(16)

      // Fill up to many transformers
      for (i <- 1 to 10) transformers += new IdentityTransformer(s"identity_$i")

    } else {
      // Pipeline for continuous/numeric data (like fraud.csv - PCA transformed features)
      println("[INFO] Using continuous data pipeline (no binning/quantile)")

      // Scaling - important for continuous data, keeps float values in 0-1 or standardized
      transformers += new StandardScaler()
      transformers += new MinMaxScaler()
      transformers += new MaxAbsScaler()
      transformers += new RobustScaler()

      // Numeric transforms - log can help with skewed continuous data
      transformers += new LogTransformer(1.0f)
      transformers += new LogTransformer(0.5f)

      // Polynomial - create interaction terms for continuous features (keeps float values)
      transformers += new PolynomialFeatures(2, interactionOnly = false)
      transformers += new PolynomialFeatures(3, interactionOnly = false)

      // SVDFeatures - dimensionality reduction (keeps float values)
      transformers += new SVDFeatures(16)

      // Interaction - multiplicative combinations (keeps float values)
      transformers += new InteractionFeatures(factorize = false)
      transformers += new InteractionFeatures(factorize = true)

      // Feature selection - variance-based
      transformers += new VarianceThreshold(0.01f)
      transformers += new VarianceThreshold(0.0f)
      transformers += new CorrelationSelector(0.95f)
      transformers += new CorrelationSelector(0.9f)

      // Chi2 selector for classification
      transformers += new Chi2Selector(10)

      // Identity transformers as fillers
      for (i <- 1 to 5) transformers += new IdentityTransformer(s"identity_$i")
    }

    // Now run pipeline sequentially, collecting stats
    var current = df
    val results = mutable.ArrayBuffer[StepResult]()

    for (tr <- transformers) {
      val beforeCols = current.numCols
      println(s"Applying ${tr.name} (${tr.getClass.getSimpleName}) ...")
      val (fitted, tfit) = time("fit") { Try(tr.fit(current)).getOrElse(tr) }
      val (outDf, ttrans) = time("transform") { Try(tr.transform(current)).getOrElse(current) }
      val afterCols = outDf.numCols
      current = outDf
      results += StepResult(tr.name + "(" + tr.getClass.getSimpleName + ")", tfit + ttrans, beforeCols, afterCols)
      println(s"  -> cols: $beforeCols -> $afterCols, time=${tfit + ttrans}ms")
      // Diagnostic: detect columns that became constant/zero after this transformer
      try {
        val zeroCols = current.columns.filter { c =>
          val col = current.col(c)
          try {
            val s = col.stats()
            // consider effectively constant if std == 0 or mean == 0 and all entries equal to 0.0/0
            (s.std == 0.0) || (s.mean == 0.0 && (0 until math.min(5, col.length)).forall(i => Option(col(i)).getOrElse("0").toString == "0"))
          } catch { case _: Throwable => false }
        }
        if (zeroCols.nonEmpty) {
          println(s"  [diag] ${tr.name} produced ${zeroCols.length} constant/zero columns (showing up to 10):")
          zeroCols.take(10).foreach { c =>
            val col = current.col(c)
            val sample = (0 until math.min(5, col.length)).map(i => Option(col(i)).getOrElse("null").toString)
            val stats = try { col.stats().toString } catch { case _: Throwable => "n/a" }
            println(s"    col='$c' dtype=${col.dtype} sample=${sample.mkString("[", ",", "]")} stats=$stats")
          }
        }
      } catch { case _: Throwable => () }
    }

    // Save final engineered DF
    val outPath = s"$OUT_DIR/${name}_engineered.csv"
//    writeCSV(current, outPath)
    println(s"Saved engineered dataset to $outPath, final cols=${current.numCols}, rows=${current.numRows}")
    current.writeCSV(outPath)
    current.show(300)
    // Print summary
    println("\n=== Pipeline Summary ===")
    results.foreach { r =>
      println(s"${r.name}: time=${r.timeMs}ms cols ${r.colsBefore} -> ${r.colsAfter}")
    }
  }

  def main(args: Array[String]): Unit = {
    ensureOutDir()
    runOn(ADULT, "adult", limit = None)
    runOn(FRAUD, "fraud", limit = None)
  }

}

// Lightweight identity transformer to fill pipeline and produce additional feature steps
class IdentityTransformer(id: String) extends FeatureTransformer {
  def name: String = s"Identity($id)"
  def fit(df: DataFrame): this.type = this
  def transform(df: DataFrame): DataFrame = df
}



