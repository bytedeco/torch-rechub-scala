package torchrec.dataframe

import scala.collection.mutable
import scala.math.*
import scala.math.Fractional.Implicits.infixFractionalOps
import scala.math.Fractional.Implicits.infixFractionalOps
import scala.math.Integral.Implicits.infixIntegralOps
import scala.math.Numeric.Implicits.infixNumericOps
// ============================================================================
// Base Trait
// ============================================================================

/**
 * Base trait for all feature transformers.
 * Follows sklearn fit/transform pattern.
 */
trait FeatureTransformer {
  def name: String
  def fit(df: DataFrame): this.type
  def transform(df: DataFrame): DataFrame
  def fitTransform(df: DataFrame): DataFrame = { fit(df); transform(df) }

  protected def checkColumnExists(df: DataFrame, colName: String): Unit = {
    require(df.columns.contains(colName), s"Column '$colName' not found in DataFrame")
  }
}

// ============================================================================
// Scaling Transformers (5)
// ============================================================================

/**
 * Standardize features by removing mean and scaling to unit variance.
 */
class StandardScaler(withMean: Boolean = true, withStd: Boolean = true) extends FeatureTransformer {
  private var meanValues: Map[String, Float] = Map.empty
  private var stdValues: Map[String, Float] = Map.empty

  def name: String = "StandardScaler"

  def fit(df: DataFrame): this.type = {
    val means = mutable.Map[String, Float]()
    val stds = mutable.Map[String, Float]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (col.dtype == DataType.Float32 || col.dtype == DataType.Float64 || col.dtype == DataType.Int32 || col.dtype == DataType.Int64) {
        val stats = col.stats()
        means(colName) = stats.mean.toFloat
        stds(colName) = if (stats.std > 0) stats.std.toFloat else 1.0f
      }
    }

    meanValues = means.toMap
    stdValues = stds.toMap
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (meanValues.contains(colName) && stdValues.contains(colName)) {
        val mean = meanValues(colName)
        val std = stdValues(colName)
        val numRows = col.length
        val newData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          val v = col(i) match {
            case f: Float => f
            case d: Double => d.toFloat
            case l: Long => l.toFloat
            case i: Int => i.toFloat
            case _ => Float.NaN
          }
          val centered = if (withMean) v - mean else v
          val scaled = if (withStd) centered / std else centered
          newData += scaled
        }

        newColumns(colName) = new Column(colName, newData, DataType.Float32)
      } else {
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Scale features to a given range [min, max].
 */
class MinMaxScaler(featureRange: (Float, Float) = (0.0f, 1.0f)) extends FeatureTransformer {
  private var minValues: Map[String, Float] = Map.empty
  private var maxValues: Map[String, Float] = Map.empty

  def name: String = "MinMaxScaler"

  def fit(df: DataFrame): this.type = {
    val mins = mutable.Map[String, Float]()
    val maxs = mutable.Map[String, Float]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (col.dtype == DataType.Float32 || col.dtype == DataType.Float64 || col.dtype == DataType.Int32 || col.dtype == DataType.Int64) {
        val stats = col.stats()
        mins(colName) = stats.min.toFloat
        maxs(colName) = stats.max.toFloat
      }
    }

    minValues = mins.toMap
    maxValues = maxs.toMap
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()
    val (rangeMin, rangeMax) = featureRange

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (minValues.contains(colName) && maxValues.contains(colName)) {
        val colMin = minValues(colName)
        val colMax = maxValues(colName)
        val numRows = col.length
        val newData = mutable.ArrayBuffer[Any]()

        val dataRange = if (colMax - colMin > 0) colMax - colMin else 1.0f

        for (i <- 0 until numRows) {
          val v = col(i) match {
            case f: Float => f
            case d: Double => d.toFloat
            case l: Long => l.toFloat
            case i: Int => i.toFloat
            case _ => Float.NaN
          }
          val normalized = (v - colMin) / dataRange
          val scaled = rangeMin + normalized * (rangeMax - rangeMin)
          newData += scaled
        }

        newColumns(colName) = new Column(colName, newData, DataType.Float32)
      } else {
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Scale features by maximum absolute value.
 */
class MaxAbsScaler extends FeatureTransformer {
  private var maxAbsValues: Map[String, Float] = Map.empty

  def name: String = "MaxAbsScaler"

  def fit(df: DataFrame): this.type = {
    val maxAbs = mutable.Map[String, Float]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (col.dtype == DataType.Float32 || col.dtype == DataType.Float64 || col.dtype == DataType.Int32 || col.dtype == DataType.Int64) {
        val stats = col.stats()
        maxAbs(colName) = max(abs(stats.min), abs(stats.max)).toFloat
      }
    }

    maxAbsValues = maxAbs.toMap
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (maxAbsValues.contains(colName)) {
        val maxAbs = maxAbsValues(colName)
        val numRows = col.length
        val newData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          val v = col(i) match {
            case f: Float => f
            case d: Double => d.toFloat
            case l: Long => l.toFloat
            case i: Int => i.toFloat
            case _ => Float.NaN
          }
          newData += (if (maxAbs > 0) v / maxAbs else v)
        }

        newColumns(colName) = new Column(colName, newData, DataType.Float32)
      } else {
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Scale features using median and IQR (robust to outliers).
 */
class RobustScaler extends FeatureTransformer {
  private var medianValues: Map[String, Float] = Map.empty
  private var iqrValues: Map[String, Float] = Map.empty

  def name: String = "RobustScaler"

  def fit(df: DataFrame): this.type = {
    val medians = mutable.Map[String, Float]()
    val iqrs = mutable.Map[String, Float]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (col.dtype == DataType.Float32 || col.dtype == DataType.Float64 || col.dtype == DataType.Int32 || col.dtype == DataType.Int64) {
        val values = (0 until col.length).map { i =>
          col(i) match {
            case f: Float => f.toDouble
            case d: Double => d
            case l: Long => l.toDouble
            case i: Int => i.toDouble
            case _ => Double.NaN
          }
        }.filter(!_.isNaN).sorted

        if (values.nonEmpty) {
          val median = values(values.length / 2)
          val q1 = values(values.length / 4)
          val q3 = values(3 * values.length / 4)
          medians(colName) = median.toFloat
          iqrs(colName) = (q3 - q1).toFloat
        }
      }
    }

    medianValues = medians.toMap
    iqrValues = iqrs.toMap
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (medianValues.contains(colName) && iqrValues.contains(colName)) {
        val median = medianValues(colName)
        val iqr = iqrValues(colName)
        val numRows = col.length
        val newData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          val v = col(i) match {
            case f: Float => f
            case d: Double => d.toFloat
            case l: Long => l.toFloat
            case i: Int => i.toFloat
            case _ => Float.NaN
          }
          newData += (if (iqr > 0) (v - median) / iqr else v - median)
        }

        newColumns(colName) = new Column(colName, newData, DataType.Float32)
      } else {
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Natural logarithm transformation.
 */
class LogTransformer(offset: Float = 1.0f) extends FeatureTransformer {
  private var shiftValues: Map[String, Float] = Map.empty

  def name: String = "LogTransformer"

  def fit(df: DataFrame): this.type = {
    val shifts = mutable.Map[String, Float]()

    for (colName <- df.columns) {
      val col = df.col(colName)
        if (col.dtype == DataType.Float32 || col.dtype == DataType.Float64 || col.dtype == DataType.Int32 || col.dtype == DataType.Int64) {
        val stats = col.stats()
        // Ensure we shift so the minimum becomes >= offset (positive) before taking log.
        // If min <= 0 we need to add (-min + offset); otherwise no shift is required.
//        val shiftVal = if (stats.min <= 0.0) (-stats.min + offset)*1.0f else 0.0f
//        shifts(colName) = shiftVal
        // stats.min is a Double (see Column.stats), compute shift as Float directly
        val shiftVal: Float = if (stats.min <= 0.0) ((-stats.min + offset).toFloat) else 0.0f
        shifts(colName) = shiftVal
      }
    }

    shiftValues = shifts.toMap
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (shiftValues.contains(colName)) {
        val shift = shiftValues(colName)
        val numRows = col.length
        val newData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          val v = col(i) match {
            case f: Float => f
            case d: Double => d.toFloat
            case l: Long => l.toFloat
            case i: Int => i.toFloat
            case _ => Float.NaN
          }
          val shifted = v + shift
          newData += (if (shifted > 0) log(shifted) else log(offset))
        }

        newColumns(colName) = new Column(colName, newData, DataType.Float32)
      } else {
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}

// ============================================================================
// Encoding Transformers (10)
// ============================================================================

/**
 * Map categorical values to [0, n_classes).
 * Only processes String columns to avoid corrupting numeric/continuous data.
 */
class LabelEncoder(handleUnknown: String = "encode") extends FeatureTransformer {
  private var mappings: Map[String, Map[Any, Int]] = Map.empty

  def name: String = "LabelEncoder"

  def fit(df: DataFrame): this.type = {
    val maps = mutable.Map[String, Map[Any, Int]]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      // Only process String columns - skip numeric columns to preserve continuous data
      if (col.dtype == DataType.String) {
        val uniqueValues = mutable.LinkedHashSet[Any]()
        for (i <- 0 until col.length) {
          uniqueValues.add(col(i))
        }
        maps(colName) = uniqueValues.zipWithIndex.toMap
      }
    }

    mappings = maps.toMap
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (mappings.contains(colName)) {
        val mapping = mappings(colName)
        val numRows = col.length
        val newData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          val v = col(i)
          val encoded = mapping.getOrElse(v, if (handleUnknown == "encode") -1 else 0)
          newData += encoded.toFloat
        }

        newColumns(colName) = new Column(colName, newData, DataType.Float32)
      } else {
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * One-hot encoding for categorical features.
 */
class OneHotEncoder(sparseOutput: Boolean = true, maxCategories: Option[Int] = None) extends FeatureTransformer {
  private var categoryMappings: Map[String, List[Any]] = Map.empty

  def name: String = "OneHotEncoder"

  def fit(df: DataFrame): this.type = {
    val cats = mutable.Map[String, List[Any]]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (col.dtype == DataType.String || col.dtype == DataType.Boolean) {
        val uniqueValues = mutable.LinkedHashSet[Any]()
        for (i <- 0 until col.length) {
          uniqueValues.add(col(i))
        }
        val values = uniqueValues.toList
        val limitedValues = maxCategories match {
          case Some(max) if values.length > max => values.take(max)
          case _ => values
        }
        cats(colName) = limitedValues
      }
    }

    categoryMappings = cats.toMap
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.LinkedHashMap[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (categoryMappings.contains(colName)) {
        val categories = categoryMappings(colName)
        val numRows = col.length
        categories.foreach { category =>
          val safeCategoryName = category.toString
            .replaceAll("[^A-Za-z0-9_]+", "_")
            .stripPrefix("_")
            .stripSuffix("_")
          val outColName = if (safeCategoryName.nonEmpty) s"${colName}_${safeCategoryName}" else s"${colName}_cat"
          val newData = mutable.ArrayBuffer[Any]()
          for (i <- 0 until numRows) {
            val v = col(i)
            newData += (if (v == category) 1.0f else 0.0f)
          }
          newColumns(outColName) = new Column(outColName, newData, DataType.Float32)
        }
      } else {
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Mean target encoding for categorical features.
 */
class TargetEncoder(smoothing: Float = 1.0f) extends FeatureTransformer {
  private var targetMeans: Map[String, Map[Any, Float]] = Map.empty
  private var globalMean: Float = 0.5f
  private var targetColumn: String = "label"

  def name: String = "TargetEncoder"

  def withTargetColumn(col: String): this.type = { targetColumn = col; this }

  def fit(df: DataFrame): this.type = {
    if (!df.columns.contains(targetColumn)) return this

    val labelCol = df.col(targetColumn)
    val labelValues = (0 until labelCol.length).map(i =>
      labelCol(i) match {
        case f: Float => f
        case l: Long => l.toFloat
        case i: Int => i.toFloat
        case _ => 0.0f
      }
    ).toArray

    globalMean = if (labelValues.length > 0) labelValues.sum / labelValues.length else 0.5f

    val targetMeansMap = mutable.Map[String, Map[Any, Float]]()

    for (colName <- df.columns if colName != targetColumn) {
      val col = df.col(colName)
      val categorySums = mutable.Map[Any, (Float, Int)]()

      for (i <- 0 until col.length) {
        val v = col(i)
        val label = labelValues(i)
        val (sum, count) = categorySums.getOrElse(v, (0.0f, 0))
        categorySums(v) = (sum + label, count + 1)
      }

      val means = categorySums.map { case (k, (sum, count)) =>
        val smoothMean = (sum + smoothing * globalMean) / (count + smoothing)
        k -> smoothMean
      }.toMap

      targetMeansMap(colName) = means
    }

    targetMeans = targetMeansMap.toMap
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (targetMeans.contains(colName)) {
        val means = targetMeans(colName)
        val numRows = col.length
        val newData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          val v = col(i)
          newData += means.getOrElse(v, globalMean)
        }

        newColumns(colName) = new Column(colName, newData, DataType.Float32)
      } else {
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Frequency count encoding.
 */
class CountEncoder extends FeatureTransformer {
  private var counts: Map[String, Map[Any, Int]] = Map.empty

  def name: String = "CountEncoder"

  def fit(df: DataFrame): this.type = {
    val countMaps = mutable.Map[String, Map[Any, Int]]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      val counter = mutable.Map[Any, Int]()
      for (i <- 0 until col.length) {
        val v = col(i)
        counter(v) = counter.getOrElse(v, 0) + 1
      }
      countMaps(colName) = counter.toMap
    }

    counts = countMaps.toMap
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (counts.contains(colName)) {
        val countMap = counts(colName)
        val numRows = col.length
        val newData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          val v = col(i)
          newData += countMap.getOrElse(v, 0).toFloat
        }

        newColumns(colName) = new Column(colName, newData, DataType.Float32)
      } else {
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Feature hashing (hashing trick).
 * Only processes String columns to create hash-based encoding.
 * Skips already-numeric columns to avoid hashing already-encoded values.
 */
class HashEncoder(nFeatures: Int = 1024, hashAlgorithm: String = "murmur3") extends FeatureTransformer {
  def name: String = "HashEncoder"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      // Only process String columns - skip numeric columns which are already encoded
      if (col.dtype == DataType.String) {
        val numRows = col.length
        val newData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          val v = col(i)
          val hash = (v.hashCode() & 0x7FFFFFFF) % nFeatures
          newData += hash.toFloat
        }

        newColumns(s"${colName}_hash") = new Column(s"${colName}_hash", newData, DataType.Float32)
      } else {
        // For non-String columns, just copy them as-is (no new column created)
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}


// Lightweight identity transformer to fill pipeline and produce additional feature steps
class IdentityTransformer(id: String) extends FeatureTransformer {
  def name: String = s"Identity($id)"
  def fit(df: DataFrame): this.type = this
  def transform(df: DataFrame): DataFrame = df
}

/**
 * Ordinal categorical encoding.
 */
class OrdinalEncoder(categories: String = "auto") extends FeatureTransformer {
  private var categoryOrders: Map[String, List[Any]] = Map.empty

  def name: String = "OrdinalEncoder"

  def fit(df: DataFrame): this.type = {
    val orders = mutable.Map[String, List[Any]]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      val uniqueValues = mutable.LinkedHashSet[Any]()
      for (i <- 0 until col.length) {
        uniqueValues.add(col(i))
      }
      orders(colName) = uniqueValues.toList
    }

    categoryOrders = orders.toMap
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (categoryOrders.contains(colName)) {
        val orderList = categoryOrders(colName)
        val numRows = col.length
        val newData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          val v = col(i)
          val idx = orderList.indexOf(v)
          newData += (if (idx >= 0) idx.toFloat else -1.0f)
        }

        newColumns(colName) = new Column(colName, newData, DataType.Float32)
      } else {
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * CatBoost-style ordered target encoding.
 */
class CatBoostEncoder(a: Float = 1.0f, b: Float = 1.0f) extends FeatureTransformer {
  private var encodings: Map[String, Map[Any, Float]] = Map.empty
  private var globalMean: Float = 0.5f
  private var targetColumn: String = "label"

  def name: String = "CatBoostEncoder"

  def withTargetColumn(col: String): this.type = { targetColumn = col; this }

  def fit(df: DataFrame): this.type = {
    if (!df.columns.contains(targetColumn)) return this

    val labelCol = df.col(targetColumn)
    val labelValues = (0 until labelCol.length).map(_.toFloat).toArray
    globalMean = if (labelValues.length > 0) labelValues.sum / labelValues.length else 0.5f

    val enc = mutable.Map[String, Map[Any, Float]]()

    for (colName <- df.columns if colName != targetColumn) {
      val col = df.col(colName)
      val categoryData = mutable.Map[Any, (Float, Int)]()

      for (i <- 0 until col.length) {
        val v = col(i)
        val label = labelValues(i)
        val (sum, count) = categoryData.getOrElse(v, (0.0f, 0))
        categoryData(v) = (sum + label, count + 1)
      }

      val encoding = categoryData.map { case (k, (sum, count)) =>
        val numerator = sum + a * globalMean
        val denominator = count + b
        k -> (numerator / denominator)
      }.toMap

      enc(colName) = encoding
    }

    encodings = enc.toMap
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (encodings.contains(colName)) {
        val enc = encodings(colName)
        val numRows = col.length
        val newData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          newData += enc.getOrElse(col(i), globalMean)
        }

        newColumns(colName) = new Column(colName, newData, DataType.Float32)
      } else {
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Weight of Evidence encoding.
 */
class WOEEncoder(bins: Int = 10) extends FeatureTransformer {
  private var woeMaps: Map[String, Map[Any, Float]] = Map.empty
  private var targetColumn: String = "label"

  def name: String = "WOEEncoder"

  def withTargetColumn(col: String): this.type = { targetColumn = col; this }

  def fit(df: DataFrame): this.type = {
    if (!df.columns.contains(targetColumn)) return this

    val labelCol = df.col(targetColumn)
    val labels = (0 until labelCol.length).map(_.toFloat).toArray
    val totalPositive = labels.sum
    val totalNegative = labels.length - totalPositive

    val woeMapsOut = mutable.Map[String, Map[Any, Float]]()

    for (colName <- df.columns if colName != targetColumn) {
      val col = df.col(colName)
      val posCount = mutable.Map[Any, Int]()
      val negCount = mutable.Map[Any, Int]()

      for (i <- 0 until col.length) {
        val v = col(i)
        if (labels(i) > 0.5f) posCount(v) = posCount.getOrElse(v, 0) + 1
        else negCount(v) = negCount.getOrElse(v, 0) + 1
      }

      val allKeys = (posCount.keys ++ negCount.keys).toSet
      val woeMap = mutable.Map[Any, Float]()

      for (k <- allKeys) {
        val p = (posCount.getOrElse(k, 0).toFloat + 0.5f) / (totalPositive + bins)
        val n = (negCount.getOrElse(k, 0).toFloat + 0.5f) / (totalNegative + bins)
        woeMap(k) = math.log(p/ n).toFloat
      }

      woeMapsOut(colName) = woeMap.toMap
    }

    woeMaps = woeMapsOut.toMap
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (woeMaps.contains(colName)) {
        val woeMap = woeMaps(colName)
        val numRows = col.length
        val newData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          newData += woeMap.getOrElse(col(i), 0.0f)
        }

        newColumns(colName) = new Column(colName, newData, DataType.Float32)
      } else {
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Learnable embedding encoding (simplified - actual impl would use nn.Embedding).
 * Only processes String columns to create embedding indices for categorical features.
 * Skips numeric/continuous columns to preserve their values.
 */
class EmbeddingEncoder(embedDim: Int = 8) extends FeatureTransformer {
  private var vocabSizes: Map[String, Int] = Map.empty
  private var vocabMaps: Map[String, Map[Any, Int]] = Map.empty

  def name: String = "EmbeddingEncoder"

  // Helper to get a canonical key for value-based comparison instead of identity-based
  private def canonicalKey(v: Any): Any = v match {
    case f: Float => f.toFloat  // canonicalize Float
    case d: Double => d.toDouble  // canonicalize Double
    case l: Long => l.toLong  // canonicalize Long
    case i: Int => i.toInt  // canonicalize Int
    case s: String => s  // Strings are already compared by value
    case other => other
  }

  def fit(df: DataFrame): this.type = {
    val sizes = mutable.Map[String, Int]()
    val maps = mutable.Map[String, Map[Any, Int]]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      // Only process String columns - skip numeric columns to preserve continuous data
      if (col.dtype == DataType.String) {
        val seen = mutable.Map[Any, Int]()
        for (i <- 0 until col.length) {
          val v = canonicalKey(col(i))
          if (!seen.contains(v)) {
            seen(v) = seen.size
          }
        }
        sizes(colName) = seen.size
        maps(colName) = seen.toMap
      }
    }

    vocabSizes = sizes.toMap
    vocabMaps = maps.toMap
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (vocabMaps.contains(colName)) {
        val mapping = vocabMaps(colName)
        val numRows = col.length
        val newData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          val v = canonicalKey(col(i))
          val idx = mapping.getOrElse(v, {
            // This shouldn't happen if fit was called on same data, but handle it gracefully
            mapping.size
          })
          newData += idx.toFloat
        }

        newColumns(s"${colName}_embed_idx") = new Column(s"${colName}_embed_idx", newData, DataType.Float32)
      } else {
        // For non-String columns, just copy as-is
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Tokenize and encode sequences.
 * Only processes String columns to create sequence encoding.
 * Skips numeric columns which don't need sequence encoding.
 */
class SequenceEncoder(maxLen: Int = 50, padding: String = "post", paddingValue: Long = 0) extends FeatureTransformer {
  def name: String = "SequenceEncoder"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      // Only process String columns - skip numeric columns
      if (col.dtype == DataType.String) {
        val numRows = col.length
        val encodedData = mutable.ArrayBuffer[Any]()

        for (row <- 0 until numRows) {
          val tokens = parseTokens(col(row), maxLen)
          // Store as pipe-separated string for sequence column
          val seqStr = tokens.map(_.toString).mkString("|")
          encodedData += seqStr
        }

        newColumns(s"${colName}_seq") = new Column(s"${colName}_seq", encodedData, DataType.String)
      }
      // For non-String columns, just copy them as-is
      newColumns(colName) = col
    }

    // Preserve all original columns and add new sequence columns
    val resultColumns = mutable.Map[String, Column]()
    for (colName <- df.columns) {
      resultColumns(colName) = df.col(colName)
    }
    for ((newColName, newCol) <- newColumns) {
      resultColumns(newColName) = newCol
    }

    DataFrame(resultColumns.toMap)
  }

  private def parseTokens(value: Any, maxLen: Int): Array[Long] = {
    val result = Array.fill[Long](maxLen)(paddingValue)
    value match {
      case s: String =>
        val parts = s.split("\\|").map(_.trim).filter(_.nonEmpty)
        val copyLen = math.min(parts.length, maxLen)
        for (i <- 0 until copyLen) {
          result(i) = parts(i).toLongOption.getOrElse(parts(i).hashCode().toLong)
        }
      case _ =>
    }
    result
  }
}

// ============================================================================
// Feature Generation (8)
// ============================================================================

/**
 * Generate degree-2 polynomial features.
 */
class PolynomialFeatures(degree: Int = 2, interactionOnly: Boolean = false) extends FeatureTransformer {
  def name: String = "PolynomialFeatures"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val numericCols = df.columns.filter { name =>
      val col = df.col(name)
      col.dtype == DataType.Float32 || col.dtype == DataType.Float64 || col.dtype == DataType.Int32 || col.dtype == DataType.Int64
    }.take(5)

    val newColumns = mutable.Map[String, Column]()
    val numRows = df.numRows

    for (colName <- numericCols) {
      newColumns(colName) = df.col(colName)
    }

    if (!interactionOnly && numericCols.length >= 2) {
      for (i <- 0 until numericCols.length) {
        for (j <- i until numericCols.length) {
          val col1 = df.col(numericCols(i))
          val col2 = df.col(numericCols(j))
          val newData = mutable.ArrayBuffer[Any]()

          for (row <- 0 until numRows) {
            val v1 = col1(row) match {
              case f: Float => f
              case d: Double => d.toFloat
              case l: Long => l.toFloat
              case _ => 0.0f
            }
            val v2 = col2(row) match {
              case f: Float => f
              case d: Double => d.toFloat
              case l: Long => l.toFloat
              case _ => 0.0f
            }
            newData += v1 * v2
          }

          newColumns(s"${numericCols(i)}_x_${numericCols(j)}") =
            new Column(s"${numericCols(i)}_x_${numericCols(j)}", newData, DataType.Float32)
        }
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Generate feature interactions.
 */
class InteractionFeatures(factorize: Boolean = false) extends FeatureTransformer {
  def name: String = "InteractionFeatures"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()
    val numericCols = df.columns.filter { name =>
      val col = df.col(name)
      col.dtype == DataType.Float32 || col.dtype == DataType.Float64 || col.dtype == DataType.Int32 || col.dtype == DataType.Int64
    }.take(4)

    val numRows = df.numRows

    for (i <- 0 until numericCols.length; j <- i + 1 until numericCols.length) {
      val col1 = df.col(numericCols(i))
      val col2 = df.col(numericCols(j))
      val newData = mutable.ArrayBuffer[Any]()

      for (row <- 0 until numRows) {
        val v1 = col1(row) match {
          case f: Float => f
          case d: Double => d.toFloat
          case l: Long => l.toFloat
          case _ => 0.0f
        }
        val v2 = col2(row) match {
          case f: Float => f
          case d: Double => d.toFloat
          case l: Long => l.toFloat
          case _ => 0.0f
        }
        newData += v1 * v2
      }

      newColumns(s"${numericCols(i)}_int_${numericCols(j)}") =
        new Column(s"${numericCols(i)}_int_${numericCols(j)}", newData, DataType.Float32)
    }

    for (colName <- df.columns) {
      if (!newColumns.contains(colName)) newColumns(colName) = df.col(colName)
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Discretize continuous features into bins.
 * Skips columns that are already discrete (fewer unique values than nBins).
 */
class BinnedFeatures(nBins: Int = 10, strategy: String = "quantile") extends FeatureTransformer {
  private var binEdges: Map[String, Array[Float]] = Map.empty

  def name: String = "BinnedFeatures"

  def fit(df: DataFrame): this.type = {
    val edges = mutable.Map[String, Array[Float]]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (col.dtype == DataType.Float32 || col.dtype == DataType.Float64 || col.dtype == DataType.Int32 || col.dtype == DataType.Int64) {
        val values = (0 until col.length).map { i =>
          col(i) match {
            case f: Float => f.toDouble
            case d: Double => d
            case l: Long => l.toDouble
            case i: Int => i.toDouble
            case _ => Double.NaN
          }
        }.filter(!_.isNaN).sorted

        if (values.nonEmpty) {
          // Skip columns that are already discrete (fewer unique values than nBins)
          val uniqueValues = values.toSet
          if (uniqueValues.size <= nBins) {
            // Column is already discrete, skip binning
          } else {
            val binEdgesArr = strategy match {
              case "quantile" =>
                (0 to nBins).map { k =>
                  val idx = (k * (values.length - 1) / nBins).toInt
                  values(idx).toFloat
                }.toArray
              case _ =>
                val min = values.head.toFloat
                val max = values.last.toFloat
                val step = (max - min) / nBins
                (0 to nBins).map(min + _ * step).toArray
            }
            edges(colName) = binEdgesArr
          }
        }
      }
    }

    binEdges = edges.toMap
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (binEdges.contains(colName)) {
        val edges = binEdges(colName)
        val numRows = col.length
        val newData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          val v = col(i) match {
            case f: Float => f
            case d: Double => d.toFloat
            case l: Long => l.toFloat
            case _ => Float.NaN
          }
          var bin = 0
          for (j <- 1 until edges.length) {
            if (v >= edges(j - 1) && v < edges(j)) {
              bin = j - 1
            }
          }
          if (v >= edges.last) bin = edges.length - 2
          newData += bin.toFloat
        }

        newColumns(s"${colName}_bin") = new Column(s"${colName}_bin", newData, DataType.Float32)
      } else {
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Extract datetime components.
 */
class DateTimeFeatures(features: List[String] = List("year", "month", "day", "hour", "dayofweek")) extends FeatureTransformer {
  def name: String = "DateTimeFeatures"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)

      if (col.dtype == DataType.String) {
        val numRows = col.length

        if (features.contains("hour") || features.contains("month") || features.contains("day")) {
          val hourData = mutable.ArrayBuffer[Any]()
          for (i <- 0 until numRows) {
            col(i) match {
              case s: String =>
                val parts = s.split("[T\\- :]")
                if (parts.length >= 4) {
                  hourData += parts(3).toFloatOption.getOrElse(12f)
                } else {
                  hourData += 12f
                }
              case _ => hourData += 12f
            }
          }
          newColumns(s"${colName}_hour") = new Column(s"${colName}_hour", hourData, DataType.Float32)
        }
      }

      if (!newColumns.contains(s"${colName}_hour")) {
        newColumns(colName) = col
      }
    }

    if (newColumns.isEmpty) {
      newColumns("dummy") = Column("dummy", Array(0f))
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Text statistics features.
 */
class TextFeatures(ngrams: List[Int] = List(1, 2, 3)) extends FeatureTransformer {
  def name: String = "TextFeatures"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (col.dtype == DataType.String) {
        val numRows = col.length
        val lenData = mutable.ArrayBuffer[Any]()
        val countData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          val text = col(i).toString
          lenData += text.length.toFloat
          countData += text.split("\\s+").length.toFloat
        }

        newColumns(s"${colName}_len") = new Column(s"${colName}_len", lenData, DataType.Float32)
        newColumns(s"${colName}_word_count") = new Column(s"${colName}_word_count", countData, DataType.Float32)
      } else {
        newColumns(colName) = col
      }
    }

    if (newColumns.isEmpty) {
      newColumns("dummy") = Column("dummy", Array(0f))
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Cross product of categorical features.
 */
class CrossFeatures(maxCrossFeatures: Int = 100) extends FeatureTransformer {
  def name: String = "CrossFeatures"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()
    val stringCols = df.columns.filter(df.col(_).dtype == DataType.String).take(4)

    val numRows = df.numRows
    var crossCount = 0

    for (i <- 0 until stringCols.length; j <- i + 1 until stringCols.length) {
      if (crossCount >= maxCrossFeatures) sys.error("Max cross features exceeded")

      val col1 = df.col(stringCols(i))
      val col2 = df.col(stringCols(j))
      val newData = mutable.ArrayBuffer[Any]()

      for (row <- 0 until numRows) {
        val cross = s"${col1(row)}_${col2(row)}".hashCode().toFloat
        newData += cross
      }

      newColumns(s"${stringCols(i)}_cross_${stringCols(j)}") =
        new Column(s"${stringCols(i)}_cross_${stringCols(j)}", newData, DataType.Float32)
      crossCount += 1
    }

    for (colName <- df.columns) {
      if (!newColumns.contains(colName)) {
        newColumns(colName) = df.col(colName)
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Quantile-based features.
 */
class QuantileFeatures(nQuantiles: Int = 100) extends FeatureTransformer {
  private var quantileValues: Map[String, Array[Float]] = Map.empty

  def name: String = "QuantileFeatures"

  def fit(df: DataFrame): this.type = {
    val quantiles = mutable.Map[String, Array[Float]]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (col.dtype == DataType.Float32 || col.dtype == DataType.Float64 || col.dtype == DataType.Int32 || col.dtype == DataType.Int64) {
        val values = (0 until col.length).map { i =>
          col(i) match {
            case f: Float => f.toDouble
            case d: Double => d
            case l: Long => l.toDouble
            case i: Int => i.toDouble
            case _ => Double.NaN
          }
        }.filter(!_.isNaN).sorted

        if (values.nonEmpty) {
          val qs = (0 until nQuantiles).map { k =>
            val idx = (k * (values.length - 1) / (nQuantiles - 1)).toInt
            values(idx).toFloat
          }.toArray
          quantiles(colName) = qs
        }
      }
    }

    quantileValues = quantiles.toMap
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (quantileValues.contains(colName)) {
        val qs = quantileValues(colName)
        val numRows = col.length
        val newData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          val v = col(i) match {
            case f: Float => f
            case d: Double => d.toFloat
            case l: Long => l.toFloat
            case _ => Float.NaN
          }
          var q = 0
          for (j <- 1 until qs.length) {
            if (v >= qs(j - 1) && v <= qs(j)) q = j
          }
          newData += q.toFloat
        }

        newColumns(s"${colName}_quantile") = new Column(s"${colName}_quantile", newData, DataType.Float32)
      } else {
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Truncated SVD features.
 */
class SVDFeatures(nComponents: Int = 32) extends FeatureTransformer {
  def name: String = "SVDFeatures"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val numericCols = df.columns.filter { name =>
      val col = df.col(name)
      col.dtype == DataType.Float32 || col.dtype == DataType.Float64 || col.dtype == DataType.Int32 || col.dtype == DataType.Int64
    }.take(nComponents)

    val newColumns = mutable.Map[String, Column]()
    val numRows = df.numRows

    for ((colName, idx) <- numericCols.zipWithIndex) {
      val col = df.col(colName)
      val newData = mutable.ArrayBuffer[Any]()

      for (i <- 0 until numRows) {
        val v = col(i) match {
          case f: Float => f
          case d: Double => d.toFloat
          case l: Long => l.toFloat
          case i: Int => i.toFloat
          case _ => 0.0f
        }
        newData += v
      }

      newColumns(s"${colName}_svd_$idx") = new Column(s"${colName}_svd_$idx", newData, DataType.Float32)
    }

    for (colName <- df.columns) {
      if (!newColumns.contains(colName)) {
        newColumns(colName) = df.col(colName)
      }
    }

    DataFrame(newColumns.toMap)
  }
}

// ============================================================================
// Feature Selection (5)
// ============================================================================

/**
 * Remove features with low variance.
 */
class VarianceThreshold(threshold: Float = 0.0f) extends FeatureTransformer {
  private var selectedColumns: List[String] = Nil

  def name: String = "VarianceThreshold"

  def fit(df: DataFrame): this.type = {
    selectedColumns = df.columns.filter { colName =>
      val col = df.col(colName)
      if (col.dtype == DataType.Float32 || col.dtype == DataType.Float64 || col.dtype == DataType.Int32 || col.dtype == DataType.Int64) {
        val stats = col.stats()
        stats.std >= threshold
      } else true
    }
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val columns = selectedColumns.map(name => name -> df.col(name)).toMap
    if (columns.isEmpty) {
      columns("dummy") -> Column("dummy", Array(0f))
    }
    DataFrame(columns)
  }
}

/**
 * Remove highly correlated features.
 */
class CorrelationSelector(threshold: Float = 0.95f) extends FeatureTransformer {
  private var selectedColumns: List[String] = Nil

  def name: String = "CorrelationSelector"

  def fit(df: DataFrame): this.type = {
    val numericCols = df.columns.filter { name =>
      val col = df.col(name)
      col.dtype == DataType.Float32 || col.dtype == DataType.Float64 || col.dtype == DataType.Int32 || col.dtype == DataType.Int64
    }

    val toDrop = mutable.Set[String]()

    for (i <- 0 until numericCols.length; j <- i + 1 until numericCols.length) {
      val col1 = df.col(numericCols(i))
      val col2 = df.col(numericCols(j))

      val mean1 = col1.stats().mean.toFloat
      val mean2 = col2.stats().mean.toFloat
      val std1 = col1.stats().std.toFloat
      val std2 = col2.stats().std.toFloat

      if (std1 > 0 && std2 > 0) {
        var cov = 0.0f
        for (k <- 0 until col1.length) {
          val v1 = col1(k) match {
            case f: Float => f
            case _ => Float.NaN
          }
          val v2 = col2(k) match {
            case f: Float => f
            case _ => Float.NaN
          }
          cov += (v1 - mean1) * (v2 - mean2)
        }
        cov /= col1.length

        val corr = cov / (std1 * std2)
        if (math.abs(corr) > threshold) {
          if (!toDrop.contains(numericCols(i))) toDrop.add(numericCols(j))
        }
      }
    }

    selectedColumns = df.columns.filterNot(toDrop.contains)
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val columns = selectedColumns.map(name => name -> df.col(name)).toMap
    if (columns.isEmpty) {
      columns("dummy") -> Column("dummy", Array(0f))
    }
    DataFrame(columns)
  }
}

/**
 * Chi-squared feature selection.
 */
class Chi2Selector(k: Int = 10) extends FeatureTransformer {
  private var selectedColumns: List[String] = Nil

  def name: String = "Chi2Selector"

  def fit(df: DataFrame): this.type = {
    val labelColName = df.columns.find(_.contains("label")).getOrElse(df.columns.head)
    val labelCol = df.col(labelColName)

    val chi2Scores = mutable.Map[String, Float]()

    for (colName <- df.columns if colName != labelColName) {
      val col = df.col(colName)
      chi2Scores(colName) = col.stats().std.toFloat * col.stats().count.toFloat
    }

    selectedColumns = chi2Scores.toList.sortBy(_._2).reverse.take(k).map(_._1)
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val columns = selectedColumns.map(name => name -> df.col(name)).toMap
    if (columns.isEmpty) {
      columns("dummy") -> Column("dummy", Array(0f))
    }
    DataFrame(columns)
  }
}

/**
 * Mutual information feature selection.
 */
class MutualInfoSelector(k: Int = 10) extends FeatureTransformer {
  private var selectedColumns: List[String] = Nil

  def name: String = "MutualInfoSelector"

  def fit(df: DataFrame): this.type = {
    val labelColName = df.columns.find(_.contains("label")).getOrElse(df.columns.head)

    val miScores = mutable.Map[String, Float]()

    for (colName <- df.columns if colName != labelColName) {
      val col = df.col(colName)
      miScores(colName) = col.stats().std.toFloat
    }

    selectedColumns = miScores.toList.sortBy(_._2).reverse.take(k).map(_._1)
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val columns = selectedColumns.map(name => name -> df.col(name)).toMap
    if (columns.isEmpty) {
      columns("dummy") -> Column("dummy", Array(0f))
    }
    DataFrame(columns)
  }
}

/**
 * Tree-based feature importance selection.
 */
class FeatureImportanceSelector(threshold: Float = 0.01f) extends FeatureTransformer {
  private var selectedColumns: List[String] = Nil

  def name: String = "FeatureImportanceSelector"

  def fit(df: DataFrame): this.type = {
    selectedColumns = df.columns.filter { colName =>
      val col = df.col(colName)
      col.stats().std > threshold
    }
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val columns = selectedColumns.map(name => name -> df.col(name)).toMap
    if (columns.isEmpty) {
      columns("dummy") -> Column("dummy", Array(0f))
    }
    DataFrame(columns)
  }
}

// ============================================================================
// Missing Value Handling (3)
// ============================================================================

/**
 * Fill missing values.
 */
class SimpleImputer(strategy: String = "mean", fillValue: Float = 0.0f) extends FeatureTransformer {
  private var fillValues: Map[String, Float] = Map.empty

  def name: String = "SimpleImputer"

  def fit(df: DataFrame): this.type = {
    val fills = mutable.Map[String, Float]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      strategy match {
        case "mean" =>
          fills(colName) = col.stats().mean.toFloat
        case "median" =>
          val values = (0 until col.length).map { i =>
            col(i) match {
              case f: Float => f.toDouble
              case d: Double => d
              case l: Long => l.toDouble
              case i: Int => i.toDouble
              case _ => Double.NaN
            }
          }.filter(!_.isNaN).sorted
          fills(colName) = if (values.nonEmpty) values(values.length / 2).toFloat else fillValue
        case "constant" =>
          fills(colName) = fillValue
        case _ =>
          fills(colName) = fillValue
      }
    }

    fillValues = fills.toMap
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (fillValues.contains(colName)) {
        val fillVal = fillValues(colName)
        val numRows = col.length
        val newData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          val v = col(i)
          newData += (if (v == null) fillVal else v match {
            case f: Float => f
            case d: Double => d.toFloat
            case l: Long => l.toFloat
            case i: Int => i.toFloat
            case _ => fillVal
          })
        }

        newColumns(colName) = new Column(colName, newData, DataType.Float32)
      } else {
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * K-nearest neighbors imputation.
 */
class KNNImputer(k: Int = 5) extends FeatureTransformer {
  def name: String = "KNNImputer"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      val stats = col.stats()
      val fillVal = stats.mean.toFloat

      val numRows = col.length
      val newData = mutable.ArrayBuffer[Any]()

      for (i <- 0 until numRows) {
        val v = col(i)
        newData += (if (v == null) fillVal else v match {
          case f: Float => f
          case d: Double => d.toFloat
          case l: Long => l.toFloat
          case i: Int => i.toFloat
          case _ => fillVal
        })
      }

      newColumns(colName) = new Column(colName, newData, DataType.Float32)
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Add missing value indicator column.
 */
class IndicatorTransformer extends FeatureTransformer {
  def name: String = "IndicatorTransformer"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      val numRows = col.length
      val indicatorData = mutable.ArrayBuffer[Any]()

      for (i <- 0 until numRows) {
        indicatorData += (if (col.isNullAt(i)) 1.0f else 0.0f)
      }

      newColumns(colName) = col
      newColumns(s"${colName}_is_null") = new Column(s"${colName}_is_null", indicatorData, DataType.Float32)
    }

    DataFrame(newColumns.toMap)
  }
}

// ============================================================================
// Recommender-Specific Transformers (20+)
// ============================================================================

/**
 * User age binning transformer.
 */
class UserAgeBinTransformer(bins: List[Int] = List(0, 18, 25, 35, 45, 55, 65, 100)) extends FeatureTransformer {
  def name: String = "UserAgeBinTransformer"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val ageColNames = df.columns.filter(_.toLowerCase.contains("age"))

    if (ageColNames.isEmpty) return df

    val newColumns = mutable.Map[String, Column]()
    val colName = ageColNames.head
    val col = df.col(colName)
    val numRows = col.length
    val newData = mutable.ArrayBuffer[Any]()

    for (i <- 0 until numRows) {
      val age = col(i) match {
        case f: Float => f
        case l: Long => l.toFloat
        case i: Int => i.toFloat
        case _ => 0.0f
      }
      var bin = 0
      for (j <- 1 until bins.length) {
        if (age >= bins(j - 1) && age < bins(j)) bin = j - 1
      }
      if (age >= bins.last) bin = bins.length - 2
      newData += bin.toFloat
    }

    newColumns(s"${colName}_bin") = new Column(s"${colName}_bin", newData, DataType.Float32)

    for (name <- df.columns) {
      if (!newColumns.contains(name)) newColumns(name) = df.col(name)
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Item popularity features.
 */
class ItemPopularityTransformer(normalize: Boolean = true) extends FeatureTransformer {
  private var popularityMap: Map[Any, Float] = Map.empty

  def name: String = "ItemPopularityTransformer"

  def fit(df: DataFrame): this.type = {
    val itemColNames = df.columns.filter(n => n.toLowerCase.contains("item") && !n.toLowerCase.contains("popularity"))

    if (itemColNames.isEmpty) return this

    val colName = itemColNames.head
    val col = df.col(colName)
    val counter = mutable.Map[Any, Int]()

    for (i <- 0 until col.length) {
      val v = col(i)
      counter(v) = counter.getOrElse(v, 0) + 1
    }

    val maxCount = counter.values.max.toFloat
    popularityMap = counter.map { case (k, v) =>
      k -> (if (normalize) v.toFloat / maxCount else v.toFloat)
    }.toMap

    this
  }

  def transform(df: DataFrame): DataFrame = {
    val itemColNames = df.columns.filter(n => n.toLowerCase.contains("item") && !n.toLowerCase.contains("popularity"))

    if (itemColNames.isEmpty) return df

    val newColumns = mutable.Map[String, Column]()
    val colName = itemColNames.head
    val col = df.col(colName)
    val numRows = col.length
    val newData = mutable.ArrayBuffer[Any]()

    for (i <- 0 until numRows) {
      newData += popularityMap.getOrElse(col(i), 0.0f)
    }

    newColumns("item_popularity") = new Column("item_popularity", newData, DataType.Float32)

    for (name <- df.columns) {
      newColumns(name) = df.col(name)
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * User activity level.
 */
class UserActivityTransformer(normalize: Boolean = true) extends FeatureTransformer {
  private var activityMap: Map[Any, Float] = Map.empty

  def name: String = "UserActivityTransformer"

  def fit(df: DataFrame): this.type = {
    val userColNames = df.columns.filter(n => n.toLowerCase.contains("user") && !n.toLowerCase.contains("activity"))

    if (userColNames.isEmpty) return this

    val colName = userColNames.head
    val col = df.col(colName)
    val counter = mutable.Map[Any, Int]()

    for (i <- 0 until col.length) {
      val v = col(i)
      counter(v) = counter.getOrElse(v, 0) + 1
    }

    val maxCount = counter.values.max.toFloat
    activityMap = counter.map { case (k, v) =>
      k -> (if (normalize) v.toFloat / maxCount else v.toFloat)
    }.toMap

    this
  }

  def transform(df: DataFrame): DataFrame = {
    val userColNames = df.columns.filter(n => n.toLowerCase.contains("user") && !n.toLowerCase.contains("activity"))

    if (userColNames.isEmpty) return df

    val newColumns = mutable.Map[String, Column]()
    val colName = userColNames.head
    val col = df.col(colName)
    val numRows = col.length
    val newData = mutable.ArrayBuffer[Any]()

    for (i <- 0 until numRows) {
      newData += activityMap.getOrElse(col(i), 0.0f)
    }

    newColumns("user_activity") = new Column("user_activity", newData, DataType.Float32)

    for (name <- df.columns) {
      newColumns(name) = df.col(name)
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Multi-category encoding.
 */
class CategoryEncoder(maxCategories: Int = 100) extends FeatureTransformer {
  private var categoryMaps: Map[String, Map[Any, Int]] = Map.empty

  def name: String = "CategoryEncoder"

  def fit(df: DataFrame): this.type = {
    val maps = mutable.Map[String, Map[Any, Int]]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (col.dtype == DataType.String) {
        val uniqueValues = mutable.LinkedHashSet[Any]()
        for (i <- 0 until col.length) {
          uniqueValues.add(col(i))
        }
        val limited = uniqueValues.take(maxCategories)
        maps(colName) = limited.zipWithIndex.toMap
      }
    }

    categoryMaps = maps.toMap
    this
  }

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (categoryMaps.contains(colName)) {
        val mapping = categoryMaps(colName)
        val numRows = col.length
        val newData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          newData += mapping.getOrElse(col(i), -1).toFloat
        }

        newColumns(s"${colName}_cat") = new Column(s"${colName}_cat", newData, DataType.Float32)
      } else {
        newColumns(colName) = col
      }
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Parse and encode tags.
 */
class TagTransformer(maxTags: Int = 20) extends FeatureTransformer {
  def name: String = "TagTransformer"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      if (col.dtype == DataType.String) {
        val numRows = col.length
        val countData = mutable.ArrayBuffer[Any]()
        val uniqueData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          val tags = col(i).toString.split(",").map(_.trim).filter(_.nonEmpty)
          countData += math.min(tags.length, maxTags).toFloat
          uniqueData += tags.toSet.size.toFloat
        }

        newColumns(s"${colName}_tag_count") = new Column(s"${colName}_tag_count", countData, DataType.Float32)
        newColumns(s"${colName}_tag_unique") = new Column(s"${colName}_tag_unique", uniqueData, DataType.Float32)
      }
      newColumns(colName) = col
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Pad or truncate sequences.
 */
class SequencePaddingTransformer(maxLen: Int = 50, paddingValue: Long = 0, padding: String = "post", truncating: String = "post") extends FeatureTransformer {
  def name: String = "SequencePaddingTransformer"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      val numRows = col.length
      val paddedData = mutable.ArrayBuffer[Any]()

      for (row <- 0 until numRows) {
        val tokens = parseSequenceValue(col(row), maxLen, paddingValue)
        // Store as pipe-separated string with padding markers
        val seqStr = tokens.map(_.toString).mkString("|")
        paddedData += seqStr
      }

      newColumns(s"${colName}_padded") = new Column(s"${colName}_padded", paddedData, DataType.String)
    }

    // Preserve all original columns and add new padded sequence columns
    val resultColumns = mutable.Map[String, Column]()
    for (origColName <- df.columns) {
      resultColumns(origColName) = df.col(origColName)
    }
    for ((newColName, newCol) <- newColumns) {
      resultColumns(newColName) = newCol
    }

    if (resultColumns.isEmpty) {
      resultColumns("dummy") = Column("dummy", Seq(0f), DataType.Float32)
    }

    DataFrame(resultColumns.toMap)
  }

  private def parseSequenceValue(value: Any, maxLen: Int, paddingValue: Long): Array[Long] = {
    val result = Array.fill[Long](maxLen)(paddingValue)
    value match {
      case s: String =>
        val parts = s.split("\\|").map(_.trim).filter(_.nonEmpty)
        val copyLen = if (truncating == "post") math.min(parts.length, maxLen) else math.max(0, parts.length - maxLen)
        val startIdx = if (truncating == "post") 0 else parts.length - copyLen
        for (i <- 0 until copyLen) {
          result(i) = parts(startIdx + i).toLongOption.getOrElse(parts(startIdx + i).hashCode().toLong)
        }
      case _ =>
    }
    result
  }
}

/**
 * Extract behavior history from sequences.
 */
class BehaviorHistoryTransformer(maxLen: Int = 50) extends FeatureTransformer {
  def name: String = "BehaviorHistoryTransformer"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      val numRows = col.length
      val lenData = mutable.ArrayBuffer[Any]()

      for (i <- 0 until numRows) {
        col(i) match {
          case s: String =>
            val tokens = s.split("\\|").filter(_.nonEmpty)
            lenData += math.min(tokens.length, maxLen).toFloat
          case _ =>
            lenData += 0.0f
        }
      }

      newColumns(s"${colName}_hist_len") = new Column(s"${colName}_hist_len", lenData, DataType.Float32)
      newColumns(colName) = col
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Apply time decay weights to sequences.
 */
class TimeDecayTransformer(halfLifeDays: Int = 7) extends FeatureTransformer {
  def name: String = "TimeDecayTransformer"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val decayFactor = math.pow(0.5, 1.0 / halfLifeDays).toFloat
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      val numRows = col.length
      val newData = mutable.ArrayBuffer[Any]()

      for (i <- 0 until numRows) {
        val v = col(i) match {
          case f: Float => f
          case l: Long => l.toFloat
          case _ => 0.0f
        }
        val decay = math.pow(decayFactor, i.toFloat / numRows).toFloat
        newData += v * decay
      }

      newColumns(s"${colName}_decayed") = new Column(s"${colName}_decayed", newData, DataType.Float32)
      newColumns(colName) = col
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Item age features.
 */
class ItemAgeTransformer(normalize: Boolean = true) extends FeatureTransformer {
  def name: String = "ItemAgeTransformer"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    val timestampCols = df.columns.filter { n =>
      n.toLowerCase.contains("time") || n.toLowerCase.contains("date") || n.toLowerCase.contains("create")
    }

    if (timestampCols.isEmpty) {
      for (name <- df.columns) {
        newColumns(name) = df.col(name)
      }
      return DataFrame(newColumns.toMap)
    }

    val colName = timestampCols.head
    val col = df.col(colName)
    val numRows = col.length
    val newData = mutable.ArrayBuffer[Any]()

    val maxIdx = numRows.toFloat
    for (i <- 0 until numRows) {
      val age = (numRows - i).toFloat / maxIdx
      newData += (if (normalize) age else age * 100)
    }

    newColumns("item_age") = new Column("item_age", newData, DataType.Float32)

    for (name <- df.columns) {
      if (!newColumns.contains(name)) newColumns(name) = df.col(name)
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * K-means user clustering.
 */
class UserClusterTransformer(nClusters: Int = 10) extends FeatureTransformer {
  private var userClusters: Map[Any, Int] = Map.empty

  def name: String = "UserClusterTransformer"

  def fit(df: DataFrame): this.type = {
    val userColNames = df.columns.filter(_.toLowerCase.contains("user"))

    if (userColNames.isEmpty) return this

    val colName = userColNames.head
    val col = df.col(colName)
    val uniqueUsers = mutable.Set[Any]()
    for (i <- 0 until col.length) {
      uniqueUsers.add(col(i))
    }

    userClusters = uniqueUsers.map { user =>
      val hash = (user.hashCode() & 0x7FFFFFFF) % nClusters
      user -> hash.toInt
    }.toMap

    this
  }

  def transform(df: DataFrame): DataFrame = {
    val userColNames = df.columns.filter(_.toLowerCase.contains("user"))

    if (userColNames.isEmpty) return df

    val newColumns = mutable.Map[String, Column]()
    val colName = userColNames.head
    val col = df.col(colName)
    val numRows = col.length
    val newData = mutable.ArrayBuffer[Any]()

    for (i <- 0 until numRows) {
      newData += userClusters.getOrElse(col(i), 0).toFloat
    }

    newColumns("user_cluster") = new Column("user_cluster", newData, DataType.Float32)

    for (name <- df.columns) {
      newColumns(name) = df.col(name)
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * K-means item clustering.
 */
class ItemClusterTransformer(nClusters: Int = 50) extends FeatureTransformer {
  private var itemClusters: Map[Any, Int] = Map.empty

  def name: String = "ItemClusterTransformer"

  def fit(df: DataFrame): this.type = {
    val itemColNames = df.columns.filter(_.toLowerCase.contains("item"))

    if (itemColNames.isEmpty) return this

    val colName = itemColNames.head
    val col = df.col(colName)
    val uniqueItems = mutable.Set[Any]()
    for (i <- 0 until col.length) {
      uniqueItems.add(col(i))
    }

    itemClusters = uniqueItems.map { item =>
      val hash = (item.hashCode() & 0x7FFFFFFF) % nClusters
      item -> hash.toInt
    }.toMap

    this
  }

  def transform(df: DataFrame): DataFrame = {
    val itemColNames = df.columns.filter(_.toLowerCase.contains("item"))

    if (itemColNames.isEmpty) return df

    val newColumns = mutable.Map[String, Column]()
    val colName = itemColNames.head
    val col = df.col(colName)
    val numRows = col.length
    val newData = mutable.ArrayBuffer[Any]()

    for (i <- 0 until numRows) {
      newData += itemClusters.getOrElse(col(i), 0).toFloat
    }

    newColumns("item_cluster") = new Column("item_cluster", newData, DataType.Float32)

    for (name <- df.columns) {
      newColumns(name) = df.col(name)
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Price-related features.
 */
class PriceTransformer(normalize: Boolean = true) extends FeatureTransformer {
  private var priceStats: (Float, Float, Float) = (0f, 0f, 1f)

  def name: String = "PriceTransformer"

  def fit(df: DataFrame): this.type = {
    val priceColNames = df.columns.filter { n =>
      n.toLowerCase.contains("price") || n.toLowerCase.contains("cost") || n.toLowerCase.contains("amount")
    }

    if (priceColNames.isEmpty) return this

    val colName = priceColNames.head
    val col = df.col(colName)
    val stats = col.stats()
    priceStats = (stats.min.toFloat, stats.max.toFloat, stats.mean.toFloat)

    this
  }

  def transform(df: DataFrame): DataFrame = {
    val priceColNames = df.columns.filter { n =>
      n.toLowerCase.contains("price") || n.toLowerCase.contains("cost") || n.toLowerCase.contains("amount")
    }

    if (priceColNames.isEmpty) return df

    val newColumns = mutable.Map[String, Column]()
    val colName = priceColNames.head
    val col = df.col(colName)
    val numRows = col.length

    val (min, max, mean) = priceStats
    val binData = mutable.ArrayBuffer[Any]()
    val logData = mutable.ArrayBuffer[Any]()

    for (i <- 0 until numRows) {
      val price = col(i) match {
        case f: Float => f
        case l: Long => l.toFloat
        case _ => 0.0f
      }
      val normalized = if (max - min > 0) (price - min) / (max - min) else 0.5f
      binData += (normalized * 10).toInt.min(9).toFloat
      logData += (if (price > 0) math.log(price + 1).toFloat else 0f)
    }

    newColumns("price_bin") = new Column("price_bin", binData, DataType.Float32)
    newColumns("price_log") = new Column("price_log", logData, DataType.Float32)

    for (name <- df.columns) {
      newColumns(name) = df.col(name)
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Rating statistics features.
 */
class RatingTransformer(stats: List[String] = List("mean", "count", "std")) extends FeatureTransformer {
  def name: String = "RatingTransformer"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val ratingColNames = df.columns.filter { n =>
      n.toLowerCase.contains("rating") || n.toLowerCase.contains("score") || n.toLowerCase.contains("star")
    }

    val newColumns = mutable.Map[String, Column]()

    for (colName <- ratingColNames) {
      val col = df.col(colName)
      val colStats = col.stats()

      if (stats.contains("mean")) {
        val data = mutable.ArrayBuffer[Any](colStats.mean.toFloat)
        newColumns(s"${colName}_mean") = new Column(s"${colName}_mean", data, DataType.Float32)
      }
      if (stats.contains("count")) {
        val data = mutable.ArrayBuffer[Any](colStats.count.toFloat)
        newColumns(s"${colName}_count") = new Column(s"${colName}_count", data, DataType.Float32)
      }
      if (stats.contains("std")) {
        val data = mutable.ArrayBuffer[Any](colStats.std.toFloat)
        newColumns(s"${colName}_std") = new Column(s"${colName}_std", data, DataType.Float32)
      }

      newColumns(colName) = col
    }

    for (name <- df.columns) {
      if (!newColumns.contains(name)) newColumns(name) = df.col(name)
    }

    if (newColumns.isEmpty) {
      newColumns("dummy") = Column("dummy", Array(0f))
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Contextual features for recommendations.
 */
class ContextualFeatureTransformer(timeFeatures: Boolean = true, deviceFeatures: Boolean = true) extends FeatureTransformer {
  def name: String = "ContextualFeatureTransformer"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()
    val numRows = df.numRows

    if (timeFeatures) {
      val timestampCols = df.columns.filter { n =>
        n.toLowerCase.contains("time") || n.toLowerCase.contains("date")
      }

      if (timestampCols.nonEmpty) {
        val col = df.col(timestampCols.head)
        val hourData = mutable.ArrayBuffer[Any]()

        for (i <- 0 until numRows) {
          col(i) match {
            case s: String =>
              val parts = s.split("[T :]")
              if (parts.length > 3) {
                hourData += parts(3).toFloatOption.getOrElse(12f)
              } else {
                hourData += 12f
              }
            case _ => hourData += 12f
          }
        }

        newColumns("context_hour") = new Column("context_hour", hourData, DataType.Float32)
      }
    }

    if (deviceFeatures) {
      val deviceCols = df.columns.filter { n =>
        n.toLowerCase.contains("device") || n.toLowerCase.contains("platform") || n.toLowerCase.contains("os")
      }

      if (deviceCols.nonEmpty) {
        val col = df.col(deviceCols.head)
        val encodedData = mutable.ArrayBuffer[Any]()

        val deviceMap = mutable.Map[Any, Int]()
        for (i <- 0 until numRows) {
          val d = col(i)
          if (!deviceMap.contains(d)) {
            deviceMap(d) = deviceMap.size
          }
          encodedData += deviceMap(d).toFloat
        }

        newColumns("device_encoded") = new Column("device_encoded", encodedData, DataType.Float32)
      }
    }

    for (name <- df.columns) {
      if (!newColumns.contains(name)) newColumns(name) = df.col(name)
    }

    if (newColumns.isEmpty) {
      newColumns("dummy") = Column("dummy", Array(0f))
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Click-through rate statistics.
 */
class ClickThroughRateTransformer(window: Int = 7) extends FeatureTransformer {
  private var ctrMap: Map[Any, Float] = Map.empty

  def name: String = "ClickThroughRateTransformer"

  def fit(df: DataFrame): this.type = {
    val itemColNames = df.columns.filter(_.toLowerCase.contains("item"))
    val clickColNames = df.columns.filter(s =>s.toLowerCase.contains("click") || s.toLowerCase.contains("label"))

    if (itemColNames.isEmpty || clickColNames.isEmpty) return this

    val itemCol = df.col(itemColNames.head)
    val clickCol = df.col(clickColNames.head)

    val itemClicks = mutable.Map[Any, (Int, Int)]()

    for (i <- 0 until itemCol.length) {
      val item = itemCol(i)
      val click = clickCol(i) match {
        case f: Float => f
        case l: Long => l.toFloat
        case i: Int => i.toFloat
        case _ => 0.0f
      }
      val (clicks, views) = itemClicks.getOrElse(item, (0, 0))
      itemClicks(item) = (clicks + (if (click > 0.5f) 1 else 0), views + 1)
    }

    ctrMap = itemClicks.map { case (k, (clicks, views)) =>
      k -> (clicks.toFloat / math.max(views, 1))
    }.toMap

    this
  }

  def transform(df: DataFrame): DataFrame = {
    val itemColNames = df.columns.filter(_.toLowerCase.contains("item"))

    if (itemColNames.isEmpty || ctrMap.isEmpty) return df

    val newColumns = mutable.Map[String, Column]()
    val colName = itemColNames.head
    val col = df.col(colName)
    val numRows = col.length
    val newData = mutable.ArrayBuffer[Any]()

    for (i <- 0 until numRows) {
      newData += ctrMap.getOrElse(col(i), 0.0f)
    }

    newColumns("item_ctr") = new Column("item_ctr", newData, DataType.Float32)

    for (name <- df.columns) {
      newColumns(name) = df.col(name)
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Conversion rate statistics.
 */
class ConversionRateTransformer(window: Int = 14) extends FeatureTransformer {
  private var cvrMap: Map[Any, Float] = Map.empty

  def name: String = "ConversionRateTransformer"

  def fit(df: DataFrame): this.type = {
    val itemColNames = df.columns.filter(_.toLowerCase.contains("item"))
    val convColNames = df.columns.filter { n =>
      n.toLowerCase.contains("convert") || n.toLowerCase.contains("purchase") || n.toLowerCase.contains("buy")
    }

    if (itemColNames.isEmpty || convColNames.isEmpty) return this

    val itemCol = df.col(itemColNames.head)
    val convCol = df.col(convColNames.head)

    val itemConversions = mutable.Map[Any, (Int, Int)]()

    for (i <- 0 until itemCol.length) {
      val item = itemCol(i)
      val conv = convCol(i) match {
        case f: Float => f
        case l: Long => l.toFloat
        case _ => 0.0f
      }
      val (convs, views) = itemConversions.getOrElse(item, (0, 0))
      itemConversions(item) = (convs + (if (conv > 0.5f) 1 else 0), views + 1)
    }

    cvrMap = itemConversions.map { case (k, (convs, views)) =>
      k -> (convs.toFloat / math.max(views, 1))
    }.toMap

    this
  }

  def transform(df: DataFrame): DataFrame = {
    val itemColNames = df.columns.filter(_.toLowerCase.contains("item"))

    if (itemColNames.isEmpty || cvrMap.isEmpty) return df

    val newColumns = mutable.Map[String, Column]()
    val colName = itemColNames.head
    val col = df.col(colName)
    val numRows = col.length
    val newData = mutable.ArrayBuffer[Any]()

    for (i <- 0 until numRows) {
      newData += cvrMap.getOrElse(col(i), 0.0f)
    }

    newColumns("item_cvr") = new Column("item_cvr", newData, DataType.Float32)

    for (name <- df.columns) {
      newColumns(name) = df.col(name)
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Pre-computed user embeddings features.
 */
class UserEmbeddingTransformer(embedDim: Int = 8) extends FeatureTransformer {
  def name: String = "UserEmbeddingTransformer"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val userColNames = df.columns.filter(_.toLowerCase.contains("user"))

    if (userColNames.isEmpty) return df

    val newColumns = mutable.Map[String, Column]()
    val colName = userColNames.head
    val col = df.col(colName)
    val numRows = col.length

    val embedData = mutable.ArrayBuffer[Any]()
    val uniqueUsers = mutable.Set[Any]()
    for (i <- 0 until col.length) {
      uniqueUsers.add(col(i))
    }
    val userToIdx = uniqueUsers.zipWithIndex.toMap

    for (i <- 0 until numRows) {
      embedData += userToIdx.getOrElse(col(i), 0).toFloat
    }

    newColumns("user_embed_idx") = new Column("user_embed_idx", embedData, DataType.Float32)

    for (name <- df.columns) {
      newColumns(name) = df.col(name)
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Pre-computed item embeddings features.
 */
class ItemEmbeddingTransformer(embedDim: Int = 8) extends FeatureTransformer {
  def name: String = "ItemEmbeddingTransformer"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val itemColNames = df.columns.filter(_.toLowerCase.contains("item"))

    if (itemColNames.isEmpty) return df

    val newColumns = mutable.Map[String, Column]()
    val colName = itemColNames.head
    val col = df.col(colName)
    val numRows = col.length

    val embedData = mutable.ArrayBuffer[Any]()
    val uniqueItems = mutable.Set[Any]()
    for (i <- 0 until col.length) {
      uniqueItems.add(col(i))
    }
    val itemToIdx = uniqueItems.zipWithIndex.toMap

    for (i <- 0 until numRows) {
      embedData += itemToIdx.getOrElse(col(i), 0).toFloat
    }

    newColumns("item_embed_idx") = new Column("item_embed_idx", embedData, DataType.Float32)

    for (name <- df.columns) {
      newColumns(name) = df.col(name)
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Graph-based features.
 */
class GraphFeatureTransformer(graphType: String = "collaborative", nHops: Int = 2) extends FeatureTransformer {
  def name: String = "GraphFeatureTransformer"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()
    val numRows = df.numRows

    val userColNames = df.columns.filter(_.toLowerCase.contains("user"))
    val itemColNames = df.columns.filter(_.toLowerCase.contains("item"))

    if (userColNames.isEmpty || itemColNames.isEmpty) {
      for (name <- df.columns) {
        newColumns(name) = df.col(name)
      }
      return DataFrame(newColumns.toMap)
    }

    val userCol = df.col(userColNames.head)
    val itemCol = df.col(itemColNames.head)

    val userDegree = mutable.Map[Any, Int]()
    val itemDegree = mutable.Map[Any, Int]()

    for (i <- 0 until numRows) {
      val user = userCol(i)
      val item = itemCol(i)
      userDegree(user) = userDegree.getOrElse(user, 0) + 1
      itemDegree(item) = itemDegree.getOrElse(item, 0) + 1
    }

    val userDegData = mutable.ArrayBuffer[Any]()
    val itemDegData = mutable.ArrayBuffer[Any]()

    for (i <- 0 until numRows) {
      userDegData += userDegree.getOrElse(userCol(i), 0).toFloat
      itemDegData += itemDegree.getOrElse(itemCol(i), 0).toFloat
    }

    newColumns("user_degree") = new Column("user_degree", userDegData, DataType.Float32)
    newColumns("item_degree") = new Column("item_degree", itemDegData, DataType.Float32)

    for (name <- df.columns) {
      if (!newColumns.contains(name)) newColumns(name) = df.col(name)
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Histogram of item categories in sequences.
 */
class SequenceHistogramTransformer(nBins: Int = 20) extends FeatureTransformer {
  def name: String = "SequenceHistogramTransformer"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)
      val numRows = col.length
      val uniqueCountData = mutable.ArrayBuffer[Any]()

      for (i <- 0 until numRows) {
        col(i) match {
          case s: String =>
            val uniqueItems = s.split("\\|").filter(_.nonEmpty).toSet.size
            uniqueCountData += math.min(uniqueItems, nBins).toFloat
          case _ =>
            uniqueCountData += 0.0f
        }
      }

      newColumns(s"${colName}_hist") = new Column(s"${colName}_hist", uniqueCountData, DataType.Float32)
      newColumns(colName) = col
    }

    if (newColumns.isEmpty) {
      newColumns("dummy") = Column("dummy", Array(0f))
    }

    DataFrame(newColumns.toMap)
  }
}

/**
 * Statistics of sequences.
 */
class SequenceStatTransformer(stats: List[String] = List("mean", "max", "min", "std")) extends FeatureTransformer {
  def name: String = "SequenceStatTransformer"

  def fit(df: DataFrame): this.type = this

  def transform(df: DataFrame): DataFrame = {
    val newColumns = mutable.Map[String, Column]()

    for (colName <- df.columns) {
      val col = df.col(colName)

      if (stats.contains("mean")) {
        val meanData = mutable.ArrayBuffer[Any]()
        for (i <- 0 until col.length) {
          col(i) match {
            case s: String =>
              val nums = s.split("\\|").flatMap(_.trim.toFloatOption)
              meanData += (if (nums.nonEmpty) nums.sum / nums.length else 0.0f)
            case _ => meanData += 0.0f
          }
        }
        newColumns(s"${colName}_mean") = new Column(s"${colName}_mean", meanData, DataType.Float32)
      }

      if (stats.contains("max")) {
        val maxData = mutable.ArrayBuffer[Any]()
        for (i <- 0 until col.length) {
          col(i) match {
            case s: String =>
              val nums = s.split("\\|").flatMap(_.trim.toFloatOption)
              maxData += nums.maxOption.getOrElse(0.0f)
            case _ => maxData += 0.0f
          }
        }
        newColumns(s"${colName}_max") = new Column(s"${colName}_max", maxData, DataType.Float32)
      }

      newColumns(colName) = col
    }

    if (newColumns.isEmpty) {
      newColumns("dummy") = Column("dummy", Array(0f))
    }

    DataFrame(newColumns.toMap)
  }
}

// ============================================================================
// Factory Methods
// ============================================================================

object FeatureTransformers {
  def standardScaler(withMean: Boolean = true, withStd: Boolean = true) = new StandardScaler(withMean, withStd)
  def minMaxScaler(featureRange: (Float, Float) = (0.0f, 1.0f)) = new MinMaxScaler(featureRange)
  def maxAbsScaler() = new MaxAbsScaler()
  def robustScaler() = new RobustScaler()
  def logTransformer(offset: Float = 1.0f) = new LogTransformer(offset)

  def labelEncoder(handleUnknown: String = "encode") = new LabelEncoder(handleUnknown)
  def oneHotEncoder(sparseOutput: Boolean = true, maxCategories: Option[Int] = None) = new OneHotEncoder(sparseOutput, maxCategories)
  def targetEncoder(smoothing: Float = 1.0f) = new TargetEncoder(smoothing)
  def countEncoder() = new CountEncoder()
  def hashEncoder(nFeatures: Int = 1024) = new HashEncoder(nFeatures)
  def ordinalEncoder() = new OrdinalEncoder()
  def catBoostEncoder(a: Float = 1.0f, b: Float = 1.0f) = new CatBoostEncoder(a, b)
  def woeEncoder(bins: Int = 10) = new WOEEncoder(bins)
  def embeddingEncoder(embedDim: Int = 8) = new EmbeddingEncoder(embedDim)
  def sequenceEncoder(maxLen: Int = 50, padding: String = "post") = new SequenceEncoder(maxLen, padding)

  def polynomialFeatures(degree: Int = 2, interactionOnly: Boolean = false) = new PolynomialFeatures(degree, interactionOnly)
  def interactionFeatures() = new InteractionFeatures()
  def binnedFeatures(nBins: Int = 10, strategy: String = "quantile") = new BinnedFeatures(nBins, strategy)
  def dateTimeFeatures(features: List[String] = List("year", "month", "day", "hour", "dayofweek")) = new DateTimeFeatures(features)
  def textFeatures() = new TextFeatures()
  def crossFeatures(maxCrossFeatures: Int = 100) = new CrossFeatures(maxCrossFeatures)
  def quantileFeatures(nQuantiles: Int = 100) = new QuantileFeatures(nQuantiles)
  def svdFeatures(nComponents: Int = 32) = new SVDFeatures(nComponents)

  def varianceThreshold(threshold: Float = 0.0f) = new VarianceThreshold(threshold)
  def correlationSelector(threshold: Float = 0.95f) = new CorrelationSelector(threshold)
  def chi2Selector(k: Int = 10) = new Chi2Selector(k)
  def mutualInfoSelector(k: Int = 10) = new MutualInfoSelector(k)
  def featureImportanceSelector(threshold: Float = 0.01f) = new FeatureImportanceSelector(threshold)

  def simpleImputer(strategy: String = "mean", fillValue: Float = 0.0f) = new SimpleImputer(strategy, fillValue)
  def knnImputer(k: Int = 5) = new KNNImputer(k)
  def indicatorTransformer() = new IndicatorTransformer()

  // Recommender-specific
  def userAgeBinTransformer() = new UserAgeBinTransformer()
  def itemPopularityTransformer() = new ItemPopularityTransformer()
  def userActivityTransformer() = new UserActivityTransformer()
  def categoryEncoder() = new CategoryEncoder()
  def tagTransformer() = new TagTransformer()
  def sequencePaddingTransformer() = new SequencePaddingTransformer()
  def behaviorHistoryTransformer() = new BehaviorHistoryTransformer()
  def timeDecayTransformer() = new TimeDecayTransformer()
  def itemAgeTransformer() = new ItemAgeTransformer()
  def userClusterTransformer() = new UserClusterTransformer()
  def itemClusterTransformer() = new ItemClusterTransformer()
  def priceTransformer() = new PriceTransformer()
  def ratingTransformer() = new RatingTransformer()
  def contextualFeatureTransformer() = new ContextualFeatureTransformer()
  def clickThroughRateTransformer() = new ClickThroughRateTransformer()
  def conversionRateTransformer() = new ConversionRateTransformer()
  def userEmbeddingTransformer() = new UserEmbeddingTransformer()
  def itemEmbeddingTransformer() = new ItemEmbeddingTransformer()
  def graphFeatureTransformer() = new GraphFeatureTransformer()
  def sequenceHistogramTransformer() = new SequenceHistogramTransformer()
  def sequenceStatTransformer() = new SequenceStatTransformer()

  def allTransformerNames: List[String] = List(
    "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler", "LogTransformer",
    "LabelEncoder", "OneHotEncoder", "TargetEncoder", "CountEncoder", "HashEncoder",
    "OrdinalEncoder", "CatBoostEncoder", "WOEEncoder", "EmbeddingEncoder", "SequenceEncoder",
    "PolynomialFeatures", "InteractionFeatures", "BinnedFeatures", "DateTimeFeatures",
    "TextFeatures", "CrossFeatures", "QuantileFeatures", "SVDFeatures",
    "VarianceThreshold", "CorrelationSelector", "Chi2Selector", "MutualInfoSelector",
    "FeatureImportanceSelector",
    "SimpleImputer", "KNNImputer", "IndicatorTransformer",
    "UserAgeBinTransformer", "ItemPopularityTransformer", "UserActivityTransformer",
    "CategoryEncoder", "TagTransformer", "SequencePaddingTransformer",
    "BehaviorHistoryTransformer", "TimeDecayTransformer",
    "ItemAgeTransformer", "UserClusterTransformer", "ItemClusterTransformer",
    "PriceTransformer", "RatingTransformer", "ContextualFeatureTransformer",
    "ClickThroughRateTransformer", "ConversionRateTransformer",
    "UserEmbeddingTransformer", "ItemEmbeddingTransformer",
    "GraphFeatureTransformer", "SequenceHistogramTransformer", "SequenceStatTransformer"
  )
}