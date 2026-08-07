package torchrec.dataframe

import torchrec.data._
import torchrec.Implicits.tensor
import org.bytedeco.pytorch._
import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.nn.modules._
import org.bytedeco.pytorch.nn.modules.container._
import org.bytedeco.pytorch.nn.options._
import org.bytedeco.pytorch.optim._
import org.bytedeco.pytorch.data.datasets._
import org.bytedeco.pytorch.data.options._
import org.bytedeco.pytorch.data.sampler._
import org.bytedeco.pytorch.distributed._
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.mutable

/**
 * Dataset that wraps a DataFrame.
 * Provides indexed access to batches for use with DataLoader.
 */
class DataFrameDataset(
  df: DataFrame,
  featureSpec: FeatureSpec,
  labelSpec: Option[LabelSpec] = None
) extends Dataset {

  override def size: Long = df.numRows.toLong

  override def get(index: Long): Batch = {
    val safeIndex = index.min(size - 1).max(0).toInt

    // Build sparse features
    val sparseMap = mutable.Map[String, Tensor]()
    for (spec <- featureSpec.sparseFeatures) {
      val col = df.col(spec.columnName)
      val values = Array.ofDim[Float](1)
      values(0) = col(safeIndex) match {
        case l: Long => l.toFloat
        case i: Int => i.toFloat
        case f: Float => f
        case d: Double => d.toFloat
        case s: String => s.hashCode().toFloat
        case null => 0.0f
        case _ => 0.0f
      }
      sparseMap(spec.columnName) = tensor(values, Array(1L)).toType(ScalarType.Long)
    }

    // Build dense features
    val denseMap = mutable.Map[String, Tensor]()
    for (spec <- featureSpec.denseFeatures) {
      val col = df.col(spec.columnName)
      val values = Array.ofDim[Float](1)
      values(0) = col(safeIndex) match {
        case f: Float => f
        case d: Double => d.toFloat
        case l: Long => l.toFloat
        case i: Int => i.toFloat
        case s: String => s.toFloatOption.getOrElse(0.0f)
        case null => 0.0f
        case _ => 0.0f
      }
      denseMap(spec.columnName) = tensor(values, Array(1L))
    }

    // Build sequence features
    val sequenceMap = mutable.Map[String, Tensor]()
    for (spec <- featureSpec.sequenceFeatures) {
      val col = df.col(spec.columnName)
      val values = Array.ofDim[Float](spec.maxLen)
      col(safeIndex) match {
        case s: String =>
          val tokens = s.split("\\|").map(_.trim).filter(_.nonEmpty)
          for (j <- 0 until spec.maxLen) {
            values(j) = if (j < tokens.length) {
              tokens(j).toLongOption.getOrElse(tokens(j).hashCode().toLong).toFloat
            } else spec.paddingIdx.toFloat
          }
        case _ =>
          for (j <- 0 until spec.maxLen) {
            values(j) = spec.paddingIdx.toFloat
          }
      }
      sequenceMap(spec.columnName) = tensor(values, Array(1L, spec.maxLen.toLong)).toType(ScalarType.Long)
    }

    // Build labels
    val labelTensor = labelSpec.map { spec =>
      val col = df.col(spec.columnName)
      val value = col(safeIndex)
      val labelValue = if (value == null) {
        0.0f
      } else {
        val isPositive = value match {
          case pv if pv == spec.positiveValue => true
          case iv: Int => iv == spec.positiveValue.asInstanceOf[Int]
          case lv: Long => lv == spec.positiveValue.asInstanceOf[Long]
          case fv: Float => fv == spec.positiveValue.asInstanceOf[Float]
          case sv: String => sv == spec.positiveValue.toString
          case _ => false
        }
        if (isPositive) 1.0f else 0.0f
      }
      tensor(Array(labelValue), Array(1L))
    }

    Batch(
      sparseFeatures = sparseMap.toMap,
      denseFeatures = denseMap.toMap,
      sequenceFeatures = sequenceMap.toMap,
      labels = labelTensor
    )
  }
}

/**
 * DataLoader that wraps a DataFrame.
 */
class DataFrameDataLoader(
  df: DataFrame,
  featureSpec: FeatureSpec,
  labelSpec: Option[LabelSpec] = None,
  batchSize: Int = 256,
  shuffle: Boolean = true,
  dropLast: Boolean = false
) extends DataLoader(
  DataFrameDataset(df, featureSpec, labelSpec),
  batchSize = batchSize,
  shuffle = shuffle,
  dropLast = dropLast
)

/**
 * Factory methods for creating DataFrameDataLoaders.
 */
object DataFrameDataLoader {

  /**
   * Create a DataFrameDataLoader from a CSV file.
   */
  def fromCSV(
    path: String,
    featureSpec: FeatureSpec,
    labelSpec: Option[LabelSpec] = None,
    batchSize: Int = 256,
    shuffle: Boolean = true,
    delimiter: Char = ','
  ): DataFrameDataLoader = {
    val df = DataFrame.readCSV(path, delimiter)
    new DataFrameDataLoader(df, featureSpec, labelSpec, batchSize, shuffle)
  }

  /**
   * Create a DataFrameDataLoader from a DataFrame.
   */
  def fromDataFrame(
    df: DataFrame,
    featureSpec: FeatureSpec,
    labelSpec: Option[LabelSpec] = None,
    batchSize: Int = 256,
    shuffle: Boolean = true,
    dropLast: Boolean = false
  ): DataFrameDataLoader = {
    new DataFrameDataLoader(df, featureSpec, labelSpec, batchSize, shuffle, dropLast)
  }

  /**
   * Create a DataFrameDataLoader from a sequence of rows.
   */
  def fromRows(
    rows: Seq[Map[String, Any]],
    featureSpec: FeatureSpec,
    labelSpec: Option[LabelSpec] = None,
    batchSize: Int = 256,
    shuffle: Boolean = true
  ): DataFrameDataLoader = {
    val df = DataFrame.fromRows(rows)
    new DataFrameDataLoader(df, featureSpec, labelSpec, batchSize, shuffle)
  }

  /**
   * Create a DataFrameDataLoader from a DataGenerator.
   */
  def fromGenerator(
    generator: torchrec.data.DataGenerator.type,
    genType: String,
    params: Map[String, Any],
    featureSpec: FeatureSpec,
    labelSpec: Option[LabelSpec] = None,
    batchSize: Int = 256
  ): DataFrameDataLoader = {
    genType match {
      case "ranking" =>
        val (train, _, _) = generator.generateRankingData(
          numSamples = params.getOrElse("numSamples", 10000).asInstanceOf[Int],
          numSparseFeatures = params.getOrElse("numSparseFeatures", 10).asInstanceOf[Int],
          numDenseFeatures = params.getOrElse("numDenseFeatures", 5).asInstanceOf[Int],
          vocabSize = params.getOrElse("vocabSize", 100).asInstanceOf[Int],
          seed = params.getOrElse("seed", 42).asInstanceOf[Int]
        )
        val df = DataFrameBridge.fromDataset(train)
        new DataFrameDataLoader(df, featureSpec, labelSpec, batchSize)

      case "matching" =>
        val (train, _, _) = generator.generateMatchingData(
          numUsers = params.getOrElse("numUsers", 1000).asInstanceOf[Int],
          numItems = params.getOrElse("numItems", 1000).asInstanceOf[Int],
          avgSequenceLength = params.getOrElse("avgSequenceLength", 10).asInstanceOf[Int],
          numUserFeatures = params.getOrElse("numUserFeatures", 3).asInstanceOf[Int],
          numItemFeatures = params.getOrElse("numItemFeatures", 2).asInstanceOf[Int],
          vocabSize = params.getOrElse("vocabSize", 100).asInstanceOf[Int],
          seed = params.getOrElse("seed", 42).asInstanceOf[Int]
        )
        val df = DataFrameBridge.fromDataset(train)
        new DataFrameDataLoader(df, featureSpec, labelSpec, batchSize)

      case "multitask" =>
        val taskNames = params.getOrElse("taskNames", List("cvr", "ctr")).asInstanceOf[List[String]]
        val (train, _, _) = generator.generateMultiTaskData(
          numSamples = params.getOrElse("numSamples", 10000).asInstanceOf[Int],
          numFeatures = params.getOrElse("numFeatures", 10).asInstanceOf[Int],
          taskNames = taskNames,
          vocabSize = params.getOrElse("vocabSize", 100).asInstanceOf[Int],
          seed = params.getOrElse("seed", 42).asInstanceOf[Int]
        )
        val df = DataFrameBridge.fromDataset(train)
        new DataFrameDataLoader(df, featureSpec, labelSpec, batchSize)

      case _ =>
        throw new IllegalArgumentException(s"Unknown generator type: $genType")
    }
  }
}