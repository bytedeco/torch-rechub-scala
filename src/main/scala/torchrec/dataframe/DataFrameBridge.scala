package torchrec.dataframe

import torchrec.data.{Batch, DataLoader, Dataset, MatchingDataset, SequenceDataset, TensorDataset}
import torchrec.basic.features.*
import org.bytedeco.pytorch.*
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
import torchrec.Implicits.*

/**
 * Bridge between DataFrame and torchrec Dataset/Batch.
 * Provides bidirectional conversion for the full recommendation pipeline.
 */
object DataFrameBridge {

  // ========================================================================
  // DataFrame → Dataset
  // ========================================================================

  /**
   * Convert DataFrame to Dataset based on FeatureSpec.
   */
  def toDataset(
    df: DataFrame,
    featureSpec: FeatureSpec,
    labelSpec: Option[LabelSpec] = None
  ): Dataset = {
    require(df != null, "DataFrame cannot be null")
    require(featureSpec.allColumnNames.forall(df.columns.contains),
      s"Missing columns: ${featureSpec.allColumnNames.filterNot(df.columns.contains)}")

    val sparseMap = mutable.Map[String, Tensor]()
    for (spec <- featureSpec.sparseFeatures) {
      val col = df.col(spec.columnName)
      val tensor = columnToLongTensor(col, spec.paddingIdx)
      sparseMap(spec.columnName) = tensor
    }

    val denseMap = mutable.Map[String, Tensor]()
    for (spec <- featureSpec.denseFeatures) {
      val col = df.col(spec.columnName)
      val tensor = columnToFloatTensor(col)
      denseMap(spec.columnName) = tensor
    }

    val sequenceMap = mutable.Map[String, Tensor]()
    for (spec <- featureSpec.sequenceFeatures) {
      val col = df.col(spec.columnName)
      val tensor = columnToSequenceTensor(col, spec.maxLen, spec.paddingIdx)
      sequenceMap(spec.columnName) = tensor
    }

    val labelTensor = labelSpec.map { spec =>
      val col = df.col(spec.columnName)
      columnToLabelTensor(col, spec)
    }

    new TensorDataset(
      sparseFeatures = sparseMap.toMap,
      denseFeatures = denseMap.toMap,
      labels = labelTensor
    )
  }

  /**
   * Convert DataFrame to SequenceDataset.
   */
  def toSequenceDataset(
    df: DataFrame,
    featureSpec: FeatureSpec,
    labelSpec: Option[LabelSpec] = None,
    tokensSpec: Option[SequenceSpec] = None,
    positionsSpec: Option[SequenceSpec] = None
  ): SequenceDataset = {
    val featuresMap = mutable.Map[String, Tensor]()
    for (spec <- featureSpec.sparseFeatures) {
      val col = df.col(spec.columnName)
      featuresMap(spec.columnName) = columnToLongTensor(col, None)
    }

    val sequenceMap = mutable.Map[String, Tensor]()
    for (spec <- featureSpec.sequenceFeatures) {
      val col = df.col(spec.columnName)
      sequenceMap(spec.columnName) = columnToSequenceTensor(col, spec.maxLen, spec.paddingIdx)
    }

    val labelTensor = labelSpec.map { spec =>
      columnToLabelTensor(df.col(spec.columnName), spec)
    }

    val tokensTensor = tokensSpec.map { spec =>
      columnToSequenceTensor(df.col(spec.columnName), spec.maxLen, spec.paddingIdx)
    }

    val positionsTensor = positionsSpec.map { spec =>
      val seqLen = spec.maxLen
      val numRows = df.numRows
      val positions = (0 until numRows).flatMap { row =>
        (0 until seqLen).map(_.toFloat)
      }.toArray
      tensor(positions, Array(numRows.toLong, seqLen.toLong)).toType(ScalarType.Long)
    }

    new SequenceDataset(
      features = featuresMap.toMap,
      sequenceFeatures = sequenceMap.toMap,
      labels = labelTensor,
      tokens = tokensTensor,
      positions = positionsTensor
    )
  }

  /**
   * Convert DataFrame to MatchingDataset.
   */
  def toMatchingDataset(
    df: DataFrame,
    userFeatures: List[SparseSpec],
    itemFeatures: List[SparseSpec],
    labelSpec: Option[LabelSpec] = None
  ): MatchingDataset = {
    val userMap = mutable.Map[String, Tensor]()
    for (spec <- userFeatures) {
      userMap(spec.columnName) = columnToLongTensor(df.col(spec.columnName), None)
    }

    val itemMap = mutable.Map[String, Tensor]()
    for (spec <- itemFeatures) {
      itemMap(spec.columnName) = columnToLongTensor(df.col(spec.columnName), None)
    }

    val labelTensor = labelSpec.map { spec =>
      columnToLabelTensor(df.col(spec.columnName), spec)
    }

    new MatchingDataset(
      userFeatures = userMap.toMap,
      itemFeatures = itemMap.toMap,
      labels = labelTensor
    )
  }

  // ========================================================================
  // Dataset → DataFrame
  // ========================================================================

  /**
   * Convert Dataset to DataFrame.
   */
  def fromDataset(dataset: Dataset): DataFrame = {
    dataset match {
      case td: TensorDataset => fromTensorDataset(td)
      case sd: SequenceDataset => fromSequenceDataset(sd)
      case md: MatchingDataset => fromMatchingDataset(md)
      case _ => throw new UnsupportedOperationException(s"Unsupported dataset type: ${dataset.getClass.getName}")
    }
  }

  /**
   * Convert TensorDataset to DataFrame.
   */
  def fromTensorDataset(td: TensorDataset): DataFrame = {
    val columns = mutable.Map[String, Column]()

    for ((name, tensor) <- td.sparseFeatures) {
      columns(name) = longTensorToColumn(name, tensor)
    }

    for ((name, tensor) <- td.denseFeatures) {
      columns(name) = floatTensorToColumn(name, tensor)
    }

    td.labels.foreach { tensor =>
      columns("label") = floatTensorToColumn("label", tensor)
    }

    DataFrame(columns.toMap)
  }

  /**
   * Convert SequenceDataset to DataFrame.
   */
  def fromSequenceDataset(sd: SequenceDataset): DataFrame = {
    val columns = mutable.Map[String, Column]()

    for ((name, tensor) <- sd.features) {
      columns(name) = longTensorToColumn(name, tensor)
    }

    for ((name, tensor) <- sd.sequenceFeatures) {
      columns(name) = longTensorToColumn(name, tensor)
    }

    sd.labels.foreach { tensor =>
      columns("label") = floatTensorToColumn("label", tensor)
    }

    DataFrame(columns.toMap)
  }

  /**
   * Convert MatchingDataset to DataFrame.
   */
  def fromMatchingDataset(md: MatchingDataset): DataFrame = {
    val columns = mutable.Map[String, Column]()

    for ((name, tensor) <- md.userFeatures) {
      columns(s"user_$name") = longTensorToColumn(s"user_$name", tensor)
    }

    for ((name, tensor) <- md.itemFeatures) {
      columns(s"item_$name") = longTensorToColumn(s"item_$name", tensor)
    }

    md.labels.foreach { tensor =>
      columns("label") = floatTensorToColumn("label", tensor)
    }

    DataFrame(columns.toMap)
  }

  // ========================================================================
  // Batch → DataFrame
  // ========================================================================

  /**
   * Convert Batch to DataFrame.
   */
  def fromBatch(batch: Batch): DataFrame = {
    val columns = mutable.Map[String, Column]()

    for ((name, tensor) <- batch.sparseFeatures) {
      columns(name) = longTensorToColumn(name, tensor)
    }

    for ((name, tensor) <- batch.denseFeatures) {
      columns(name) = floatTensorToColumn(name, tensor)
    }

    for ((name, tensor) <- batch.sequenceFeatures) {
      columns(name) = longTensorToColumn(name, tensor)
    }

    batch.labels.foreach { tensor =>
      columns("label") = floatTensorToColumn("label", tensor)
    }

    for ((name, tensor) <- batch.itemFeatures) {
      columns(s"item_$name") = longTensorToColumn(s"item_$name", tensor)
    }

    batch.taskLabels.foreach { taskLabels =>
      for ((taskName, tensor) <- taskLabels) {
        columns(s"task_$taskName") = floatTensorToColumn(s"task_$taskName", tensor)
      }
    }

    DataFrame(columns.toMap)
  }

  // ========================================================================
  // DataFrame → DataLoader
  // ========================================================================

  /**
   * Convert DataFrame to DataLoader.
   */
  def toDataLoader(
    df: DataFrame,
    featureSpec: FeatureSpec,
    labelSpec: Option[LabelSpec] = None,
    batchSize: Int = 256,
    shuffle: Boolean = true,
    dropLast: Boolean = false
  ): DataLoader = {
    val dataset = toDataset(df, featureSpec, labelSpec)
    new DataLoader(dataset, batchSize, shuffle, dropLast = dropLast)
  }

  /**
   * Convert DataFrame to DataLoader with sequence features.
   */
  def toSequenceDataLoader(
    df: DataFrame,
    featureSpec: FeatureSpec,
    labelSpec: Option[LabelSpec] = None,
    tokensSpec: Option[SequenceSpec] = None,
    positionsSpec: Option[SequenceSpec] = None,
    batchSize: Int = 256,
    shuffle: Boolean = true
  ): DataLoader = {
    val dataset = toSequenceDataset(df, featureSpec, labelSpec, tokensSpec, positionsSpec)
    new DataLoader(dataset, batchSize, shuffle)
  }

  // ========================================================================
  // DataFrame → Feature List
  // ========================================================================

  /**
   * Extract Feature list from FeatureSpec for model building.
   */
  def toFeatureList(featureSpec: FeatureSpec): List[Feature] = {
    val features = mutable.ListBuffer[Feature]()

    for (spec <- featureSpec.sparseFeatures) {
      features += SparseFeature(
        name = spec.columnName,
        vocabSize = spec.vocabSize,
        embedDim = spec.embedDim,
        sharedWith = spec.sharedWith,
        paddingIdx = spec.paddingIdx
      )
    }

    for (spec <- featureSpec.denseFeatures) {
      features += DenseFeature(
        name = spec.columnName,
        embedDim = spec.embedDim
      )
    }

    for (spec <- featureSpec.sequenceFeatures) {
      features += SequenceFeature(
        name = spec.columnName,
        vocabSize = spec.vocabSize,
        embedDim = spec.embedDim,
        pooling = spec.pooling,
        sharedWith = spec.sharedWith,
        maxLen = spec.maxLen,
        paddingIdx = spec.paddingIdx
      )
    }

    features.toList
  }

  // ========================================================================
  // Helper Methods
  // ========================================================================

  private def columnToLongTensor(col: Column, paddingIdx: Option[Long]): Tensor = {
    val numRows = col.length
    val values = Array.ofDim[Float](numRows)

    for (i <- 0 until numRows) {
      val v = col(i)
      values(i) = v match {
        case l: Long => l.toFloat
        case i: Int => i.toFloat
        case f: Float => f
        case d: Double => d.toFloat
        case s: String => s.hashCode().toFloat
        case null => paddingIdx.getOrElse(0L).toFloat
        case _ => 0.0f
      }
    }

    tensor(values, Array(numRows.toLong)).toType(ScalarType.Long)
  }

  private def columnToFloatTensor(col: Column): Tensor = {
    val numRows = col.length
    val values = Array.ofDim[Float](numRows)

    for (i <- 0 until numRows) {
      val v = col(i)
      values(i) = v match {
        case f: Float => f
        case d: Double => d.toFloat
        case l: Long => l.toFloat
        case i: Int => i.toFloat
        case s: String => s.toFloatOption.getOrElse(0.0f)
        case null => 0.0f
        case _ => 0.0f
      }
    }

    tensor(values, Array(numRows.toLong))
  }

  private def columnToSequenceTensor(col: Column, maxLen: Int, paddingIdx: Long): Tensor = {
    val numRows = col.length
    val flatValues = Array.ofDim[Float](numRows * maxLen)

    for (row <- 0 until numRows) {
      val tokens = parseSequenceValue(col(row), maxLen, paddingIdx)
      for (j <- 0 until maxLen) {
        flatValues(row * maxLen + j) = tokens(j).toFloat
      }
    }

    tensor(flatValues, Array(numRows.toLong, maxLen.toLong)).toType(ScalarType.Long)
  }

  private def parseSequenceValue(value: Any, maxLen: Int, paddingIdx: Long): Array[Float] = {
    val result = Array.fill[Float](maxLen)(paddingIdx.toFloat)
    value match {
      case s: String =>
        val tokens = s.split("\\|").map(_.trim).filter(_.nonEmpty)
        val copyLen = math.min(tokens.length, maxLen)
        for (i <- 0 until copyLen) {
          result(i) = tokens(i).toLongOption.getOrElse(tokens(i).hashCode().toLong).toFloat
        }
      case null =>
      case _ =>
    }
    result
  }

  private def columnToLabelTensor(col: Column, spec: LabelSpec): Tensor = {
    val numRows = col.length
    val values = Array.ofDim[Float](numRows)
    val posVal = spec.positiveValue

    for (i <- 0 until numRows) {
      val v = col(i)
      val labelValue = if (v == null) {
        0.0f
      } else {
        val isPositive = (v, posVal) match {
          case (pv: Int, sv: Int) => pv == sv
          case (pv: Long, sv: Long) => pv == sv
          case (pv: Float, sv: Float) => pv == sv
          case (pv: Double, sv: Double) => pv == sv
          case (pv: Int, sv: Long) => pv.toLong == sv
          case (pv: Long, sv: Int) => pv == sv.toLong
          case (pv: Int, sv: Float) => pv.toFloat == sv
          case (pv: Float, sv: Int) => pv == sv.toFloat
          case (pv: String, sv: String) => pv == sv
          case (pv, sv) => pv.toString == sv.toString
        }
        if (isPositive) 1.0f else 0.0f
      }
      values(i) = labelValue
    }

    spec.taskType match {
      case TaskType.Regression => tensor(values, Array(numRows.toLong))
      case _ => tensor(values, Array(numRows.toLong)).toType(ScalarType.Long)
    }
  }

  private def longTensorToColumn(name: String, tensor: Tensor): Column = {
    val arr = try { tensor.toLongArray } catch { case _: Throwable => Array.emptyLongArray }
    val data = mutable.ArrayBuffer[Any]()
    for (v <- arr) data += v.toFloat.asInstanceOf[Any]
    new Column(name, data, DataType.Float32)
  }

  private def floatTensorToColumn(name: String, tensor: Tensor): Column = {
    val arr = try { tensor.toFloatArray } catch { case _: Throwable => Array.emptyFloatArray }
    val data = mutable.ArrayBuffer[Any]()
    for (v <- arr) data += v.asInstanceOf[Any]
    new Column(name, data, DataType.Float32)
  }
}