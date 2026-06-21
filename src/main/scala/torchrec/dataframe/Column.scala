package torchrec.dataframe

import scala.collection.mutable

/** A single column in a DataFrame.
 *  Stores data in a mutable array with associated type information.
 */
class Column(
  val name: String,
  private val data: mutable.ArrayBuffer[Any],
  val dtype: DataType
) {
  require(name != null && name.nonEmpty, "Column name cannot be empty")
  require(data != null, "Column data cannot be null")

  /** Number of elements in this column */
  def length: Int = data.length

  /** Check if column is empty */
  def isEmpty: Boolean = length == 0

  /** Get element at index */
  def apply(i: Int): Any = {
    require(i >= 0 && i < length, s"Index $i out of bounds [0, $length)")
    data(i)
  }

  /** Update element at index */
  def update(i: Int, value: Any): Unit = {
    require(i >= 0 && i < length, s"Index $i out of bounds [0, $length)")
    data(i) = value
  }

  /** Get all values as Scala sequence */
  def toSeq: Seq[Any] = data.toSeq

  /** Check if element at index is null */
  def isNullAt(i: Int): Boolean = data(i) == null

  /** Create a null-mask column */
  def isNull: Column = {
    val nulls = mutable.ArrayBuffer[Boolean]()
    for (i <- 0 until length) {
      nulls += isNullAt(i)
    }
    new Column(name + "_is_null", nulls.map(_.asInstanceOf[Any]), DataType.Boolean)
  }

  /** Fill null values with specified value */
  def fillNull(value: Any): Column = {
    val filled = mutable.ArrayBuffer[Any]()
    for (i <- 0 until length) {
      if (isNullAt(i)) filled += value else filled += data(i)
    }
    new Column(name, filled, dtype)
  }

  /** Drop null values */
  def dropNull: (Column, Column) = {
    val notNull = mutable.ArrayBuffer[Boolean]()
    val notNullData = mutable.ArrayBuffer[Any]()
    val nullMask = mutable.ArrayBuffer[Boolean]()

    for (i <- 0 until length) {
      val notNullVal = !isNullAt(i)
      notNull += notNullVal
      if (notNullVal) notNullData += data(i)
      nullMask += !notNullVal
    }

    val resultCol = new Column(name, notNullData, dtype)
    val nullMaskCol = new Column(name + "_is_null", nullMask.map(_.asInstanceOf[Any]), DataType.Boolean)
    (resultCol, nullMaskCol)
  }

  /** Cast to different type */
  def cast(newType: DataType): Column = {
    if (dtype == newType) return this

    val newData = mutable.ArrayBuffer[Any]()

    (dtype, newType) match
      case (DataType.Int32, DataType.Float32) =>
        for (i <- 0 until length) {
          data(i) match
            case v: Int => newData += v.toFloat
            case v: Float => newData += v
            case _ => newData += 0.0f
        }
        new Column(name, newData, newType)

      case (DataType.Float32, DataType.Int32) =>
        for (i <- 0 until length) {
          data(i) match
            case v: Float => newData += v.toInt
            case v: Int => newData += v
            case _ => newData += 0
        }
        new Column(name, newData, newType)

      case (DataType.String, DataType.Int32) =>
        for (i <- 0 until length) {
          data(i) match
            case s: String => newData += s.toIntOption.getOrElse(0)
            case v: Int => newData += v
            case _ => newData += 0
        }
        new Column(name, newData, newType)

      case _ =>
        throw new UnsupportedOperationException(s"Cast from $dtype to $newType not supported")
  }

  /** Get basic statistics */
  def stats(): ColumnStats = {
    dtype match
      case DataType.Float32 | DataType.Int32 | DataType.Int64 | DataType.Float64 =>
        var minVal = Double.MaxValue
        var maxVal = Double.MinValue
        var sum = 0.0
        var count = 0L

        for (i <- 0 until length) {
          if (!isNullAt(i)) {
            val v = data(i) match
              case f: Float => f.toDouble
              case d: Double => d
              case l: Long => l.toDouble
              case i: Int => i.toDouble
              case _ => Double.NaN
            if (!v.isNaN) {
              if (v < minVal) minVal = v
              if (v > maxVal) maxVal = v
              sum += v
              count += 1
            }
          }
        }

        val mean = if (count > 0) sum / count else Double.NaN
        val variance = if (count > 1) {
          var sqDiff = 0.0
          for (i <- 0 until length) {
            if (!isNullAt(i)) {
              val v = data(i) match
                case f: Float => f.toDouble
                case d: Double => d
                case l: Long => l.toDouble
                case i: Int => i.toDouble
                case _ => Double.NaN
              if (!v.isNaN) sqDiff += (v - mean) * (v - mean)
            }
          }
          sqDiff / count
        } else 0.0

        ColumnStats(
          count = count,
          nullCount = length - count,
          min = if (minVal == Double.MaxValue) Double.NaN else minVal,
          max = if (maxVal == Double.MinValue) Double.NaN else maxVal,
          mean = mean,
          std = math.sqrt(variance)
        )

      case _ =>
        ColumnStats(length, 0, Double.NaN, Double.NaN, Double.NaN, 0.0)
  }

  override def toString: String = s"Column($name, $dtype, $length elements)"
}

/** Statistics for a numeric column */
case class ColumnStats(
  count: Long,
  nullCount: Long,
  min: Double,
  max: Double,
  mean: Double,
  std: Double = 0.0,
  sum: Double = 0.0
)

/** Factory methods for creating columns */
object Column {
  /** Create column from Scala sequence */
  def apply(name: String, values: Seq[Any], dtype: DataType): Column = {
    val data = mutable.ArrayBuffer[Any]()
    dtype match
      case DataType.Float32 =>
        for (v <- values) {
          data += (v match
            case f: Float => f
            case i: Int => i.toFloat
            case d: Double => d.toFloat
            case l: Long => l.toFloat
            case null | _ => Float.NaN
          )
        }
      case DataType.Int32 =>
        for (v <- values) {
          data += (v match
            case i: Int => i
            case f: Float => f.toInt
            case d: Double => d.toInt
            case l: Long => l.toInt
            case null | _ => 0
          )
        }
      case DataType.String =>
        for (v <- values) {
          data += (v match
            case s: String => s
            case null => ""
            case _ => v.toString
          )
        }
      case _ =>
        for (v <- values) {
          data += v
        }

    new Column(name, data, dtype)
  }

  /** Create column from array of floats */
  def apply(name: String, data: Array[Float]): Column = {
    val buffer = mutable.ArrayBuffer[Any]()
    for (f <- data) buffer += f
    new Column(name, buffer, DataType.Float32)
  }

  /** Create column from array of ints */
  def int(name: String, data: Array[Int]): Column = {
    val buffer = mutable.ArrayBuffer[Any]()
    for (i <- data) buffer += i
    new Column(name, buffer, DataType.Int32)
  }

  /** Create column from array of strings */
  def string(name: String, data: Array[String]): Column = {
    val buffer = mutable.ArrayBuffer[Any]()
    for (s <- data) buffer += s
    new Column(name, buffer, DataType.String)
  }
}