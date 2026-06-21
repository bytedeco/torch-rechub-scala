package torchrec.dataframe

import scala.collection.mutable
import scala.collection.immutable.ListMap

/** DataFrame with columnar storage.
 *  Provides pandas-like API for data manipulation.
 */
class DataFrame(
  private val _columnMap: Map[String, Column],
  val schema: StructType
) {
  // Always use ListMap internally to preserve insertion order
  private val columnMap = _columnMap.to(collection.immutable.ListMap)

  require(_columnMap.nonEmpty || schema.fields.isEmpty, "DataFrame must have at least one column or empty schema")

  if (_columnMap.nonEmpty) {
    require(_columnMap.keySet == schema.fields.map(_.name).toSet,
      "Column names must match schema")
  }

  /** Factory method that infers schema from columns */
  def this(columns: Map[String, Column]) = {
    this(columns, StructType(columns.map { case (name, col) => StructField(name, col.dtype) }.toSeq))
  }

  /** Get column by name */
  def apply(col: String): Column = columnMap(col)

  /** Get column by name (alternative) */
  def col(col: String): Column = columnMap(col)

  /** Get column names */
  def columns: List[String] = columnMap.keys.toList

  /** Get column map directly */
  def columnData: Map[String, Column] = columnMap

  /** Number of rows */
  def numRows: Int = if (columnMap.values.isEmpty) 0 else columnMap.values.head.length

  /** Number of columns */
  def numCols: Int = columnMap.size

  /** Check if DataFrame is empty */
  def isEmpty: Boolean = numRows == 0

  // =========================================================================
  // Selection Operations
  // =========================================================================

  /** Select columns by name */
  def select(cols: String*): DataFrame = {
    val selectedCols = cols.map(c => c -> columnMap(c)).toMap
    val selectedSchema = StructType(cols.flatMap(c => schema.fields.find(_.name == c)))
    new DataFrame(selectedCols, selectedSchema)
  }

  /** Drop columns by name */
  def drop(cols: String*): DataFrame = {
    val remainingCols = columnMap.filter { case (name, _) => !cols.contains(name) }
    val remainingSchema = StructType(schema.fields.filter(f => !cols.contains(f.name)))
    new DataFrame(remainingCols, remainingSchema)
  }

  /** Select columns by index range */
  def selectCols(range: Range): DataFrame = {
    val names = columnMap.keys.toList.slice(range.start, range.end)
    select(names: _*)
  }

  // =========================================================================
  // Filtering Operations
  // =========================================================================

  /** Filter rows by boolean column condition */
  def filter(condition: Column): DataFrame = {
    require(condition.dtype == DataType.Boolean, "Filter condition must be boolean")
    require(condition.length == numRows, "Filter condition length must match DataFrame rows")

    val mask = new Array[Boolean](numRows)
    for (i <- 0 until numRows) {
      mask(i) = condition(i).asInstanceOf[Boolean]
    }

    val newColumns = columnMap.map { case (name, col) =>
      (name, filterColumn(col, mask))
    }

    val newSchema = StructType(schema.fields.filter(f => newColumns.contains(f.name)))
    new DataFrame(newColumns, newSchema)
  }

  /** Filter using a predicate function */
  def filter(pred: Any => Boolean): DataFrame = {
    val firstCol = columnMap.values.head
    val mask = new Array[Boolean](numRows)
    for (i <- 0 until numRows) {
      mask(i) = pred(firstCol(i))
    }

    val newColumns = columnMap.map { case (name, col) =>
      (name, filterColumn(col, mask))
    }

    val newSchema = StructType(schema.fields.filter(f => newColumns.contains(f.name)))
    new DataFrame(newColumns, newSchema)
  }

  private def filterColumn(col: Column, mask: Array[Boolean]): Column = {
    val newData = mutable.ArrayBuffer[Any]()
    for (i <- 0 until col.length) {
      if (mask(i)) newData += col(i)
    }
    new Column(col.name, newData, col.dtype)
  }

  /** Limit number of rows */
  def limit(n: Int): DataFrame = {
    if (n >= numRows) return this

    val newColumns = columnMap.map { case (name, col) =>
      val newData = mutable.ArrayBuffer[Any]()
      for (i <- 0 until n) {
        newData += col(i)
      }
      (name, new Column(name, newData, col.dtype))
    }

    val newSchema = StructType(schema.fields.filter(f => newColumns.contains(f.name)))
    new DataFrame(newColumns, newSchema)
  }

  /** Take first n rows */
  def head(n: Int = 5): DataFrame = limit(n)

  // =========================================================================
  // Column Operations
  // =========================================================================

  /** Add or replace a column */
  def withColumn(name: String, col: Column): DataFrame = {
    val newCol = if (col.name != name) {
      new Column(name, col.toSeq.to(mutable.ArrayBuffer), col.dtype)
    } else col

    val newColumns = columnMap + (name -> newCol)
    val newField = StructField(name, col.dtype)
    val newSchema = if (schema.contains(name)) {
      StructType(schema.fields.map(f => if (f.name == name) newField else f))
    } else {
      StructType(schema.fields :+ newField)
    }
    new DataFrame(newColumns, newSchema)
  }

  /** Rename a column */
  def rename(oldName: String, newName: String): DataFrame = {
    require(columnMap.contains(oldName), s"Column '$oldName' not found")
    val col = columnMap(oldName)
    val newCol = new Column(newName, col.toSeq.to(mutable.ArrayBuffer), col.dtype)
    val newColumns = columnMap - oldName + (newName -> newCol)
    val newSchema = StructType(
      schema.fields.map(f => if (f.name == oldName) f.copy(name = newName) else f)
    )
    new DataFrame(newColumns, newSchema)
  }

  /** Cast column to different type */
  def cast(colName: String, newType: DataType): DataFrame = {
    require(columnMap.contains(colName), s"Column '$colName' not found")
    val oldCol = columnMap(colName)
    val newCol = oldCol.cast(newType)
    val newColumns = columnMap + (colName -> newCol)
    new DataFrame(newColumns, schema)
  }

  // =========================================================================
  // Aggregation Operations
  // =========================================================================

  /** Group by columns */
  def groupBy(cols: String*): GroupedDataFrame = {
    val missing = cols.filterNot(columns.contains)
    require(missing.isEmpty, s"Columns not found: $missing")
    new GroupedDataFrame(this, cols.toList)
  }

  // =========================================================================
  // Binary Operations
  // =========================================================================

  /** Union with another DataFrame */
  def union(other: DataFrame): DataFrame = {
    val newColumns = columnMap.map { case (name, col) =>
      val otherCol = other.columnMap(name)
      val newData = mutable.ArrayBuffer[Any]()
      for (v <- col.toSeq) newData += v
      for (v <- otherCol.toSeq) newData += v
      (name, new Column(name, newData, col.dtype))
    }
    new DataFrame(newColumns, schema)
  }

  /** Join with another DataFrame */
  def join(other: DataFrame, on: String, how: JoinType = JoinType.Inner): DataFrame = {
    require(columns.contains(on), s"Join column '$on' not found in left DataFrame")
    require(other.columns.contains(on), s"Join column '$on' not found in right DataFrame")

    how match
      case JoinType.Inner => innerJoin(other, on)
      case JoinType.Left => leftJoin(other, on)
      case _ => throw new UnsupportedOperationException(s"Join type $how not implemented")
  }

  private def innerJoin(other: DataFrame, on: String): DataFrame = {
    val rightCol = other.columnMap(on)
    val rightMap = mutable.Map[Any, mutable.ListBuffer[Int]]()
    for (i <- 0 until rightCol.length) {
      val key = rightCol(i)
      rightMap.getOrElseUpdate(key, mutable.ListBuffer()) += i
    }

    val outputColumns = mutable.LinkedHashMap[String, mutable.ArrayBuffer[Any]]()
    val outputTypes = mutable.LinkedHashMap[String, DataType]()

    for ((name, col) <- columnMap) {
      outputColumns(name) = mutable.ArrayBuffer[Any]()
      outputTypes(name) = col.dtype
    }
    for ((name, col) <- other.columnMap if name != on) {
      val outName = if (outputColumns.contains(name)) s"${name}_right" else name
      outputColumns(outName) = mutable.ArrayBuffer[Any]()
      outputTypes(outName) = col.dtype
    }

    val leftCol = columnMap(on)
    var matchedRows = 0
    for (leftIdx <- 0 until leftCol.length) {
      val key = leftCol(leftIdx)
      rightMap.get(key).foreach { rightIndices =>
        rightIndices.foreach { rightIdx =>
          for ((name, col) <- columnMap) {
            outputColumns(name) += col(leftIdx)
          }
          for ((name, col) <- other.columnMap if name != on) {
            val outName = if (columnMap.contains(name)) s"${name}_right" else name
            outputColumns(outName) += col(rightIdx)
          }
          matchedRows += 1
        }
      }
    }

    if (matchedRows == 0) {
      return DataFrame.empty()
    }

    val finalColumns = outputColumns.map { case (name, data) =>
      name -> new Column(name, data, outputTypes(name))
    }.toMap
    val finalSchema = StructType(finalColumns.toSeq.map { case (name, col) => StructField(name, col.dtype) })
    new DataFrame(finalColumns, finalSchema)
  }

  private def leftJoin(other: DataFrame, on: String): DataFrame = {
    throw new UnsupportedOperationException("Left join not yet implemented")
  }

  // =========================================================================
  // Evaluation
  // =========================================================================

  /** Force evaluation and return as Row sequence */
  def collect(): Seq[Row] = {
    (0 until numRows).map(i => Row.fromIndex(this, i))
  }

  /** Convert to Map sequence */
  def toMapSeq: Seq[Map[String, Any]] = {
    collect().map(_.toMap)
  }

  /** Get column statistics */
  def describe(cols: String*): DataFrame = {
    val targetCols = if (cols.isEmpty) columnMap.keys.toList else cols.toList

    val statsMap = mutable.Map[String, ColumnStats]()
    for (colName <- targetCols) {
      val col = columnMap(colName)
      statsMap(colName) = col.stats()
    }

    // Return a DataFrame with statistics
    val statsData = mutable.Map[String, Column]()
    for ((colName, stats) <- statsMap) {
      val data = mutable.ArrayBuffer[Any]()
      data += stats.count
      data += stats.nullCount
      data += stats.min
      data += stats.max
      data += stats.mean
      statsData(s"${colName}_stats") = new Column(s"${colName}_stats", data, DataType.Float64)
    }

    if (statsData.isEmpty) {
      DataFrame.empty()
    } else {
      val columns = statsData.toMap
      val schema = StructType(statsData.map { case (n, c) => StructField(n, c.dtype) }.toSeq)
      new DataFrame(columns, schema)
    }
  }

  // =========================================================================
  // String Operations
  // =========================================================================

  /** Access column for string operations */
  def str(col: String): StringColumnOps = {
    require(columnMap.contains(col), s"Column '$col' not found")
    require(columnMap(col).dtype == DataType.String, s"Column '$col' is not String type")
    new StringColumnOps(columnMap(col))
  }

  // =========================================================================
  // Show / Print
  // =========================================================================

  /** Show DataFrame (like pandas head) */
  def show(n: Int = 20, truncate: Boolean = true): Unit = {
    val rows = 0 until math.min(n, numRows)
    val colWidths = columnMap.map { case (name, col) =>
      name -> math.max(name.length, if (truncate) 10 else col.length.toString.length)
    }

    // Header
    println(colWidths.map { case (name, w) => name.padTo(w, ' ') }.mkString(" | "))
    println(colWidths.map { case (_, w) => "-".repeat(w) }.mkString("-+-"))

    // Rows
    for (i <- rows) {
      val values = columnMap.map { case (name, col) =>
        val v = col(i)
        val s = if (v == null) "null" else v.toString
        val truncated = if (truncate && s.length > 10) s.substring(0, 7) + "..." else s
        truncated.padTo(colWidths(name), ' ')
      }
      println(values.mkString(" | "))
    }

    if (n < numRows) {
      println(s"... ${numRows - n} more rows")
    }
  }

  override def toString: String = {
    s"DataFrame(${numRows}x$numCols, columns: ${columnMap.keys.mkString(", ")})"
  }

  /** Create DataFrame with auto-inferred schema from column map */
  def apply(columns: Map[String, Column]): DataFrame = {
    val schema = StructType(columns.map { case (n, c) => StructField(n, c.dtype) }.toSeq)
    new DataFrame(columns, schema)
  }


  /** Write DataFrame to CSV file */
  def writeCSV(path: String, delimiter: Char = ','): Unit = {
    val cols = columnMap.keys.toList
    val nRows = if (columnMap.values.isEmpty) 0 else columnMap.values.head.length
    val header = cols.mkString(delimiter.toString)
    val csvLines = (0 until nRows).map { rowIdx =>
      cols.map { colName =>
        val value = columnMap(colName)(rowIdx)
        if (value == null) "" else value.toString
      }.mkString(delimiter.toString)
    }

    val content = header + "\n" + csvLines.mkString("\n")
    java.nio.file.Files.write(
      java.nio.file.Paths.get(path),
      content.getBytes(java.nio.charset.StandardCharsets.UTF_8)
    )
  }

  /** Export DataFrame to Map sequence */
//  def toMapSeq: Seq[Map[String, Any]] = {
//    val cols = columnMap.keys.toList
//    val nRows = if (columnMap.values.isEmpty) 0 else columnMap.values.head.length
//    (0 until nRows).map { rowIdx =>
//      cols.map { colName =>
//        colName -> columnMap(colName)(rowIdx)
//      }.toMap
//    }
//  }

  /** Export DataFrame to JSON file */
  def writeJSON(path: String): Unit = {
    val json = DataFrame.this.toMapSeq.map(_.toString).mkString("[", ",", "]")
    java.nio.file.Files.write(
      java.nio.file.Paths.get(path),
      json.getBytes(java.nio.charset.StandardCharsets.UTF_8)
    )
  }

}

/** Companion object for DataFrame factory methods */
object DataFrame {
  /** Create empty DataFrame */
  def empty(): DataFrame = {
    new DataFrame(Map.empty, StructType(Nil))
  }

  /** Create DataFrame from rows */
  def fromRows(rows: Seq[Map[String, Any]]): DataFrame = {
    if (rows.isEmpty) return empty()

    val columnData = mutable.LinkedHashMap[String, mutable.ArrayBuffer[Any]]()
    for ((k, v) <- rows.head) {
      val buffer = mutable.ArrayBuffer[Any]()
      for (row <- rows) {
        buffer += row.getOrElse(k, null)
      }
      columnData(k) = buffer
    }

    val colMap = columnData.map { case (name, values) =>
      val firstVal = values.headOption.getOrElse("")
      val dtype = firstVal match {
        case _: Int => DataType.Int32
        case _: Long => DataType.Int64
        case _: Float => DataType.Float32
        case _: Double => DataType.Float64
        case _: Boolean => DataType.Boolean
        case _: String => DataType.String
        case _ => DataType.String
      }
      (name, new Column(name, values, dtype))
    }.toMap

    DataFrame(colMap)
  }

  /** Read CSV file */
  def readCSV(path: String, delimiter: Char = ','): DataFrame = {
    import scala.io.Source
    val source = Source.fromFile(path)
    val lines = source.getLines().toVector
    source.close()

    if (lines.isEmpty) return empty()

    val header = parseCsvLine(lines.head, delimiter).map(unquoteCsvField).map(_.trim)
    val dataLines = lines.tail

    // Debug: print header and first few parsed lines to help diagnose parsing issues
    try {
      System.err.println(s"[readCSV DEBUG] header(${header.length})=${header.take(10).mkString(",")}")
      val sampleN = math.min(5, dataLines.length)
      for (i <- 0 until sampleN) {
        val pl = parseCsvLine(dataLines(i), delimiter).map(unquoteCsvField).map(_.trim)
        System.err.println(s"[readCSV DEBUG] parsed line ${i + 1}: cols=${pl.length} sample=${pl.take(10).mkString("[", ",", "]")} ")
      }
    } catch { case _: Throwable => () }

    val columnData = mutable.LinkedHashMap[String, mutable.ArrayBuffer[Any]]()
    for (col <- header) {
      columnData(col) = mutable.ArrayBuffer(dataLines.map { line =>
        val cols = parseCsvLine(line, delimiter)
        val idx = header.indexOf(col)
        if (idx >= 0 && cols.length > idx) unquoteCsvField(cols(idx)).trim else ""
      }: _*)
    }

    val colMap = columnData.map { case (name, values) =>
      val stringValues = values.asInstanceOf[mutable.ArrayBuffer[Any]].map(_.toString)
      val dtype = inferType(stringValues.toVector)
      val data = mutable.ArrayBuffer[Any]()
      for (v <- values) {
        val s = v.toString
        dtype match {
              case DataType.Int32 => data += s.toIntOption.getOrElse(0)
              case DataType.Int64 => data += s.toLongOption.getOrElse(0L)
              case DataType.Float32 => data += s.toFloatOption.getOrElse(Float.NaN)
              case DataType.Float64 => data += s.toDoubleOption.getOrElse(Double.NaN)
              case DataType.Boolean => data += s.toBooleanOption.getOrElse(false)
          case _ => data += s
        }
      }
      (name, Column(name, data.toSeq, dtype))
    }.toMap

    DataFrame(colMap)
  }

//  /** Write DataFrame to CSV file */
//  def writeCSV(path: String, delimiter: Char = ','): Unit = {
//    val cols = columnMap.keys.toList
//    val nRows = if (columnMap.values.isEmpty) 0 else columnMap.values.head.length
//    val header = cols.mkString(delimiter.toString)
//    val csvLines = (0 until nRows).map { rowIdx =>
//      cols.map { colName =>
//        val value = columnMap(colName)(rowIdx)
//        if (value == null) "" else value.toString
//      }.mkString(delimiter.toString)
//    }
//
//    val content = header + "\n" + csvLines.mkString("\n")
//    java.nio.file.Files.write(
//      java.nio.file.Paths.get(path),
//      content.getBytes(java.nio.charset.StandardCharsets.UTF_8)
//    )
//  }
//
//  /** Export DataFrame to Map sequence */
//  def toMapSeq: Seq[Map[String, Any]] = {
//    val cols = columnMap.keys.toList
//    val nRows = if (columnMap.values.isEmpty) 0 else columnMap.values.head.length
//    (0 until nRows).map { rowIdx =>
//      cols.map { colName =>
//        colName -> columnMap(colName)(rowIdx)
//      }.toMap
//    }
//  }

  /** Export DataFrame to JSON file */
//  def writeJSON(path: String): Unit = {
//    val json = DataFrame.this.toMapSeq.map(_.toString).mkString("[", ",", "]")
//    java.nio.file.Files.write(
//      java.nio.file.Paths.get(path),
//      json.getBytes(java.nio.charset.StandardCharsets.UTF_8)
//    )
//  }

  private def inferType(values: Vector[String]): DataType = {
    val sample = values.filter(_.nonEmpty).take(100)
    if (sample.isEmpty) return DataType.String
    if (sample.forall(v => v.toLongOption.isDefined)) DataType.Int64
    else if (sample.forall(v => v.toDoubleOption.isDefined)) DataType.Float64
    else if (sample.forall(v => v.toBooleanOption.isDefined)) DataType.Boolean
    else DataType.String
  }

  private def parseCsvLine(line: String, delimiter: Char): Array[String] = {
    val fields = mutable.ArrayBuffer[String]()
    val current = new StringBuilder
    var inQuotes = false
    var i = 0
    while (i < line.length) {
      val ch = line.charAt(i)
      if (ch == '"') {
        if (inQuotes && i + 1 < line.length && line.charAt(i + 1) == '"') {
          current.append('"')
          i += 1
        } else {
          inQuotes = !inQuotes
        }
      } else if (ch == delimiter && !inQuotes) {
        fields += current.result()
        current.clear()
      } else {
        current.append(ch)
      }
      i += 1
    }
    fields += current.result()
    fields.toArray
  }

  private def unquoteCsvField(value: String): String = {
    if (value.length >= 2 && value.head == '"' && value.last == '"') value.substring(1, value.length - 1)
    else value
  }
}

/** Row representation */
class Row(df: DataFrame, index: Int) {
  def apply(col: String): Any = df(col)(index)
  def getString(col: String): String = df(col)(index).toString
  def getInt(col: String): Int = df(col)(index) match { case i: Int => i case _ => 0 }
  def getLong(col: String): Long = df(col)(index) match { case l: Long => l case i: Int => i case _ => 0L }
  def getFloat(col: String): Float = df(col)(index) match { case f: Float => f case _ => 0.0f }
  def getDouble(col: String): Double = df(col)(index) match { case d: Double => d case f: Float => f case _ => 0.0 }
  def getBoolean(col: String): Boolean = df(col)(index).asInstanceOf[Boolean]
  def isNullAt(col: String): Boolean = df(col).isNullAt(index)
  def toMap: Map[String, Any] = df.columns.map(c => c -> df(c)(index)).toMap
  def toSeq: Seq[Any] = df.columns.map(c => df(c)(index))
  def size: Int = df.numCols
}

object Row {
  def fromIndex(df: DataFrame, i: Int): Row = new Row(df, i)
  def concat(rows: Seq[Row]): DataFrame = {
    if (rows.isEmpty) return DataFrame.empty()
    val firstRow = rows.head
    val columns = mutable.Map[String, mutable.ArrayBuffer[Any]]()
    for (colName <- firstRow.toMap.keys) {
      columns(colName) = mutable.ArrayBuffer()
    }
    for (row <- rows) {
      for ((colName, value) <- row.toMap) {
        columns(colName) += value
      }
    }
    val cols = columns.map { case (name, data) =>
      val dtype = DataType.fromClass(data.head.getClass)
      name -> new Column(name, data, dtype)
    }.toMap
    DataFrame(cols)
  }
}

/** Grouped DataFrame from groupBy operation */
class GroupedDataFrame(
  df: DataFrame,
  groupCols: List[String]
) {
  /** Aggregate with specified aggregations */
  def agg(aggs: Aggregation*): DataFrame = {
    val groupCol = df.columnData(groupCols.head)
    val numGroups = countUnique(groupCol)
    val uniqueValues = extractUnique(groupCol, numGroups)

    val aggCols = mutable.Map[String, Column]()

    for (agg <- aggs) {
      val col = df.columnData(agg.column)
      val result = mutable.ArrayBuffer[Any]()

      for (g <- 0 until numGroups) {
        val gv = uniqueValues(g)
        val filtered = filterByGroupValue(col, groupCol, gv)
        val stats = filtered.stats()
        val value: Any = agg.aggType match {
          case AggregationType.Count => stats.count.toFloat
          case AggregationType.Sum => stats.sum.toFloat
          case AggregationType.Mean | AggregationType.Avg => stats.mean.toFloat
          case AggregationType.Min => stats.min.toFloat
          case AggregationType.Max => stats.max.toFloat
          case AggregationType.Std => stats.std.toFloat
          case AggregationType.Var => (stats.std * stats.std).toFloat
          case AggregationType.Median => stats.mean.toFloat
          case AggregationType.First => filtered(0)
          case AggregationType.Last => filtered(filtered.length - 1)
          case AggregationType.NUnique => numGroups.toFloat
          case AggregationType.Distinct => numGroups.toFloat
          case _ => 0.0f
        }
        result += value
      }

      aggCols(agg.outputName) = new Column(agg.outputName, result, DataType.Float32)
    }

    // Build group key columns
    val keyData = mutable.ArrayBuffer[Any]()
    for (g <- 0 until numGroups) {
      keyData += uniqueValues(g)
    }
    val keyCols = Map(groupCols.head -> new Column(groupCols.head, keyData, DataType.String))

    val allCols = keyCols ++ aggCols.toMap
    val newSchema = StructType(
      keyCols.map { case (n, c) => StructField(n, c.dtype) }.toSeq ++
      aggCols.map { case (n, c) => StructField(n, c.dtype) }.toSeq
    )
    new DataFrame(allCols, newSchema)
  }

  private def countUnique(col: Column): Int = {
    val seen = mutable.Set[Any]()
    for (i <- 0 until col.length) {
      seen.add(col(i))
    }
    seen.size
  }

  private def extractUnique(col: Column, numUnique: Int): Seq[Any] = {
    val seen = mutable.Set[Any]()
    val result = mutable.ArrayBuffer[Any]()
    for (i <- 0 until col.length) {
      val v = col(i)
      if (!seen.contains(v)) {
        seen.add(v)
        result += v
      }
    }
    result.toSeq
  }

  private def filterByGroupValue(col: Column, groupCol: Column, groupValue: Any): Column = {
    val newData = mutable.ArrayBuffer[Any]()
    for (i <- 0 until col.length) {
      if (groupCol(i) == groupValue) {
        newData += col(i)
      }
    }
    new Column(col.name, newData, col.dtype)
  }

  /** Count rows in each group */
  def count(): DataFrame = {
    val groupCol = df.columnData(groupCols.head)
    val numGroups = countUnique(groupCol)
    val uniqueValues = extractUnique(groupCol, numGroups)

    val countData = mutable.ArrayBuffer[Any]()
    for (g <- 0 until numGroups) {
      var c = 0
      for (i <- 0 until groupCol.length) {
        if (groupCol(i) == uniqueValues(g)) c += 1
      }
      countData += c.toFloat
    }

    val keyData = mutable.ArrayBuffer[Any]()
    for (g <- 0 until numGroups) {
      keyData += uniqueValues(g)
    }

    val keyCols = Map(groupCols.head -> new Column(groupCols.head, keyData, DataType.String))
    val countCol = new Column("count", countData, DataType.Float32)
    val allCols = keyCols + ("count" -> countCol)
    val newSchema = StructType(
      StructField(groupCols.head, DataType.String) :: StructField("count", DataType.Float32) :: Nil
    )
    new DataFrame(allCols, newSchema)
  }
}

/** String column operations */
class StringColumnOps(col: Column) {
  require(col.dtype == DataType.String, "StringColumnOps requires String column")

  def length: Column = {
    val newData = mutable.ArrayBuffer[Any]()
    for (i <- 0 until col.length) {
      val s = col(i).toString
      newData += s.length
    }
    new Column(col.name + "_length", newData, DataType.Int32)
  }

  def upper: Column = {
    val newData = mutable.ArrayBuffer[Any]()
    for (i <- 0 until col.length) {
      newData += col(i).toString.toUpperCase
    }
    new Column(col.name + "_upper", newData, DataType.String)
  }

  def lower: Column = {
    val newData = mutable.ArrayBuffer[Any]()
    for (i <- 0 until col.length) {
      newData += col(i).toString.toLowerCase
    }
    new Column(col.name + "_lower", newData, DataType.String)
  }

  def contains(substring: String): Column = {
    val newData = mutable.ArrayBuffer[Any]()
    for (i <- 0 until col.length) {
      newData += col(i).toString.contains(substring)
    }
    new Column(col.name + "_contains", newData, DataType.Boolean)
  }

  def trim: Column = {
    val newData = mutable.ArrayBuffer[Any]()
    for (i <- 0 until col.length) {
      newData += col(i).toString.trim
    }
    new Column(col.name + "_trim", newData, DataType.String)
  }
}