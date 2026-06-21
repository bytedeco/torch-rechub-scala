package benchmarks

import torchrec.dataframe.{Column, DataFrame, DataType, FeatureTransformers, JoinType}

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import scala.collection.mutable

object RunDataFrameParityChecks {
  def main(args: Array[String]): Unit = {
    verifyCsvQuotedParsing()
    verifyOneHotExpansion()
    verifyInnerJoinKeepsRightColumns()
    println("[DATAFRAME_PARITY] all checks passed")
  }

  private def verifyCsvQuotedParsing(): Unit = {
    val tmp = Files.createTempFile("adult_like", ".csv")
    try {
      val csv = "\"name\",\"note\"\n\"alice\",\"x,y\"\n\"bob\",\"plain\"\n"
      Files.write(tmp, csv.getBytes(StandardCharsets.UTF_8))
      val df = DataFrame.readCSV(tmp.toString)
      require(df.col("note")(0).toString == "x,y", s"Quoted CSV parse failed: ${df.col("note")(0)}")
    } finally {
      Files.deleteIfExists(tmp)
    }
  }

  private def verifyOneHotExpansion(): Unit = {
    val rows = Seq(
      Map("color" -> "red", "value" -> 1),
      Map("color" -> "blue", "value" -> 2),
      Map("color" -> "red", "value" -> 3)
    )
    val df = DataFrame.fromRows(rows)
    val out = FeatureTransformers.oneHotEncoder(sparseOutput = false).fitTransform(df)
    val expanded = out.columns.filter(_.startsWith("color_"))
    require(expanded.size >= 2, s"OneHot did not expand categories: ${out.columns}")
    require(out.columns.contains("value"), "OneHot should preserve non-categorical columns")
  }

  private def verifyInnerJoinKeepsRightColumns(): Unit = {
    val left = DataFrame.fromRows(Seq(Map("id" -> 1, "a" -> "L1"), Map("id" -> 2, "a" -> "L2")))
    val right = DataFrame.fromRows(Seq(Map("id" -> 1, "b" -> "R1"), Map("id" -> 1, "b" -> "R1x"), Map("id" -> 3, "b" -> "R3")))

    val joined = left.join(right, on = "id", how = JoinType.Inner)
    require(joined.columns.contains("a"), "Join lost left column")
    require(joined.columns.contains("b"), "Join lost right column")
    require(joined.numRows == 2, s"Join row cardinality wrong: ${joined.numRows}")
  }
}

