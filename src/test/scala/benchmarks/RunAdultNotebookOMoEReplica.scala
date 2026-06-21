package benchmarks

import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.tensor
import torchrec.basic.features.SparseFeature
import torchrec.data.{DataLoader, MultiTaskDataset}
import torchrec.dataframe.{Column, DataFrame, DataType, FeatureTransformers, JoinType}
import torchrec.models.multi_task.OMoE
import torchrec.trainers.MTLTrainer

import scala.collection.mutable
import scala.util.Random

/**
 * Adult Census Income notebook-style replica using the local DataFrame/Pipeline stack
 * and an OMoE multi-task model.
 */
object RunAdultNotebookOMoEReplica {

  def main(args: Array[String]): Unit = {
    val params = PythonRankingReplicaSupport.parseArgs(args)
    val csvPath = params.getOrElse("dataset", "/home/muller/IdeaProjects/torch-rechub-scala/src/main/resources/adult.csv")
    val seed = params.getOrElse("seed", "2024").toInt
    val batchSize = params.getOrElse("batch_size", "1024").toInt
    val epochs = params.getOrElse("epochs", "200").toInt
    val device = params.getOrElse("device", "cuda")

    val raw = DataFrame.readCSV(csvPath)
    require(raw.numRows > 1000, s"Unexpectedly small dataset: ${raw.numRows}")

    val cleaned = notebookStyleCleaning(raw)
    val withId = addRowId(cleaned, "id")

    val numericCols = List("age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week")
    val categoricalCols = List(
      "workclass", "education", "marital.status", "occupation",
      "relationship", "race", "sex", "native.country"
    )

    val numericScaled = {
      val scaled = FeatureTransformers.standardScaler().fitTransform(withId.select(numericCols*))
      scaled.withColumn("id", withId.col("id"))
    }
    val categoricalEncoded = FeatureTransformers.oneHotEncoder(sparseOutput = false).fitTransform(
      withId.select((categoricalCols :+ "id")*)
    )

    // Verify key pandas/sklearn parity points used by the notebook path.
    require(categoricalEncoded.numCols > categoricalCols.size, "OneHotEncoder did not expand columns")

    val merged = categoricalEncoded.join(numericScaled, on = "id", how = JoinType.Inner)
    require(merged.numRows == withId.numRows, s"Merged rows mismatch: ${merged.numRows} vs ${withId.numRows}")

    val withLabels = merged
      .withColumn("income_label", buildBinaryLabel(withId.col("income"), ">50K"))
      .withColumn("marital_label", buildBinaryLabel(withId.col("marital.status"), "Married"))

    val rawFeatureCols = withLabels.columns.filterNot(c => c == "id" || c == "income_label" || c == "marital_label")
    val renameMap = rawFeatureCols.map(c => c -> sanitizeFeatureName(c)).toMap
    val renamed = renameMap.foldLeft(withLabels) { case (acc, (oldName, newName)) =>
      if (oldName == newName) acc else acc.rename(oldName, newName)
    }

    val featureCols = renamed.columns.filterNot(c => c == "id" || c == "income_label" || c == "marital_label")
    val encoded = encodeAsSparseIds(renamed, featureCols)

    val (trainDS, validDS, testDS) = splitMultiTask(encoded, seed = seed)

    val features = featureCols.map(name => SparseFeature(name, vocabSize = inferredVocab(encoded, name) + 2, embedDim = 8))
    val model = new OMoE(
      features = features,
      taskNames = List("income", "marital"),
      embedDim = 8,
      numExperts = 4,
      expertDims = List(64L),
      towerDims = List(32L),
      dropout = 0.2f,
      device = device
    )

    val trainLoader = new DataLoader(trainDS, batchSize = batchSize, shuffle = true, device = device)
    val validLoader = new DataLoader(validDS, batchSize = batchSize, shuffle = false, device = device)
    val testLoader = new DataLoader(testDS, batchSize = batchSize, shuffle = false, device = device)

    val trainer = new MTLTrainer(
      model = model,
      taskNames = List("income", "marital"),
      learningRate = 1e-3f,
      weightDecay = 1e-5f,
      device = device,
      numEpochs = epochs,
      earlyStopPatience = 300,
      verbose = true
    )

    trainer.fit(trainLoader, Some(validLoader))
    val metrics = trainer.evaluate(testLoader)
    println(s"[ADULT_NOTEBOOK_OMOE] metrics=$metrics")

    val incomeAuc = metrics.getOrElse("income_AUC", 0.0f)
    require(!incomeAuc.isNaN && incomeAuc >= 0.0f && incomeAuc <= 1.0f, f"Invalid income AUC: $incomeAuc%.4f")
  }

  private def notebookStyleCleaning(df: DataFrame): DataFrame = {
    val unknownCols = List("native.country", "occupation", "workclass")
    val modePerCol = unknownCols.map(c => c -> mode(df.col(c), skip = Set("?", ""))).toMap

    var out = df
    for (col <- unknownCols) {
      out = mapColumn(out, col) { v =>
        val s = normalizeString(v)
        if (s == "?" || s.isEmpty) modePerCol(col) else s
      }
    }

    out = mapColumn(out, "education") { v =>
      val s = normalizeString(v)
      if (Set("HS-grad", "11th", "10th", "9th", "12th").contains(s)) "HS-grad" else s
    }

    out = mapColumn(out, "marital.status") { v =>
      val s = normalizeString(v)
      if (Set("Married-spouse-absent", "Married-civ-spouse", "Married-AF-spouse").contains(s)) "Married"
      else if (Set("Separated", "Divorced").contains(s)) "Separated"
      else s
    }

    out = mapColumn(out, "workclass") { v =>
      val s = normalizeString(v)
      if (Set("Self-emp-not-inc", "Self-emp-inc").contains(s)) "Self-Employed"
      else if (Set("Local-gov", "State-gov", "Federal-gov").contains(s)) "Government"
      else s
    }

    val mask = mutable.ArrayBuffer[Any]()
    val ageCol = out.col("age")
    val hoursCol = out.col("hours.per.week")
    for (i <- 0 until out.numRows) {
      val age = toDouble(ageCol(i))
      val hours = toDouble(hoursCol(i))
      mask += (age != 90.0 && hours != 99.0)
    }

    out.filter(new Column("keep", mask, DataType.Boolean))
  }

  private def addRowId(df: DataFrame, idCol: String): DataFrame = {
    val idData = mutable.ArrayBuffer[Any]()
    for (i <- 0 until df.numRows) idData += i
    df.withColumn(idCol, new Column(idCol, idData, DataType.Int32))
  }

  private def mapColumn(df: DataFrame, colName: String)(f: Any => Any): DataFrame = {
    val src = df.col(colName)
    val data = mutable.ArrayBuffer[Any]()
    for (i <- 0 until src.length) data += f(src(i))
    df.withColumn(colName, new Column(colName, data, DataType.String))
  }

  private def mode(col: Column, skip: Set[String]): String = {
    val counter = mutable.Map[String, Int]()
    for (i <- 0 until col.length) {
      val s = normalizeString(col(i))
      if (!skip.contains(s)) counter(s) = counter.getOrElse(s, 0) + 1
    }
    if (counter.isEmpty) "unknown" else counter.maxBy(_._2)._1
  }

  private def normalizeString(v: Any): String = if (v == null) "" else v.toString.trim

  private def sanitizeFeatureName(name: String): String = {
    val cleaned = name.replaceAll("[^A-Za-z0-9_]+", "_").stripPrefix("_").stripSuffix("_")
    if (cleaned.nonEmpty) cleaned else "feature"
  }

  private def buildBinaryLabel(col: Column, positiveContains: String): Column = {
    val data = mutable.ArrayBuffer[Any]()
    for (i <- 0 until col.length) {
      val s = normalizeString(col(i))
      data += (if (s.contains(positiveContains)) 1.0f else 0.0f)
    }
    new Column("label", data, DataType.Float32)
  }

  private def encodeAsSparseIds(df: DataFrame, featureCols: List[String]): DataFrame = {
    val vocabByCol = featureCols.map { name =>
      val col = df.col(name)
      val vocab = mutable.LinkedHashMap[String, Int]()
      var nextId = 1
      for (i <- 0 until col.length) {
        val key = normalizeString(col(i))
        if (!vocab.contains(key)) {
          vocab(key) = nextId
          nextId += 1
        }
      }
      name -> vocab.toMap
    }.toMap

    var out = df
    for (name <- featureCols) {
      val src = df.col(name)
      val data = mutable.ArrayBuffer[Any]()
      for (i <- 0 until src.length) {
        data += vocabByCol(name).getOrElse(normalizeString(src(i)), 0).toFloat
      }
      out = out.withColumn(name, new Column(name, data, DataType.Float32))
    }
    out
  }

  private def inferredVocab(df: DataFrame, colName: String): Int = {
    val col = df.col(colName)
    val values = mutable.Set[Int]()
    for (i <- 0 until col.length) values += toDouble(col(i)).toInt
    values.size
  }

  private def splitMultiTask(df: DataFrame, seed: Int): (MultiTaskDataset, MultiTaskDataset, MultiTaskDataset) = {
    val n = df.numRows
    val indices = Random(seed).shuffle((0 until n).toList)
    val trainN = (n * 0.7).toInt
    val validN = (n * 0.15).toInt
    val trainIdx = indices.take(trainN).toSet
    val validIdx = indices.slice(trainN, trainN + validN).toSet

    def build(part: String): MultiTaskDataset = {
      val featureMap = mutable.Map[String, Array[Float]]()
      val featureNames = df.columns.filterNot(c => c == "income_label" || c == "marital_label" || c == "id")
      featureNames.foreach(name => featureMap(name) = mutable.ArrayBuffer[Float]().toArray)

      val labelIncome = mutable.ArrayBuffer[Float]()
      val labelMarital = mutable.ArrayBuffer[Float]()
      val featureBuffers = featureNames.map(name => name -> mutable.ArrayBuffer[Float]()).toMap

      for (i <- 0 until n) {
        val inPart = part match {
          case "train" => trainIdx.contains(i)
          case "valid" => validIdx.contains(i)
          case _ => !trainIdx.contains(i) && !validIdx.contains(i)
        }
        if (inPart) {
          featureNames.foreach { name =>
            featureBuffers(name) += toDouble(df.col(name)(i)).toFloat
          }
          labelIncome += toDouble(df.col("income_label")(i)).toFloat
          labelMarital += toDouble(df.col("marital_label")(i)).toFloat
        }
      }

      val sparse = featureNames.map { name =>
        name -> tensor(featureBuffers(name).toArray, Array(featureBuffers(name).length.toLong)).toType(ScalarType.Long)
      }.toMap

      val taskLabels = Map(
        "income" -> tensor(labelIncome.toArray, Array(labelIncome.length.toLong)),
        "marital" -> tensor(labelMarital.toArray, Array(labelMarital.length.toLong))
      )

      new MultiTaskDataset(sparse, taskLabels)
    }

    (build("train"), build("valid"), build("test"))
  }

  private def toDouble(v: Any): Double = v match {
    case f: Float => f.toDouble
    case d: Double => d
    case i: Int => i.toDouble
    case l: Long => l.toDouble
    case s: String => s.toDoubleOption.getOrElse(0.0)
    case _ => 0.0
  }
}

