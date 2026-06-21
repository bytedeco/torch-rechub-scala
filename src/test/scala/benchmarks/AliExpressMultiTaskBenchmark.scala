package benchmarks

import torchrec.basic.features._
import torchrec.basic.metrics._
import torchrec.data._
import torchrec.models.multi_task._
import torchrec.trainers._
import torchrec.utils.DeviceSupport
import torchrec.Implicits.tensor
import torchrec.Implicits._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.util.Random
import scala.collection.mutable
import java.io.File

/**
 * AliExpress Multi-Task CTR Benchmark
 *
 * Mirrors the Python implementation in:
 * /home/muller/IdeaProjects/torch-rechub/examples/ranking/run_aliexpress.py
 *
 * Supports models: SharedBottom, MMOE, PLE, AITM
 */
object AliExpressMultiTaskBenchmark {

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("AliExpress Multi-Task CTR Benchmark")
    println("=" * 80)

    val results = mutable.ListBuffer[Result]()
    val datasetPath = "/home/muller/IdeaProjects/torch-rechub/examples/ranking/data/aliexpress"

    results += runSharedBottom(datasetPath)
    System.gc()
    results += runMMOE(datasetPath)
    System.gc()
    results += runPLE(datasetPath)
    System.gc()
    results += runAITM(datasetPath)
    System.gc()

    printResults(results.toList)
  }

  class MultiTaskDataset(
    val sparseFeatures: Map[String, Tensor],
    val denseFeatures: Map[String, Tensor],
    val taskLabels: Map[String, Tensor]
  ) extends Dataset {
    override def size: Long = {
      sparseFeatures.headOption.map(_._2.size(0)).getOrElse(0)
    }

    override def get(index: Long): Batch = {
      val safeIndex = index.min(size - 1).max(0)
      def getFeature(v: Tensor): Tensor = {
        val sliced = v.narrow(0, safeIndex, 1)
        val result = if (sliced.dim() == 0) sliced.unsqueeze(0) else sliced
        result.contiguous().clone()
      }
      Batch(
        sparseFeatures.map { case (k, v) => k -> getFeature(v) },
        denseFeatures.map { case (k, v) => k -> getFeature(v) },
        Map.empty,
        labels = None,
        taskLabels = Some(taskLabels.map { case (k, v) => k -> getFeature(v) })
      )
    }
  }

  def loadAliExpressData(dataPath: String): (List[Feature], Map[String, Int], MultiTaskDataset, MultiTaskDataset) = {
    println(s"Loading AliExpress data from: $dataPath")

    val trainFile = new File(dataPath, "aliexpress_train_sample.csv")
    val testFile = new File(dataPath, "aliexpress_test_sample.csv")

    require(trainFile.exists(), s"Train file not found: $trainFile")

    val header = scala.io.Source.fromFile(trainFile).getLines().next().split(",").toList

    val sparseColNames = header.filter(_.startsWith("categorical"))
    val denseColNames = header.filter(_.startsWith("numerical"))

    println(s"  Sparse features: ${sparseColNames.size}, Dense features: ${denseColNames.size}")

    def parseRows(file: File): Array[(Array[Float], Map[String, Float], Map[String, Float])] = {
      if (!file.exists()) return Array.empty
      val lines = scala.io.Source.fromFile(file).getLines().drop(1)
      val colNameToIdx = header.zipWithIndex.toMap
      val clickIdx = colNameToIdx("click")
      val conversionIdx = colNameToIdx("conversion")
      lines.map { line =>
        val fields = line.split(",")
        val click = fields(clickIdx).toFloat
        val conversion = fields(conversionIdx).toFloat
        val sparseValues = mutable.Map[String, Float]()
        val denseValues = mutable.Map[String, Float]()
        header.zipWithIndex.foreach { case (colName, i) =>
          if (colName.startsWith("categorical")) {
            sparseValues(colName) = fields(i).toFloat
          } else if (colName.startsWith("numerical")) {
            denseValues(colName) = fields(i).toFloat
          }
        }
        (Array(click, conversion), sparseValues.toMap, denseValues.toMap)
      }.toArray
    }

    val trainData = parseRows(trainFile)
    val testData = parseRows(testFile)

    println(s"  Train: ${trainData.size}, Test: ${testData.size}")

    val allData = trainData ++ testData
    val vocabSizes = mutable.Map[String, Int]()
    sparseColNames.foreach { col =>
      val maxVal = allData.map(_._2(col).toInt).max
      vocabSizes(col) = maxVal + 1
    }

    val sparseFeatures = sparseColNames.map { col =>
      SparseFeature(col, vocabSizes(col), 5)
    }
    val denseFeatures = denseColNames.map { col =>
      DenseFeature(col, 1)
    }
    // Current Scala multi-task models/trainer consume sparse feature maps only.
    // Keep dense tensors in dataset for future use, but avoid adding them to model feature list,
    // otherwise inputDims is overestimated and first Linear layer shape mismatches at runtime.
    val allFeatures: List[Feature] = sparseFeatures.map(_.asInstanceOf[Feature])

    val sparseDim = sparseFeatures.map(_.embedDim).sum
    val totalDim = sparseDim + denseFeatures.size
    println(s"  Feature dims: sparse=$sparseDim, total=$totalDim")

    def buildDS(data: Array[(Array[Float], Map[String, Float], Map[String, Float])]): MultiTaskDataset = {
      val n = data.size
      val sparseTensors = sparseColNames.map { col =>
        val indices = data.map(_._2(col).toLong.toFloat)
        col -> tensor(indices.toArray, Array(n.toLong)).toType(ScalarType.Long)
      }.toMap

      val denseTensors = denseColNames.map { col =>
        val values = data.map(_._3(col))
        col -> tensor(values.toArray, Array(n.toLong))
      }.toMap

      val clickLabels = data.map(_._1(0))
      val conversionLabels = data.map(_._1(1))

      val clickTensor = tensor(clickLabels.toArray, Array(n.toLong))
      val conversionTensor = tensor(conversionLabels.toArray, Array(n.toLong))

      val taskLabels = Map("click" -> clickTensor, "conversion" -> conversionTensor)
      new MultiTaskDataset(sparseTensors, denseTensors, taskLabels)
    }

    val trainDS = buildDS(trainData)
    val testDS = buildDS(testData)

    (allFeatures, vocabSizes.toMap, trainDS, testDS)
  }

  def runSharedBottom(datasetPath: String): Result = {
    println("\n--- SharedBottom AliExpress Multi-Task Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (features, _, trainDS, testDS) = loadAliExpressData(datasetPath)

      val trainLoader = new DataLoader(trainDS, 1024, shuffle = true)
      val valLoader = new DataLoader(testDS, 1024, shuffle = false)
      val testLoader = new DataLoader(testDS, 1024, shuffle = false)

      val taskNames = List("conversion", "click")

      val bottomParams = Map("dims" -> List(192L, 96L, 48L), "activation" -> "relu", "dropout" -> 0.0f)
      val towerParamsList = List(Map("dims" -> List(8L)), Map("dims" -> List(8L)))

      val model = new SharedBottom(
        features = features,
        taskTypes = List("classification", "classification"),
        bottomParams = bottomParams,
        towerParamsList = towerParamsList,
        device = device
      )

      println("  [Model] SharedBottom created")

      val trainer = new MTLTrainer(
        model, taskNames, 0.001f, 1e-5f, device = device,
        numEpochs = 300, earlyStopPatience = 300, verbose = true
      )
      trainer.fit(trainLoader, Some(valLoader))

      val metrics = trainer.evaluate(testLoader)
      val convAuc = metrics.getOrElse("conversion_AUC", 0.0f)
      val clickAuc = metrics.getOrElse("click_AUC", 0.0f)
      println(f"  [PASS] SharedBottom: conversion=$convAuc%.4f, click=$clickAuc%.4f")

      Result("SharedBottom", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] SharedBottom: ${e.getMessage}")
        e.printStackTrace()
        Result("SharedBottom", Map("error" -> 0.0f))
    }
  }

  def runMMOE(datasetPath: String): Result = {
    println("\n--- MMOE AliExpress Multi-Task Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (features, _, trainDS, testDS) = loadAliExpressData(datasetPath)

      val trainLoader = new DataLoader(trainDS, 1024, shuffle = true)
      val valLoader = new DataLoader(testDS, 1024, shuffle = false)
      val testLoader = new DataLoader(testDS, 1024, shuffle = false)

      val taskNames = List("conversion", "click")

      val expertParams = Map("dims" -> List(64L, 32L, 16L), "activation" -> "relu", "dropout" -> 0.0f)
      val towerParamsList = List(Map("dims" -> List(8L)), Map("dims" -> List(8L)))

      val model = new MMOE(
        features = features,
        taskTypes = List("classification", "classification"),
        nExpert = 3,
        expertParams = expertParams,
        towerParamsList = towerParamsList,
        device = device
      )

      println("  [Model] MMOE created")

      val trainer = new MTLTrainer(
        model, taskNames, 0.001f, 1e-5f, device = device,
        numEpochs = 300, earlyStopPatience = 300, verbose = true
      )
      trainer.fit(trainLoader, Some(valLoader))

      val metrics = trainer.evaluate(testLoader)
      val convAuc = metrics.getOrElse("conversion_AUC", 0.0f)
      val clickAuc = metrics.getOrElse("click_AUC", 0.0f)
      println(f"  [PASS] MMOE: conversion=$convAuc%.4f, click=$clickAuc%.4f")

      Result("MMOE", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] MMOE: ${e.getMessage}")
        e.printStackTrace()
        Result("MMOE", Map("error" -> 0.0f))
    }
  }

  def runPLE(datasetPath: String): Result = {
    println("\n--- PLE AliExpress Multi-Task Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (features, _, trainDS, testDS) = loadAliExpressData(datasetPath)

      val trainLoader = new DataLoader(trainDS, 1024, shuffle = true)
      val valLoader = new DataLoader(testDS, 1024, shuffle = false)
      val testLoader = new DataLoader(testDS, 1024, shuffle = false)

      val taskNames = List("conversion", "click")

      val expertParams = Map("dims" -> List(64L, 32L, 16L), "activation" -> "relu", "dropout" -> 0.0f)
      val towerParamsList = List(Map("dims" -> List(8L)), Map("dims" -> List(8L)))

      val model = new PLE(
        features = features,
        taskTypes = List("classification", "classification"),
        nLevel = 1,
        nExpertSpecific = 1,
        nExpertShared = 1,
        expertParams = expertParams,
        towerParamsList = towerParamsList,
        device = device
      )

      println("  [Model] PLE created")

      val trainer = new MTLTrainer(
        model, taskNames, 0.001f, 1e-5f, device = device,
        numEpochs = 300, earlyStopPatience = 300, verbose = true
      )
      trainer.fit(trainLoader, Some(valLoader))

      val metrics = trainer.evaluate(testLoader)
      val convAuc = metrics.getOrElse("conversion_AUC", 0.0f)
      val clickAuc = metrics.getOrElse("click_AUC", 0.0f)
      println(f"  [PASS] PLE: conversion=$convAuc%.4f, click=$clickAuc%.4f")

      Result("PLE", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] PLE: ${e.getMessage}")
        e.printStackTrace()
        Result("PLE", Map("error" -> 0.0f))
    }
  }

  def runAITM(datasetPath: String): Result = {
    println("\n--- AITM AliExpress Multi-Task Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (features, _, trainDS, testDS) = loadAliExpressData(datasetPath)

      val trainLoader = new DataLoader(trainDS, 1024, shuffle = true)
      val valLoader = new DataLoader(testDS, 1024, shuffle = false)
      val testLoader = new DataLoader(testDS, 1024, shuffle = false)

      val taskNames = List("conversion", "click")

      val bottomParams = Map("dims" -> List(128L, 64L, 32L), "activation" -> "relu", "dropout" -> 0.0f)
      val towerParamsList = List(Map("dims" -> List(8L)), Map("dims" -> List(8L)))

      val model = new AITM(
        features = features,
        nTask = 2,
        bottomParams = bottomParams,
        towerParamsList = towerParamsList,
        device = device
      )

      println("  [Model] AITM created")

      val trainer = new MTLTrainer(
        model, taskNames, 0.001f, 1e-5f, device = device,
        numEpochs = 300, earlyStopPatience = 300, verbose = true
      )
      trainer.fit(trainLoader, Some(valLoader))

      val metrics = trainer.evaluate(testLoader)
      val convAuc = metrics.getOrElse("conversion_AUC", 0.0f)
      val clickAuc = metrics.getOrElse("click_AUC", 0.0f)
      println(f"  [PASS] AITM: conversion=$convAuc%.4f, click=$clickAuc%.4f")

      Result("AITM", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] AITM: ${e.getMessage}")
        e.printStackTrace()
        Result("AITM", Map("error" -> 0.0f))
    }
  }

  def printResults(results: List[Result]): Unit = {
    println("\n" + "=" * 80)
    println("AliExpress Multi-Task Benchmark Results Summary")
    println("=" * 80)
    println("%-15s %12s %12s".format("Model", "Conversion", "Click"))
    println("-" * 80)

    results.foreach { r =>
      val convAuc = r.metrics.getOrElse("conversion_AUC", 0.0f)
      val clickAuc = r.metrics.getOrElse("click_AUC", 0.0f)
      val status = if (r.metrics.contains("error")) "[FAIL]" else "[PASS]"
      println("%s %-13s %12.4f %12.4f".format(status, r.model, convAuc, clickAuc))
    }
    println("=" * 80)
  }

  case class Result(model: String, metrics: Map[String, Float])
}
