package benchmarks

import torchrec.basic.features._
import torchrec.basic.metrics._
import torchrec.data._
import torchrec.models.ranking._
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
 * AliCCP CTR Benchmark using real data
 */
object AliCCPBenchmark {

  // Dense features from the dataset
  private val DenseCols = List("D109_14", "D110_14", "D127_14", "D150_14", "D508", "D509", "D702", "D853")

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("AliCCP CTR Benchmark (WideDeep, DeepFM, DCN)")
    println("=" * 80)

    val results = mutable.ListBuffer[AliCCPResult]()

    val datasetPath = "/home/muller/IdeaProjects/torch-rechub/examples/ranking/data/ali-ccp"

    // Run models
    results += runWideDeepAliCCP(datasetPath)
    System.gc()
    results += runDeepFMAliCCP(datasetPath)
    System.gc()
    results += runDCNAliCCP(datasetPath)
    System.gc()

    printResults(results.toList)
  }

  def loadAliCCPData(dataPath: String): (List[SparseFeature], List[DenseFeature], torchrec.data.TensorDataset, torchrec.data.TensorDataset, torchrec.data.TensorDataset) = {
    println(s"Loading AliCCP data from: $dataPath")

    val trainFile = new File(dataPath, "ali_ccp_train_sample.csv")
    val valFile = new File(dataPath, "ali_ccp_val_sample.csv")
    val testFile = new File(dataPath, "ali_ccp_test_sample.csv")

    require(trainFile.exists(), s"Train file not found: $trainFile")

    // Parse header
    val header = scala.io.Source.fromFile(trainFile).getLines().next().split(",").toList

    // Identify sparse and dense columns
    val sparseColNames = header.filterNot(DenseCols.contains).filterNot(List("click", "purchase").contains)
    val denseColNames = DenseCols.filter(header.contains)

    println(s"  Sparse features: ${sparseColNames.size}, Dense features: ${denseColNames.size}")

    // Parse all rows
    def parseRows(file: File): Array[(Array[Float], Map[String, Float], Map[String, Float])] = {
      val lines = scala.io.Source.fromFile(file).getLines().drop(1)  // Skip header
      val colNameToIdx = header.zipWithIndex.toMap
      val labelIndices = Set(colNameToIdx("click"), colNameToIdx("purchase"))
      lines.map { line =>
        val fields = line.split(",")
        val labels = Array(fields(0).toFloat, fields(1).toFloat)
        val sparseValues = mutable.Map[String, Float]()
        val denseValues = mutable.Map[String, Float]()
        header.zipWithIndex.foreach { case (colName, i) =>
          if (!labelIndices.contains(i)) {
            val value = fields(i).toFloat
            if (DenseCols.contains(colName)) {
              denseValues(colName) = value
            } else {
              sparseValues(colName) = value
            }
          }
        }
        (labels, sparseValues.toMap, denseValues.toMap)
      }.toArray
    }

    val trainData = parseRows(trainFile)
    val valData = parseRows(valFile)
    val testData = parseRows(testFile)

    println(s"  Train: ${trainData.size}, Val: ${valData.size}, Test: ${testData.size}")

    // Calculate vocab sizes - must consider ALL datasets (train, val, test) to avoid index out of bounds
    val vocabSizes = mutable.Map[String, Int]()
    sparseColNames.foreach { col =>
      val allData = trainData ++ valData ++ testData
      val maxVal = allData.map(_._2(col).toInt).max
      vocabSizes(col) = maxVal + 1
    }

    // Build features
    val sparseFeatures = sparseColNames.map { col =>
      SparseFeature(col, vocabSizes(col), 16)
    }

    val denseFeatures = denseColNames.map { col =>
      DenseFeature(col, 1)
    }

    // Build datasets
    def buildDS(data: Array[(Array[Float], Map[String, Float], Map[String, Float])]): torchrec.data.TensorDataset = {
      val n = data.size
      val sparseTensors = sparseColNames.map { col =>
        val indices = data.map(_._2(col).toLong.toFloat)
        col -> tensor(indices.toArray, Array(n.toLong)).toType(ScalarType.Long)
      }.toMap

      val denseTensors = denseColNames.map { col =>
        val values = data.map(_._3(col))
        col -> tensor(values.toArray, Array(n.toLong))
      }.toMap

      val labels = data.map(_._1(0))  // click
      val labelTensor = tensor(labels.toArray[Float], Array(n.toLong))

      new torchrec.data.TensorDataset(sparseTensors, denseTensors, Some(labelTensor))
    }

    val trainDS = buildDS(trainData)
    val valDS = buildDS(valData)
    val testDS = buildDS(testData)

    (sparseFeatures, denseFeatures, trainDS, valDS, testDS)
  }

  def runWideDeepAliCCP(datasetPath: String): AliCCPResult = {
    println("\n--- WideDeep AliCCP Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (sparseFeatures, denseFeatures, trainDS, valDS, testDS) = loadAliCCPData(datasetPath)

      val trainLoader = new DataLoader(trainDS, 2048, shuffle = true)
      val valLoader = new DataLoader(valDS, 2048, shuffle = false)
      val testLoader = new DataLoader(testDS, 2048, shuffle = false)

      val model = new WideDeep(
        features = sparseFeatures,
        embedDim = 16,
        mlpDims = List(256L, 128L),
        dropout = 0.2f,
        device = device
      )

      println("  [Model] WideDeep created")

      val trainer = new CTRTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader, Some(valLoader))

      val metrics = trainer.evaluate(testLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)
      println(f"  [PASS] WideDeep: AUC=$auc%.4f")

      AliCCPResult("WideDeep", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] WideDeep: ${e.getMessage}")
        e.printStackTrace()
        AliCCPResult("WideDeep", Map("error" -> 0.0f))
    }
  }

  def runDeepFMAliCCP(datasetPath: String): AliCCPResult = {
    println("\n--- DeepFM AliCCP Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (sparseFeatures, denseFeatures, trainDS, valDS, testDS) = loadAliCCPData(datasetPath)

      val trainLoader = new DataLoader(trainDS, 2048, shuffle = true)
      val valLoader = new DataLoader(valDS, 2048, shuffle = false)
      val testLoader = new DataLoader(testDS, 2048, shuffle = false)

      val halfIdx = sparseFeatures.size / 2
      val model = new DeepFM(
        deepFeatures = sparseFeatures.take(halfIdx),
        fmFeatures = sparseFeatures.drop(halfIdx),
        embedDim = 16,
        mlpDims = List(256L, 128L),
        dropout = 0.2f,
        device = device
      )

      println("  [Model] DeepFM created")

      val trainer = new CTRTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader, Some(valLoader))

      val metrics = trainer.evaluate(testLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)
      println(f"  [PASS] DeepFM: AUC=$auc%.4f")

      AliCCPResult("DeepFM", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] DeepFM: ${e.getMessage}")
        e.printStackTrace()
        AliCCPResult("DeepFM", Map("error" -> 0.0f))
    }
  }

  def runDCNAliCCP(datasetPath: String): AliCCPResult = {
    println("\n--- DCN AliCCP Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (sparseFeatures, denseFeatures, trainDS, valDS, testDS) = loadAliCCPData(datasetPath)

      val trainLoader = new DataLoader(trainDS, 2048, shuffle = true)
      val valLoader = new DataLoader(valDS, 2048, shuffle = false)
      val testLoader = new DataLoader(testDS, 2048, shuffle = false)

      val allFeatures: List[Feature] = denseFeatures.map(_.asInstanceOf[Feature]) ++ sparseFeatures.map(_.asInstanceOf[Feature])

      val model = new DCN(
        features = allFeatures,
        embedDim = 16,
        numCrossLayers = 3,
        mlpDims = List(256L, 128L),
        dropout = 0.2f,
        device = device
      )

      println("  [Model] DCN created")

      val trainer = new CTRTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader, Some(valLoader))

      val metrics = trainer.evaluate(testLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)
      println(f"  [PASS] DCN: AUC=$auc%.4f")

      AliCCPResult("DCN", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] DCN: ${e.getMessage}")
        e.printStackTrace()
        AliCCPResult("DCN", Map("error" -> 0.0f))
    }
  }

  def printResults(results: List[AliCCPResult]): Unit = {
    println("\n" + "=" * 80)
    println("AliCCP Benchmark Results Summary")
    println("=" * 80)

    results.foreach { r =>
      val metricStr = r.metrics.map { case (k, v) => f"$k=$v%.4f" }.mkString(", ")
      val status = if (r.metrics.contains("error")) "[FAIL]" else "[PASS]"
      println(f"$status ${r.model}%-15s $metricStr")
    }
    println("=" * 80)
  }
}

case class AliCCPResult(
  model: String,
  metrics: Map[String, Float]
)
