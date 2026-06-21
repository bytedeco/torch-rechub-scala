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
 * AliCCP Multi-Task CTR Benchmark
 *
 * Mirrors the Python implementation in:
 * /home/muller/IdeaProjects/torch-rechub/examples/ranking/run_ali_ccp_multi_task.py
 *
 * Supports models: SharedBottom, ESMM, MMOE, PLE, AITM
 *
 * Task definitions:
 * - task 1 (cvr): main task, purchase prediction
 * - task 2 (ctr): auxiliary task, click prediction
 */
object AliCCPMultiTaskBenchmark {

  // Dense features from the dataset
  private val DenseCols = List("D109_14", "D110_14", "D127_14", "D150_14", "D508", "D509", "D702", "D853")

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("AliCCP Multi-Task CTR Benchmark")
    println("=" * 80)

    val results = mutable.ListBuffer[MultiTaskResult]()
    val datasetPath = "/home/muller/IdeaProjects/torch-rechub/examples/ranking/data/ali-ccp"

    // Run all multi-task models
    results += runSharedBottom(datasetPath)
    System.gc()
    results += runESMM(datasetPath)
    System.gc()
    results += runMMOE(datasetPath)
    System.gc()
    results += runPLE(datasetPath)
    System.gc()
    results += runAITM(datasetPath)
    System.gc()

    printResults(results.toList)
  }

  /**
   * MultiTaskDataset - Dataset that properly supports multiple task labels
   */
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

  /**
   * Load AliCCP data and return features, datasets, and vocab sizes
   * Note: vocab sizes are calculated from ALL data (train + val + test) to avoid index out of bounds
   */
  def loadAliCCPData(dataPath: String): (List[SparseFeature], Map[String, Int], MultiTaskDataset, MultiTaskDataset, MultiTaskDataset) = {
    println(s"Loading AliCCP data from: $dataPath")

    val trainFile = new File(dataPath, "ali_ccp_train_sample.csv")
    val valFile = new File(dataPath, "ali_ccp_val_sample.csv")
    val testFile = new File(dataPath, "ali_ccp_test_sample.csv")

    require(trainFile.exists(), s"Train file not found: $trainFile")

    // Parse header
    val header = scala.io.Source.fromFile(trainFile).getLines().next().split(",").toList

    // Identify sparse and dense columns (exclude labels)
    val labelCols = Set("click", "purchase")
    val sparseColNames = header.filterNot(DenseCols.contains).filterNot(labelCols.contains)
    val denseColNames = DenseCols.filter(header.contains)

    println(s"  Sparse features: ${sparseColNames.size}, Dense features: ${denseColNames.size}")

    // Parse all rows from all datasets
    // Return type: (labels = [click, purchase], sparseFeatures, denseFeatures)
    def parseRows(file: File): Array[(Array[Float], Map[String, Float], Map[String, Float])] = {
      if (!file.exists()) return Array.empty
      val lines = scala.io.Source.fromFile(file).getLines().drop(1)  // Skip header
      val colNameToIdx = header.zipWithIndex.toMap
      val clickIdx = colNameToIdx("click")
      val purchaseIdx = colNameToIdx("purchase")
      lines.map { line =>
        val fields = line.split(",")
        val click = fields(clickIdx).toFloat
        val purchase = fields(purchaseIdx).toFloat
        val sparseValues = mutable.Map[String, Float]()
        val denseValues = mutable.Map[String, Float]()
        header.zipWithIndex.foreach { case (colName, i) =>
          if (colName != "click" && colName != "purchase") {
            val value = fields(i).toFloat
            if (DenseCols.contains(colName)) {
              denseValues(colName) = value
            } else {
              sparseValues(colName) = value
            }
          }
        }
        (Array(click, purchase), sparseValues.toMap, denseValues.toMap)
      }.toArray
    }

    val trainData = parseRows(trainFile)
    val valData = parseRows(valFile)
    val testData = parseRows(testFile)

    println(s"  Train: ${trainData.size}, Val: ${valData.size}, Test: ${testData.size}")

    // Calculate vocab sizes from ALL data to avoid index out of bounds
    val allData = trainData ++ valData ++ testData
    val vocabSizes = mutable.Map[String, Int]()
    sparseColNames.foreach { col =>
      val maxVal = allData.map(_._2(col).toInt).max
      vocabSizes(col) = maxVal + 1
      println(s"    $col: vocabSize=${vocabSizes(col)}, maxVal=$maxVal")
    }

    // Build SPARSE features only (matching Python behavior)
    val sparseFeatures = sparseColNames.map { col =>
      SparseFeature(col, vocabSizes(col), 4)  // embed_dim=4 to match Python
    }

    // Build MultiTaskDataset with only sparse features
    def buildMultiTaskDS(data: Array[(Array[Float], Map[String, Float], Map[String, Float])]): MultiTaskDataset = {
      val n = data.size
      val sparseTensors = sparseColNames.map { col =>
        val indices = data.map(_._2(col).toLong.toFloat)
        col -> tensor(indices.toArray, Array(n.toLong)).toType(ScalarType.Long)
      }.toMap

      // Dense tensors are empty to match Python behavior
      val denseTensors: Map[String, Tensor] = Map.empty

      // Task labels: cvr (purchase), ctr (click), ctcvr (cvr * ctr)
      val cvrLabels = data.map(_._1(1))  // purchase -> cvr
      val ctrLabels = data.map(_._1(0))  // click -> ctr
      val ctcvrLabels = data.map(r => r._1(0) * r._1(1))  // click * purchase -> ctcvr

      val cvrTensor = tensor(cvrLabels.toArray, Array(n.toLong))
      val ctrTensor = tensor(ctrLabels.toArray, Array(n.toLong))
      val ctcvrTensor = tensor(ctcvrLabels.toArray, Array(n.toLong))

      // Include ctcvr for ESMM compatibility
      val taskLabels = Map(
        "cvr" -> cvrTensor,
        "ctr" -> ctrTensor,
        "ctcvr" -> ctcvrTensor
      )

      new MultiTaskDataset(sparseTensors, denseTensors, taskLabels)
    }

    val trainDS = buildMultiTaskDS(trainData)
    val valDS = buildMultiTaskDS(valData)
    val testDS = buildMultiTaskDS(testData)

    (sparseFeatures, vocabSizes.toMap, trainDS, valDS, testDS)
  }

  /**
   * SharedBottom Model
   */
  def runSharedBottom(datasetPath: String): MultiTaskResult = {
    println("\n--- SharedBottom AliCCP Multi-Task Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (sparseFeatures, vocabSizes, trainDS, valDS, testDS) = loadAliCCPData(datasetPath)

      // Calculate actual feature dims (sparse only)
      val sparseDim = sparseFeatures.map(_.embedDim).sum
      println(s"  Sparse feature dims: $sparseDim")

      val trainLoader = new DataLoader(trainDS, 1024, shuffle = true)
      val valLoader = new DataLoader(valDS, 1024, shuffle = false)
      val testLoader = new DataLoader(testDS, 1024, shuffle = false)

      val taskNames = List("cvr", "ctr")

      // Use actual sparse feature dims
      val bottomParams = Map("dims" -> List(sparseDim.toLong), "activation" -> "relu", "dropout" -> 0.0f)
      val towerParamsList = List(Map("dims" -> List(8L)), Map("dims" -> List(8L)))

      val model = new SharedBottom(
        features = sparseFeatures.map(_.asInstanceOf[Feature]),
        taskTypes = List("classification", "classification"),
        bottomParams = bottomParams,
        towerParamsList = towerParamsList,
        device = device
      )

      println("  [Model] SharedBottom created")

      val trainer = new MTLTrainer(
        model, taskNames, 0.001f, 1e-4f, device = device,
        numEpochs = 300, earlyStopPatience = 300, verbose = true
      )
      trainer.fit(trainLoader, Some(valLoader))

      val metrics = trainer.evaluate(testLoader)
      val cvrAuc = metrics.getOrElse("cvr_AUC", 0.0f)
      val ctrAuc = metrics.getOrElse("ctr_AUC", 0.0f)
      println(f"  [PASS] SharedBottom: cvr_auc=$cvrAuc%.4f, ctr_auc=$ctrAuc%.4f")

      MultiTaskResult("SharedBottom", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] SharedBottom: ${e.getMessage}")
        e.printStackTrace()
        MultiTaskResult("SharedBottom", Map("error" -> 0.0f))
    }
  }

  /**
   * ESMM (Entire Space Multi-Task Model)
   */
  def runESMM(datasetPath: String): MultiTaskResult = {
    println("\n--- ESMM AliCCP Multi-Task Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (sparseFeatures, vocabSizes, trainDS, valDS, testDS) = loadAliCCPData(datasetPath)

      val sparseDim = sparseFeatures.map(_.embedDim).sum
      println(s"  Sparse feature dims: $sparseDim")

      val trainLoader = new DataLoader(trainDS, 1024, shuffle = true)
      val valLoader = new DataLoader(valDS, 1024, shuffle = false)
      val testLoader = new DataLoader(testDS, 1024, shuffle = false)

      // ESMM has 3 tasks: cvr, ctr, ctcvr
      val taskNames = List("cvr", "ctr", "ctcvr")

      val model = ESMM(
        features = sparseFeatures.map(_.asInstanceOf[Feature]),
        taskNames = taskNames,
        embedDim = 4,
        towerDims = List(16L, 8L),
        dropout = 0.0f,
        device = device
      )

      println("  [Model] ESMM created")

      val trainer = new MTLTrainer(
        model, taskNames, 0.001f, 1e-4f, device = device,
        numEpochs = 300, earlyStopPatience = 300, verbose = true
      )
      trainer.fit(trainLoader, Some(valLoader))

      val metrics = trainer.evaluate(testLoader)
      val cvrAuc = metrics.getOrElse("cvr_AUC", 0.0f)
      val ctrAuc = metrics.getOrElse("ctr_AUC", 0.0f)
      val ctcvrAuc = metrics.getOrElse("ctcvr_AUC", 0.0f)
      println(f"  [PASS] ESMM: cvr_auc=$cvrAuc%.4f, ctr_auc=$ctrAuc%.4f, ctcvr_auc=$ctcvrAuc%.4f")

      MultiTaskResult("ESMM", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] ESMM: ${e.getMessage}")
        e.printStackTrace()
        MultiTaskResult("ESMM", Map("error" -> 0.0f))
    }
  }

  /**
   * MMOE (Multi-gate Mixture of Experts)
   */
  def runMMOE(datasetPath: String): MultiTaskResult = {
    println("\n--- MMOE AliCCP Multi-Task Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (sparseFeatures, vocabSizes, trainDS, valDS, testDS) = loadAliCCPData(datasetPath)

      val sparseDim = sparseFeatures.map(_.embedDim).sum
      println(s"  Sparse feature dims: $sparseDim")

      val trainLoader = new DataLoader(trainDS, 1024, shuffle = true)
      val valLoader = new DataLoader(valDS, 1024, shuffle = false)
      val testLoader = new DataLoader(testDS, 1024, shuffle = false)

      val taskNames = List("cvr", "ctr")

      // Use actual sparse feature dims
      val expertParams = Map("dims" -> List(sparseDim.toLong), "activation" -> "relu", "dropout" -> 0.0f)
      val towerParamsList = List(Map("dims" -> List(8L)), Map("dims" -> List(8L)))

      val model = new MMOE(
        features = sparseFeatures.map(_.asInstanceOf[Feature]),
        taskTypes = List("classification", "classification"),
        nExpert = 8,
        expertParams = expertParams,
        towerParamsList = towerParamsList,
        device = device
      )

      println("  [Model] MMOE created")

      val trainer = new MTLTrainer(
        model, taskNames, 0.001f, 1e-4f, device = device,
        numEpochs = 300, earlyStopPatience = 300, verbose = true
      )
      trainer.fit(trainLoader, Some(valLoader))

      val metrics = trainer.evaluate(testLoader)
      val cvrAuc = metrics.getOrElse("cvr_AUC", 0.0f)
      val ctrAuc = metrics.getOrElse("ctr_AUC", 0.0f)
      println(f"  [PASS] MMOE: cvr_auc=$cvrAuc%.4f, ctr_auc=$ctrAuc%.4f")

      MultiTaskResult("MMOE", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] MMOE: ${e.getMessage}")
        e.printStackTrace()
        MultiTaskResult("MMOE", Map("error" -> 0.0f))
    }
  }

  /**
   * PLE (Progressive Layered Extraction)
   */
  def runPLE(datasetPath: String): MultiTaskResult = {
    println("\n--- PLE AliCCP Multi-Task Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (sparseFeatures, vocabSizes, trainDS, valDS, testDS) = loadAliCCPData(datasetPath)

      val sparseDim = sparseFeatures.map(_.embedDim).sum
      println(s"  Sparse feature dims: $sparseDim")

      val trainLoader = new DataLoader(trainDS, 1024, shuffle = true)
      val valLoader = new DataLoader(valDS, 1024, shuffle = false)
      val testLoader = new DataLoader(testDS, 1024, shuffle = false)

      val taskNames = List("cvr", "ctr")

      // Use actual sparse feature dims
      val expertParams = Map("dims" -> List(sparseDim.toLong), "activation" -> "relu", "dropout" -> 0.0f)
      val towerParamsList = List(Map("dims" -> List(8L)), Map("dims" -> List(8L)))

      val model = new PLE(
        features = sparseFeatures.map(_.asInstanceOf[Feature]),
        taskTypes = List("classification", "classification"),
        nLevel = 1,
        nExpertSpecific = 2,
        nExpertShared = 1,
        expertParams = expertParams,
        towerParamsList = towerParamsList,
        device = device
      )

      println("  [Model] PLE created")

      val trainer = new MTLTrainer(
        model, taskNames, 0.001f, 1e-4f, device = device,
        numEpochs = 300, earlyStopPatience = 300, verbose = true
      )
      trainer.fit(trainLoader, Some(valLoader))

      val metrics = trainer.evaluate(testLoader)
      val cvrAuc = metrics.getOrElse("cvr_AUC", 0.0f)
      val ctrAuc = metrics.getOrElse("ctr_AUC", 0.0f)
      println(f"  [PASS] PLE: cvr_auc=$cvrAuc%.4f, ctr_auc=$ctrAuc%.4f")

      MultiTaskResult("PLE", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] PLE: ${e.getMessage}")
        e.printStackTrace()
        MultiTaskResult("PLE", Map("error" -> 0.0f))
    }
  }

  /**
   * AITM (Adversarial Induced Multi-task Learning)
   */
  def runAITM(datasetPath: String): MultiTaskResult = {
    println("\n--- AITM AliCCP Multi-Task Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (sparseFeatures, vocabSizes, trainDS, valDS, testDS) = loadAliCCPData(datasetPath)

      val sparseDim = sparseFeatures.map(_.embedDim).sum
      println(s"  Sparse feature dims: $sparseDim")

      val trainLoader = new DataLoader(trainDS, 1024, shuffle = true)
      val valLoader = new DataLoader(valDS, 1024, shuffle = false)
      val testLoader = new DataLoader(testDS, 1024, shuffle = false)

      val taskNames = List("cvr", "ctr")

      // Use actual sparse feature dims for bottom input
      val model = new AITM(
        features = sparseFeatures.map(_.asInstanceOf[Feature]),
        nTask = 2,
        bottomParams = Map("dims" -> List(sparseDim.toLong, 32L, 16L), "activation" -> "relu", "dropout" -> 0.0f),
        towerParamsList = List(Map("dims" -> List(8L)), Map("dims" -> List(8L))),
        device = device
      )

      println("  [Model] AITM created")

      val trainer = new MTLTrainer(
        model, taskNames, 0.001f, 1e-4f, device = device,
        numEpochs = 300, earlyStopPatience = 300, verbose = true
      )
      trainer.fit(trainLoader, Some(valLoader))

      val metrics = trainer.evaluate(testLoader)
      val cvrAuc = metrics.getOrElse("cvr_AUC", 0.0f)
      val ctrAuc = metrics.getOrElse("ctr_AUC", 0.0f)
      println(f"  [PASS] AITM: cvr_auc=$cvrAuc%.4f, ctr_auc=$ctrAuc%.4f")

      MultiTaskResult("AITM", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] AITM: ${e.getMessage}")
        e.printStackTrace()
        MultiTaskResult("AITM", Map("error" -> 0.0f))
    }
  }

  def printResults(results: List[MultiTaskResult]): Unit = {
    println("\n" + "=" * 80)
    println("AliCCP Multi-Task Benchmark Results Summary")
    println("=" * 80)
    println("%-15s %12s %12s".format("Model", "CVR_AUC", "CTR_AUC"))
    println("-" * 80)

    results.foreach { r =>
      val cvrAuc = r.metrics.getOrElse("cvr_AUC", 0.0f)
      val ctrAuc = r.metrics.getOrElse("ctr_AUC", 0.0f)
      val status = if (r.metrics.contains("error")) "[FAIL]" else "[PASS]"
      println("%s %-13s %12.4f %12.4f".format(status, r.model, cvrAuc, ctrAuc))
    }
    println("=" * 80)
  }
}

case class MultiTaskResult(
  model: String,
  metrics: Map[String, Float]
)
