package benchmarks

import torchrec.basic.features._
import torchrec.basic.metrics._
import torchrec.data._
import torchrec.models.ranking._
import torchrec.models.matching._
import torchrec.trainers._
import torchrec.utils.DeviceSupport
import torchrec.Implicits.tensor
import torchrec.Implicits._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.util.Random
import scala.collection.mutable

/**
 * Standalone benchmark runner for AliExpress dataset with various models.
 */
object AliExpressBenchmarks {

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("AliExpress Benchmark Suite (Multiple Models)")
    println("=" * 80)

    val results = mutable.ListBuffer[AliExpressResult]()
    System.gc()
    // Run all models

    System.gc()

    // Skip sequence models - they require real sequence data
    // results += runDIENAliExpress()
    // results += runMINDAiExpress()
    // results += runDINAiExpress()
    // results += runBSTAliExpress()
    System.gc()
    System.gc()
    results += runDeepFMAliExpress()
    System.gc()
    results += runDCNv2AliExpress()
    System.gc()
    results += runWideDeepAliExpress()
    System.gc()
    results += runAFMAliExpress()
    System.gc()
    results += runFiBiNetAliExpress()
    results += runAutoIntAliExpress()
    System.gc()
    results += runMAMBAAiExpress()

    results += runXGBoostAliExpress()
    System.gc()
    results += runxDeepFMAliExpress()
    System.gc()
    printResults(results.toList)
  }

  def runDeepFMAliExpress(): AliExpressResult = {
    println("\n--- DeepFM AliExpress Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (trainDS, testDS) = AliExpressDataset.load("./data/AliExpress_NL")
      println(s"  [Data] Train: ${trainDS.size}, Test: ${testDS.size}")

      val sparseNames = trainDS.features.keys.filter(k => k.startsWith("cat_") || k.startsWith("categorical")).toList.take(10)
      val features = sparseNames.map(name => SparseFeature(name, 1000, 8))

      val clickLabel = trainDS.taskLabels.get("click")
      val testClickLabel = testDS.taskLabels.get("click")

      val trainTensorDS = new torchrec.data.TensorDataset(trainDS.features, Map.empty[String, Tensor], clickLabel)
      val testTensorDS = new torchrec.data.TensorDataset(testDS.features, Map.empty[String, Tensor], testClickLabel)

      val trainLoader = new DataLoader(trainTensorDS, 256, shuffle = true)
      val testLoader = new DataLoader(testTensorDS, 256, shuffle = false)

      val halfIdx = features.size / 2
      val model = new DeepFM(
        deepFeatures = features.take(halfIdx),
        fmFeatures = features.drop(halfIdx),
        embedDim = 8,
        mlpDims = List(256L, 128L),
        dropout = 0.2f,
        device = device
      )

      println("  [Model] DeepFM created")

      val trainer = new CTRTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader, Some(testLoader))

      val metrics = trainer.evaluate(testLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)
      println(f"  [PASS] DeepFM: AUC=$auc%.4f")

      AliExpressResult("ranking", "DeepFM", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] DeepFM: ${e.getMessage}")
        e.printStackTrace()
        AliExpressResult("ranking", "DeepFM", Map("error" -> 0.0f))
    }
  }

  def runDCNv2AliExpress(): AliExpressResult = {
    println("\n--- DCNv2 AliExpress Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (trainDS, testDS) = AliExpressDataset.load("./data/AliExpress_NL")
      println(s"  [Data] Train: ${trainDS.size}, Test: ${testDS.size}")

      val sparseNames = trainDS.features.keys.filter(k => k.startsWith("cat_") || k.startsWith("categorical")).toList.take(10)
      val features = sparseNames.map(name => SparseFeature(name, 1000, 8))

      val clickLabel = trainDS.taskLabels.get("click")
      val testClickLabel = testDS.taskLabels.get("click")

      val trainTensorDS = new torchrec.data.TensorDataset(trainDS.features, Map.empty[String, Tensor], clickLabel)
      val testTensorDS = new torchrec.data.TensorDataset(testDS.features, Map.empty[String, Tensor], testClickLabel)

      val trainLoader = new DataLoader(trainTensorDS, 256, shuffle = true)
      val testLoader = new DataLoader(testTensorDS, 256, shuffle = false)

      val model = new DCNv2(features, embedDim = 8, numCrossLayers = 3, useCrossNetMix = true, lowRank = 4, mlpDims = List(256L, 128L), dropout = 0.2f, device = device)

      println("  [Model] DCNv2 created")

      val trainer = new CTRTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader, Some(testLoader))

      val metrics = trainer.evaluate(testLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)
      println(f"  [PASS] DCNv2: AUC=$auc%.4f")

      AliExpressResult("ranking", "DCNv2", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] DCNv2: ${e.getMessage}")
        e.printStackTrace()
        AliExpressResult("ranking", "DCNv2", Map("error" -> 0.0f))
    }
  }

  def runWideDeepAliExpress(): AliExpressResult = {
    println("\n--- WideDeep AliExpress Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (trainDS, testDS) = AliExpressDataset.load("./data/AliExpress_NL")
      println(s"  [Data] Train: ${trainDS.size}, Test: ${testDS.size}")

      val sparseNames = trainDS.features.keys.filter(k => k.startsWith("cat_") || k.startsWith("categorical")).toList.take(10)
      val features = sparseNames.map(name => SparseFeature(name, 1000, 8))

      val clickLabel = trainDS.taskLabels.get("click")
      val testClickLabel = testDS.taskLabels.get("click")

      val trainTensorDS = new torchrec.data.TensorDataset(trainDS.features, Map.empty[String, Tensor], clickLabel)
      val testTensorDS = new torchrec.data.TensorDataset(testDS.features, Map.empty[String, Tensor], testClickLabel)

      val trainLoader = new DataLoader(trainTensorDS, 256, shuffle = true)
      val testLoader = new DataLoader(testTensorDS, 256, shuffle = false)

      val model = new WideDeep(features, embedDim = 8, mlpDims = List(256L, 128L), dropout = 0.2f, device = device)

      println("  [Model] WideDeep created")

      val trainer = new CTRTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader, Some(testLoader))

      val metrics = trainer.evaluate(testLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)
      println(f"  [PASS] WideDeep: AUC=$auc%.4f")

      AliExpressResult("ranking", "WideDeep", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] WideDeep: ${e.getMessage}")
        e.printStackTrace()
        AliExpressResult("ranking", "WideDeep", Map("error" -> 0.0f))
    }
  }

  def runAFMAliExpress(): AliExpressResult = {
    println("\n--- AFM AliExpress Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (trainDS, testDS) = AliExpressDataset.load("./data/AliExpress_NL")
      println(s"  [Data] Train: ${trainDS.size}, Test: ${testDS.size}")

      val sparseNames = trainDS.features.keys.filter(k => k.startsWith("cat_") || k.startsWith("categorical")).toList.take(10)
      val features = sparseNames.map(name => SparseFeature(name, 1000, 8))

      val clickLabel = trainDS.taskLabels.get("click")
      val testClickLabel = testDS.taskLabels.get("click")

      val trainTensorDS = new torchrec.data.TensorDataset(trainDS.features, Map.empty[String, Tensor], clickLabel)
      val testTensorDS = new torchrec.data.TensorDataset(testDS.features, Map.empty[String, Tensor], testClickLabel)

      val trainLoader = new DataLoader(trainTensorDS, 256, shuffle = true)
      val testLoader = new DataLoader(testTensorDS, 256, shuffle = false)

      val model = new AFM(features, embedDim = 8, attentionDim = 64, dropout = 0.2f, device = device)

      println("  [Model] AFM created")

      val trainer = new CTRTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader, Some(testLoader))

      val metrics = trainer.evaluate(testLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)
      println(f"  [PASS] AFM: AUC=$auc%.4f")

      AliExpressResult("ranking", "AFM", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] AFM: ${e.getMessage}")
        e.printStackTrace()
        AliExpressResult("ranking", "AFM", Map("error" -> 0.0f))
    }
  }

  def runFiBiNetAliExpress(): AliExpressResult = {
    println("\n--- FiBiNet AliExpress Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (trainDS, testDS) = AliExpressDataset.load("./data/AliExpress_NL")
      println(s"  [Data] Train: ${trainDS.size}, Test: ${testDS.size}")

      val sparseNames = trainDS.features.keys.filter(k => k.startsWith("cat_") || k.startsWith("categorical")).toList.take(10)
      val features = sparseNames.map(name => SparseFeature(name, 1000, 8))

      val clickLabel = trainDS.taskLabels.get("click")
      val testClickLabel = testDS.taskLabels.get("click")

      val trainTensorDS = new torchrec.data.TensorDataset(trainDS.features, Map.empty[String, Tensor], clickLabel)
      val testTensorDS = new torchrec.data.TensorDataset(testDS.features, Map.empty[String, Tensor], testClickLabel)

      val trainLoader = new DataLoader(trainTensorDS, 256, shuffle = true)
      val testLoader = new DataLoader(testTensorDS, 256, shuffle = false)

      val model = new FiBiNet(features, embedDim = 8, mlpDims = List(256L, 128L), reduction = 3, bilinearType = "field_all", dropout = 0.2f, device = device)

      println("  [Model] FiBiNet created")

      val trainer = new CTRTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader, Some(testLoader))

      val metrics = trainer.evaluate(testLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)
      println(f"  [PASS] FiBiNet: AUC=$auc%.4f")

      AliExpressResult("ranking", "FiBiNet", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] FiBiNet: ${e.getMessage}")
        e.printStackTrace()
        AliExpressResult("ranking", "FiBiNet", Map("error" -> 0.0f))
    }
  }

  def runAutoIntAliExpress(): AliExpressResult = {
    println("\n--- AutoInt AliExpress Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (trainDS, testDS) = AliExpressDataset.load("./data/AliExpress_NL")
      println(s"  [Data] Train: ${trainDS.size}, Test: ${testDS.size}")

      val sparseNames = trainDS.features.keys.filter(k => k.startsWith("cat_") || k.startsWith("categorical")).toList.take(10)
      val features = sparseNames.map(name => SparseFeature(name, 1000, 8))

      val clickLabel = trainDS.taskLabels.get("click")
      val testClickLabel = testDS.taskLabels.get("click")

      val trainTensorDS = new torchrec.data.TensorDataset(trainDS.features, Map.empty[String, Tensor], clickLabel)
      val testTensorDS = new torchrec.data.TensorDataset(testDS.features, Map.empty[String, Tensor], testClickLabel)

      val trainLoader = new DataLoader(trainTensorDS, 256, shuffle = true)
      val testLoader = new DataLoader(testTensorDS, 256, shuffle = false)

      val model = new AutoInt(features, embedDim = 8, numAttnHeads = 4, numLayers = 2, mlpDims = List(128L, 64L), dropout = 0.2f, useMlp = true, device = device)

      println("  [Model] AutoInt created")

      val trainer = new CTRTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader, Some(testLoader))

      val metrics = trainer.evaluate(testLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)
      println(f"  [PASS] AutoInt: AUC=$auc%.4f")

      AliExpressResult("ranking", "AutoInt", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] AutoInt: ${e.getMessage}")
        e.printStackTrace()
        AliExpressResult("ranking", "AutoInt", Map("error" -> 0.0f))
    }
  }

  def runBSTAliExpress(): AliExpressResult = {
    println("\n--- BST AliExpress Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (trainDS, testDS) = AliExpressDataset.load("./data/AliExpress_NL")
      println(s"  [Data] Train: ${trainDS.size}, Test: ${testDS.size}")

      // Use actual feature names from the dataset
      val catNames = trainDS.features.keys.filter(k => k.startsWith("cat_") || k.startsWith("categorical")).toList
      val seqFeatName = catNames.head  // Use first available feature
      val sparseNames = catNames.filterNot(_ == seqFeatName).take(5)
      val seqFeature = SequenceFeature(seqFeatName, 1000, 8, maxLen = 10)

      val clickLabel = trainDS.taskLabels.get("click")
      val testClickLabel = testDS.taskLabels.get("click")

      val trainTensorDS = new torchrec.data.TensorDataset(trainDS.features, Map.empty[String, Tensor], clickLabel)
      val testTensorDS = new torchrec.data.TensorDataset(testDS.features, Map.empty[String, Tensor], testClickLabel)

      val trainLoader = new DataLoader(trainTensorDS, 256, shuffle = true)
      val testLoader = new DataLoader(testTensorDS, 256, shuffle = false)

      val model = new BST(sparseNames.map(n => SparseFeature(n, 1000, 8)).toList, List(seqFeature), List(seqFeature), embedDim = 8, numHeads = 8, numLayers = 2, maxSeqLen = 10, mlpDims = List(128L, 64L), dropout = 0.2f, device = device)

      println("  [Model] BST created")

      val trainer = new CTRTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader, Some(testLoader))

      val metrics = trainer.evaluate(testLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)
      println(f"  [PASS] BST: AUC=$auc%.4f")

      AliExpressResult("ranking", "BST", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] BST: ${e.getMessage}")
        e.printStackTrace()
        AliExpressResult("ranking", "BST", Map("error" -> 0.0f))
    }
  }

  def runXGBoostAliExpress(): AliExpressResult = {
    println("\n--- XGBoost AliExpress Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (trainDS, testDS) = AliExpressDataset.load("./data/AliExpress_NL")
      println(s"  [Data] Train: ${trainDS.size}, Test: ${testDS.size}")

      val sparseNames = trainDS.features.keys.filter(k => k.startsWith("cat_") || k.startsWith("categorical")).toList.take(10)
      val features = sparseNames.map(name => SparseFeature(name, 1000, 8))

      val clickLabel = trainDS.taskLabels.get("click")
      val testClickLabel = testDS.taskLabels.get("click")

      val trainTensorDS = new torchrec.data.TensorDataset(trainDS.features, Map.empty[String, Tensor], clickLabel)
      val testTensorDS = new torchrec.data.TensorDataset(testDS.features, Map.empty[String, Tensor], testClickLabel)

      val trainLoader = new DataLoader(trainTensorDS, 256, shuffle = true)
      val testLoader = new DataLoader(testTensorDS, 256, shuffle = false)

      val linkFeatDim = (sparseNames.size * 8).toLong
      val model = new XGBoostModel(features, numTrees = 16, treeDepth = 4, embedDim = 8, linkFeatDim = linkFeatDim, device = device)

      println("  [Model] XGBoost created")

      val trainer = new CTRTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader, Some(testLoader))

      val metrics = trainer.evaluate(testLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)
      println(f"  [PASS] XGBoost: AUC=$auc%.4f")

      AliExpressResult("ranking", "XGBoost", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] XGBoost: ${e.getMessage}")
        e.printStackTrace()
        AliExpressResult("ranking", "XGBoost", Map("error" -> 0.0f))
    }
  }

  def runxDeepFMAliExpress(): AliExpressResult = {
    println("\n--- xDeepFM AliExpress Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (trainDS, testDS) = AliExpressDataset.load("./data/AliExpress_NL")
      println(s"  [Data] Train: ${trainDS.size}, Test: ${testDS.size}")

      val sparseNames = trainDS.features.keys.filter(k => k.startsWith("cat_") || k.startsWith("categorical")).toList.take(10)
      val features = sparseNames.map(name => SparseFeature(name, 1000, 8))

      val clickLabel = trainDS.taskLabels.get("click")
      val testClickLabel = testDS.taskLabels.get("click")

      val trainTensorDS = new torchrec.data.TensorDataset(trainDS.features, Map.empty[String, Tensor], clickLabel)
      val testTensorDS = new torchrec.data.TensorDataset(testDS.features, Map.empty[String, Tensor], testClickLabel)

      val trainLoader = new DataLoader(trainTensorDS, 256, shuffle = true)
      val testLoader = new DataLoader(testTensorDS, 256, shuffle = false)

      val model = new xDeepFM(features, embedDim = 8, crossLayerSizes = List(64, 32), mlpDims = List(256L, 128L), splitHalf = true, dropout = 0.2f, device = device)

      println("  [Model] xDeepFM created")

      val trainer = new CTRTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader, Some(testLoader))

      val metrics = trainer.evaluate(testLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)
      println(f"  [PASS] xDeepFM: AUC=$auc%.4f")

      AliExpressResult("ranking", "xDeepFM", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] xDeepFM: ${e.getMessage}")
        e.printStackTrace()
        AliExpressResult("ranking", "xDeepFM", Map("error" -> 0.0f))
    }
  }

  def runDINAiExpress(): AliExpressResult = {
    println("\n--- DIN AliExpress Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (trainDS, testDS) = AliExpressDataset.load("./data/AliExpress_NL")
      println(s"  [Data] Train: ${trainDS.size}, Test: ${testDS.size}")

      // Use actual feature names from the dataset
      val catNames = trainDS.features.keys.filter(k => k.startsWith("cat_") || k.startsWith("categorical")).toList
      val seqFeatName = catNames.head  // Use first available feature
      val sparseNames = catNames.filterNot(_ == seqFeatName).take(5)
      val seqFeature = SequenceFeature(seqFeatName, 1000, 8, maxLen = 10)

      val clickLabel = trainDS.taskLabels.get("click")
      val testClickLabel = testDS.taskLabels.get("click")

      val trainTensorDS = new torchrec.data.TensorDataset(trainDS.features, Map.empty[String, Tensor], clickLabel)
      val testTensorDS = new torchrec.data.TensorDataset(testDS.features, Map.empty[String, Tensor], testClickLabel)

      val trainLoader = new DataLoader(trainTensorDS, 256, shuffle = true)
      val testLoader = new DataLoader(testTensorDS, 256, shuffle = false)

      val model = new DIN(
        features = sparseNames.map(n => SparseFeature(n, 1000, 8)).toList,
        sequenceFeatures = List(seqFeature),
        embedDim = 8,
        mlpDims = List(128L, 64L),
        dropout = 0.2f,
        attentionUnits = 64,
        device = device
      )

      println("  [Model] DIN created")

      val trainer = new CTRTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader, Some(testLoader))

      val metrics = trainer.evaluate(testLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)
      println(f"  [PASS] DIN: AUC=$auc%.4f")

      AliExpressResult("ranking", "DIN", metrics)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] DIN: ${e.getMessage}")
        e.printStackTrace()
        AliExpressResult("ranking", "DIN", Map("error" -> 0.0f))
    }
  }

  def runMAMBAAiExpress(): AliExpressResult = {
    println("\n--- MAMBA AliExpress Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (trainDS, testDS) = AliExpressDataset.load("./data/AliExpress_NL")
      println(s"  [Data] Train: ${trainDS.size}, Test: ${testDS.size}")

      val numSamples = trainDS.size.toInt
      val seqLen = 10
      val vocabSize = 1000L
      val rng = new Random(42)

      // Use actual feature names from the dataset
      val catNames = trainDS.features.keys.filter(k => k.startsWith("cat_") || k.startsWith("categorical")).toList
      val seqFeatName = catNames.head  // Use first available feature
      val catTensor = trainDS.features(seqFeatName)

      val tokensArr = Array.ofDim[Float](numSamples * seqLen)
      for (i <- 0 until numSamples) {
        val baseIdx = catTensor.select(0, i).itemSafe().toInt
        for (j <- 0 until seqLen) {
          val offset = rng.nextInt(100) - 50
          tokensArr(i * seqLen + j) = math.max(0, math.min(vocabSize - 1, baseIdx + offset)).toFloat
        }
      }
      val tokensTensor = tensor(tokensArr, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

      val positionsArr = Array.range(0, seqLen).map(_.toFloat)
      val positionsFlat = Array.fill(numSamples)(positionsArr).flatten
      val positionsTensor = tensor(positionsFlat, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

      val clickLabelKey = "click"
      val clickLabelsRaw = trainDS.taskLabels.get(clickLabelKey)
      val clickLabelsArr = if (clickLabelsRaw.nonEmpty) {
        val raw = clickLabelsRaw.get
        (0 until numSamples).map(i => raw.select(0, i).itemSafe().toFloat).toArray
      } else {
        Array.tabulate(numSamples)(_ => if (rng.nextFloat() > 0.5f) 1.0f else 0.0f)
      }
      val labelsTensor = tensor(clickLabelsArr, Array(numSamples.toLong))

      val userCatNames = catNames.filterNot(_ == seqFeatName).take(4)
      val itemCatNames = catNames.filterNot(n => userCatNames.contains(n) || n == seqFeatName).take(4)

      val userFeatureTensors = userCatNames.map(name => name -> trainDS.features(name)).toMap
      val itemFeatureTensors = itemCatNames.map(name => name -> trainDS.features(name)).toMap

      val matchingDS = new torchrec.data.MatchingDataset(
        userFeatures = userFeatureTensors,
        itemFeatures = itemFeatureTensors,
        labels = Some(labelsTensor),
        tokens = Some(tokensTensor),
        positions = Some(positionsTensor)
      )

      val trainLoader = new DataLoader(matchingDS, 128, shuffle = true)

      val model = new MAMBA(
        vocabSize = vocabSize,
        embedDim = 32,
        dState = 8,
        numLayers = 2,
        maxSeqLen = seqLen,
        mlpDims = List(64L, 32L),
        dropout = 0.1f,
        device = device
      )

      println("  [Model] MAMBA created")

      val trainer = new MatchTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader)

      val recall = trainer.evaluate(trainLoader, topk = 10)
      println(f"  [PASS] MAMBA: Recall@10=$recall%.4f")

      AliExpressResult("matching", "MAMBA", Map("recall@10" -> recall))
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] MAMBA: ${e.getMessage}")
        e.printStackTrace()
        AliExpressResult("matching", "MAMBA", Map("error" -> 0.0f))
    }
  }

  def runMINDAiExpress(): AliExpressResult = {
    println("\n--- MIND AliExpress Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (trainDS, testDS) = AliExpressDataset.load("./data/AliExpress_NL")
      println(s"  [Data] Train: ${trainDS.size}, Test: ${testDS.size}")

      val numSamples = trainDS.size.toInt
      val seqLen = 10
      val vocabSize = 1000L
      val rng = new Random(42)

      // Use actual feature names from the dataset
      val catNames = trainDS.features.keys.filter(k => k.startsWith("cat_") || k.startsWith("categorical")).toList
      val seqFeatName = catNames.head  // Use first available feature
      val catTensor = trainDS.features(seqFeatName)

      // Build sequence indices for MIND
      val seqIndicesArr = Array.ofDim[Float](numSamples * seqLen)
      for (i <- 0 until numSamples) {
        val baseIdx = catTensor.select(0, i).itemSafe().toInt
        for (j <- 0 until seqLen) {
          val offset = rng.nextInt(100) - 50
          seqIndicesArr(i * seqLen + j) = math.max(0, math.min(vocabSize - 1, baseIdx + offset)).toFloat
        }
      }
      val seqIndicesTensor = tensor(seqIndicesArr, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

      val clickLabelKey = "click"
      val clickLabelsRaw = trainDS.taskLabels.get(clickLabelKey)
      val clickLabelsArr = if (clickLabelsRaw.nonEmpty) {
        val raw = clickLabelsRaw.get
        (0 until numSamples).map(i => raw.select(0, i).itemSafe().toFloat).toArray
      } else {
        Array.tabulate(numSamples)(_ => if (rng.nextFloat() > 0.5f) 1.0f else 0.0f)
      }
      val labelsTensor = tensor(clickLabelsArr, Array(numSamples.toLong))

      val userCatNames = catNames.filterNot(_ == seqFeatName).take(4)
      val userFeatureTensors = userCatNames.map(name => name -> trainDS.features(name)).toMap

      val matchingDS = new torchrec.data.MatchingDataset(
        userFeatures = userFeatureTensors,
        itemFeatures = userFeatureTensors,  // MIND uses same features for user/item matching
        labels = Some(labelsTensor)
      )

      val trainLoader = new DataLoader(matchingDS, 128, shuffle = true)

      val features = userCatNames.map(n => SparseFeature(n, 1000, 8))
      val seqFeature = SequenceFeature(seqFeatName, vocabSize.toInt, 8, maxLen = seqLen)

      val model = new MIND(
        features = features,
        sequenceFeature = seqFeature,
        embedDim = 8,
        numInterests = 4,
        capsuleDim = 4,
        mlpDims = List(64L, 32L),
        dropout = 0.1f,
        device = device
      )

      println("  [Model] MIND created")

      val trainer = new MatchTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader)

      val recall = trainer.evaluate(trainLoader, topk = 10)
      println(f"  [PASS] MIND: Recall@10=$recall%.4f")

      AliExpressResult("matching", "MIND", Map("recall@10" -> recall))
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] MIND: ${e.getMessage}")
        e.printStackTrace()
        AliExpressResult("matching", "MIND", Map("error" -> 0.0f))
    }
  }

  def runDIENAliExpress(): AliExpressResult = {
    println("\n--- DIEN AliExpress Benchmark ---")
    val device = DeviceSupport.backend
    try {
      val (trainDS, testDS) = AliExpressDataset.load("./data/AliExpress_NL")
      println(s"  [Data] Train: ${trainDS.size}, Test: ${testDS.size}")

      val numSamples = trainDS.size.toInt
      val seqLen = 10
      val vocabSize = 1000L
      val rng = new Random(42)

      // Use actual feature names from the dataset
      val catNames = trainDS.features.keys.filter(k => k.startsWith("cat_") || k.startsWith("categorical")).toList
      val seqFeatName = catNames.head  // Use first available feature
      val catTensor = trainDS.features(seqFeatName)

      // Build sequence indices for DIEN
      val seqIndicesArr = Array.ofDim[Float](numSamples * seqLen)
      for (i <- 0 until numSamples) {
        val baseIdx = catTensor.select(0, i).itemSafe().toInt
        for (j <- 0 until seqLen) {
          val offset = rng.nextInt(100) - 50
          seqIndicesArr(i * seqLen + j) = math.max(0, math.min(vocabSize - 1, baseIdx + offset)).toFloat
        }
      }
      val seqIndicesTensor = tensor(seqIndicesArr, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

      val clickLabelKey = "click"
      val clickLabelsRaw = trainDS.taskLabels.get(clickLabelKey)
      val clickLabelsArr = if (clickLabelsRaw.nonEmpty) {
        val raw = clickLabelsRaw.get
        (0 until numSamples).map(i => raw.select(0, i).itemSafe().toFloat).toArray
      } else {
        Array.tabulate(numSamples)(_ => if (rng.nextFloat() > 0.5f) 1.0f else 0.0f)
      }
      val labelsTensor = tensor(clickLabelsArr, Array(numSamples.toLong))

      val sparseNames = catNames.filterNot(_ == seqFeatName).take(4)

      // Use "seq" as the fixed sequence feature name for DIEN
      val features = sparseNames.map(n => SparseFeature(n, 1000, 8))
      val seqFeature = SequenceFeature("seq", vocabSize.toInt, 8, maxLen = seqLen)

      val matchingDS = new torchrec.data.MatchingDataset(
        userFeatures = sparseNames.map(n => n -> trainDS.features(n)).toMap,
        itemFeatures = sparseNames.map(n => n -> trainDS.features(n)).toMap,
        labels = Some(labelsTensor),
        tokens = Some(seqIndicesTensor)
      )

      val trainLoader = new DataLoader(matchingDS, 128, shuffle = true)

      val model = new DIEN(
        features = features,
        sequenceFeatures = List(seqFeature),
        embedDim = 8,
        mlpDims = List(64L, 32L),
        dropout = 0.1f,
        device = device
      )

      println("  [Model] DIEN created")

      val trainer = new MatchTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader)

      val recall = trainer.evaluate(trainLoader, topk = 10)
      println(f"  [PASS] DIEN: Recall@10=$recall%.4f")

      AliExpressResult("matching", "DIEN", Map("recall@10" -> recall))
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] DIEN: ${e.getMessage}")
        e.printStackTrace()
        AliExpressResult("matching", "DIEN", Map("error" -> 0.0f))
    }
  }

  def printResults(results: List[AliExpressResult]): Unit = {
    println("\n" + "=" * 80)
    println("AliExpress Benchmark Results Summary")
    println("=" * 80)

    results.groupBy(_.task).foreach { case (task, taskResults) =>
      println(s"\n[$task]")
      println("-" * 70)
      taskResults.foreach { r =>
        val metricStr = r.metrics.map { case (k, v) => f"$k=$v%.4f" }.mkString(", ")
        val status = if (r.metrics.contains("error")) "[FAIL]" else "[PASS]"
        println(f"$status ${r.model}%-15s $metricStr")
      }
    }
    println("=" * 80)
  }
}

case class AliExpressResult(
  task: String,
  model: String,
  metrics: Map[String, Float]
)
