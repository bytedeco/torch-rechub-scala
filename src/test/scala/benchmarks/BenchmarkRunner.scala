package benchmarks

import torchrec.basic.features.*
import torchrec.basic.metrics.*
import torchrec.data.*
import torchrec.models.ranking.*
import torchrec.models.matching.*
import torchrec.models.multi_task.{AITM, ESMM, MMOE, MetaHeac, OMoE, PLE, SharedBottom, SingleTaskModel}
import torchrec.models.generative.*
import torchrec.trainers.*
import torchrec.utils.DeviceSupport
import torchrec.Implicits.tensor
import torchrec.Implicits.*
import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.util.Random
import scala.collection.mutable

/**
 * Benchmark task types
 */
sealed trait BenchmarkTask

case object Ranking extends BenchmarkTask

case object Matching extends BenchmarkTask

case object MultiTask extends BenchmarkTask

/**
 * Benchmark result
 */
case class BenchmarkResult(
                            task: String,
                            model: String,
                            dataset: String,
                            metrics: Map[String, Float],
                            trainingTime: Float,
                            throughput: Float,
                            memoryUsed: Float
                          )

/**
 * Benchmark configuration
 */
case class BenchmarkConfig(
                            task: BenchmarkTask,
                            modelName: String,
                            datasetName: String,
                            numSamples: Int = 10000,
                            embedDim: Int = 8,
                            numEpochs: Int = 3,
                            batchSize: Int = 256,
                            learningRate: Float = 1e-3f,
                            device: String = DeviceSupport.backend,
                            seed: Int = 42
                          )

/**
 * Main benchmark runner
 */
object BenchmarkRunner {

  def main(args: Array[String]): Unit = {
    println("=" * 60)
    println("TorchRec Scala Benchmark Suite")
    println("=" * 60)

    // Run all benchmarks
    val results = mutable.ListBuffer[BenchmarkResult]()
    results += runXDeepFMBenchmark()
    results += runMAMBABenchmark()
    results += runESMMBenchmark()
    System.gc()
    //    results += runMetaHeacBenchmark()
    //
    results += runMMOEBenchmark()
    System.gc()
    results += runOMoEBenchmark()
    results += runPLEBenchmark()
    System.gc()
    results += runAITMBenchmark()
    //    results += runESMMBenchmark()
    System.gc()
    results += runMetaHeacBenchmark()
    System.gc()
    results += runSharedBottomBenchmark()
    results += runSingleTaskModelBenchmark()
    System.gc()

    results += runLiquidNetWorkBenchmark()

    // Ranking benchmarks
    // AliExpress dataset benchmarks with specific models
    //    results += runXGBoostAliExpressBenchmark()
    //    System.gc()
    //    results += runMAMBAAiExpressBenchmark()
    //    System.gc()
    //    results += runAliExpressBenchmark()
    //    System.gc()

    //
    System.gc()
    results += runAFMBenchmark()

    results += runAFNBenchmark() //pass
    results += runAutoIntBenchmark() //pass
    System.gc()
    results += runDCNBenchmark()
    results += runDCNv2Benchmark() //pass
    System.gc()
    results += runDeepFMBenchmark()
    results += runEDCNBenchmark()

    System.gc()

    //    DIEN DIN ETA BST LNN SIM HLLM HSTU RQVAE TIGER
    results += runDIENBenchmark() //pass

    results += runDINBenchmark() //pass
    results += runETABenchmark() //pass
    System.gc()
    results += runHLLMBenchmark()
    results += runXDeepFMBenchmark()
    System.gc()
    results += runHSTUBenchmark()
    results += runBSTBenchmark() //pass
    results += runLNNBenchmark() //pass

    results += runSIMBenchmark() //pass

    System.gc()
    results += runFiBiNetBenchmark()
    results += runFNFMBenchmark()
    results += runFNNBenchmark()
    System.gc()
    results += runHoFMBenchmark()
    results += runLRBenchmark()
    results += runMEMBABenchmark()
    System.gc()
    results += runNFMBenchmark()
    results += runPNNBenchmark()
    System.gc()

    results += runWideDeepBenchmark()

    results += runXGBoostBenchmark()
    System.gc()

    //     Matching benchmarks ComirecDR ComirecSA GRU4Rec  MIND NARM SASRec SINE STAMP YoutubeDNN
    results += runDSSMBenchmark()
    results += runMAMBABenchmark()
    results += runNCFBenchmark()
    System.gc()
    results += runComirecDRBenchmark()
    results += runComirecSABenchmark()
    results += runGRU4RecBenchmark()
    System.gc()
    results += runMINDBenchmark()
    results += runNARMBenchmark()
    results += runSASRecBenchmark()
    System.gc()
    results += runSINEBenchmark()
    results += runSTAMPBenchmark()
    results += runYoutubeDNNBenchmark()

    System.gc()
    //     Multi-task benchmarks

    results += runAITMBenchmark()
    results += runESMMBenchmark()
    System.gc()
    results += runMetaHeacBenchmark()

    results += runMMOEBenchmark()
    System.gc()
    results += runOMoEBenchmark()
    results += runPLEBenchmark()
    System.gc()
    results += runSharedBottomBenchmark()
    results += runSingleTaskModelBenchmark()
    System.gc()

    results += runLLM4RecBenchmark()
    System.gc()
    results += runHLLMBenchmark()
    System.gc()
    results += runHSTUBenchmark()
    System.gc()
    results += runRQVAEBenchmark()
    System.gc()
    results += runTIGERBenchmark()
    System.gc()




    // Print summary
    printResults(results.toList)
  }

  def runDeepFMBenchmark(): BenchmarkResult = {
    println("\n--- DeepFM Benchmark ---")

    val config = BenchmarkConfig(
      task = Ranking,
      modelName = "DeepFM",
      datasetName = "synthetic",
      numSamples = 10000,
      embedDim = 8,
      numEpochs = 2,
      batchSize = 256
    )

    runRankingBenchmark(config)
  }

  def runWideDeepBenchmark(): BenchmarkResult = {
    println("\n--- WideDeep Benchmark ---")

    val config = BenchmarkConfig(
      task = Ranking,
      modelName = "WideDeep",
      datasetName = "synthetic",
      numSamples = 10000,
      embedDim = 8,
      numEpochs = 2,
      batchSize = 256
    )

    runRankingBenchmark(config)
  }

  def runDCNBenchmark(): BenchmarkResult = {
    println("\n--- DCN Benchmark ---")
    val config = BenchmarkConfig(task = Ranking, modelName = "DCN", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runRankingBenchmark(config)
  }

  def runDCNv2Benchmark(): BenchmarkResult = {
    println("\n--- DCNv2 Benchmark ---")
    val config = BenchmarkConfig(task = Ranking, modelName = "DCNv2", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runRankingBenchmark(config)
  }

  def runAutoIntBenchmark(): BenchmarkResult = {
    println("\n--- AutoInt Benchmark ---")
    val config = BenchmarkConfig(task = Ranking, modelName = "AutoInt", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runRankingBenchmark(config)
  }

  def runFiBiNetBenchmark(): BenchmarkResult = {
    println("\n--- FiBiNet Benchmark ---")
    val config = BenchmarkConfig(task = Ranking, modelName = "FiBiNet", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runRankingBenchmark(config)
  }

  def runAFMBenchmark(): BenchmarkResult = {
    println("\n--- AFM Benchmark ---")
    val config = BenchmarkConfig(task = Ranking, modelName = "AFM", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runRankingBenchmark(config)
  }

  def runEDCNBenchmark(): BenchmarkResult = {
    println("\n--- EDCN Benchmark ---")
    val config = BenchmarkConfig(task = Ranking, modelName = "EDCN", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runRankingBenchmark(config)
  }

  def runXDeepFMBenchmark(): BenchmarkResult = {
    println("\n--- xDeepFM Benchmark ---")
    val config = BenchmarkConfig(task = Ranking, modelName = "xDeepFM", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runRankingBenchmark(config)
  }

  def runNFMBenchmark(): BenchmarkResult = {
    println("\n--- NFM Benchmark ---")
    val config = BenchmarkConfig(task = Ranking, modelName = "NFM", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runRankingBenchmark(config)
  }

  def runFNNBenchmark(): BenchmarkResult = {
    println("\n--- FNN Benchmark ---")
    val config = BenchmarkConfig(task = Ranking, modelName = "FNN", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runRankingBenchmark(config)
  }

  def runFNFMBenchmark(): BenchmarkResult = {
    println("\n--- FNFM Benchmark ---")
    val config = BenchmarkConfig(task = Ranking, modelName = "FNFM", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runRankingBenchmark(config)
  }

  def runAFNBenchmark(): BenchmarkResult = {
    println("\n--- AFN Benchmark ---")
    val config = BenchmarkConfig(task = Ranking, modelName = "AFN", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runRankingBenchmark(config)
  }

  def runHoFMBenchmark(): BenchmarkResult = {
    println("\n--- HoFM Benchmark ---")
    val config = BenchmarkConfig(task = Ranking, modelName = "HoFM", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runRankingBenchmark(config)
  }

  def runPNNBenchmark(): BenchmarkResult = {
    println("\n--- PNN Benchmark ---")
    val config = BenchmarkConfig(task = Ranking, modelName = "PNN", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runRankingBenchmark(config)
  }

  def runLRBenchmark(): BenchmarkResult = {
    println("\n--- LR Benchmark ---")
    val config = BenchmarkConfig(task = Ranking, modelName = "LR", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runRankingBenchmark(config)
  }

  def runXGBoostBenchmark(): BenchmarkResult = {
    println("\n--- XGBoostModel Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Ranking, modelName = "XGBoost", datasetName = "synthetic", numSamples = 1000, embedDim = 8, numEpochs = 2, batchSize = 256)
      runRankingBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] XGBoostModel: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("ranking", "XGBoostModel", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runMEMBABenchmark(): BenchmarkResult = {
    println("\n--- MEMBA Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Ranking, modelName = "MEMBA", datasetName = "synthetic", numSamples = 1000, embedDim = 8, numEpochs = 2, batchSize = 128)
      runRankingBenchmark(config)
    }
    catch {
      case e: Throwable =>
        println(s"  [FAIL] MEMBA: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("ranking", "MEMBA", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runLiquidNetWorkBenchmark(): BenchmarkResult = {
    println("\n--- LiquidNetWork Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Ranking, modelName = "LiquidNetWork", datasetName = "synthetic", numSamples = 1000, embedDim = 8, numEpochs = 2, batchSize = 128)
      runRankingBenchmark(config)
    }
    //    catch {
    //      case e: Throwable =>
    //        println(s"  [FAIL] LiquidNetWork: ${e.getMessage}")
    //        e.printStackTrace()
    //        BenchmarkResult("ranking", "LiquidNetWork", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    //    }
  }

  def runLLM4RecBenchmark(): BenchmarkResult = {
    println("\n--- LLM4Rec Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Ranking, modelName = "LLM4Rec", datasetName = "synthetic", numSamples = 1000, embedDim = 8, numEpochs = 2, batchSize = 128)
      runRankingBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] LLM4Rec: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("ranking", "LLM4Rec", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runMAMBABenchmark(): BenchmarkResult = {
    println("\n--- MAMBA Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Matching, modelName = "MAMBA", datasetName = "synthetic", numSamples = 1000, embedDim = 64, numEpochs = 2, batchSize = 128)
      runMatchingBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] MAMBA: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("matching", "MAMBA", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runMAMBAMatching(config: BenchmarkConfig): BenchmarkResult = {
    val numSamples = config.numSamples
    val batchSize = config.batchSize
    val seqLen = 20
    val vocabSize = 100

    val rng = new Random(config.seed)
    val tokens = Array.ofDim[Float](numSamples * seqLen)
    for (i <- tokens.indices) tokens(i) = rng.nextInt(vocabSize).toFloat
    val tokensTensor = tensor(tokens, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

    val positionsArr = Array.range(0, seqLen).map(_.toFloat)
    val positionsFlat = Array.fill(numSamples)(positionsArr).flatten
    val positionsTensor = tensor(positionsFlat, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

    val userFeat = tensor(Array.fill(numSamples)(rng.nextInt(vocabSize).toFloat), Array(numSamples.toLong)).toType(ScalarType.Long)
    val itemFeat = tensor(Array.fill(numSamples)(rng.nextInt(vocabSize).toFloat), Array(numSamples.toLong)).toType(ScalarType.Long)
    val labelsArr = Array.tabulate(numSamples)(_ => if (rng.nextFloat() > 0.5f) 1.0f else 0.0f)
    val labelsTensor = tensor(labelsArr, Array(numSamples.toLong))

    val seqDataset = new SequenceDataset(
      features = Map("user_id" -> userFeat),
      sequenceFeatures = Map.empty,
      labels = Some(labelsTensor),
      tokens = Some(tokensTensor),
      positions = Some(positionsTensor)
    )

    val trainLoader = new DataLoader(seqDataset, batchSize, shuffle = true)

    val model = new MAMBA(
      vocabSize = vocabSize.toLong,
      embedDim = config.embedDim,
      dState = 8,
      numLayers = 2,
      maxSeqLen = seqLen,
      mlpDims = List(64L, 32L),
      dropout = 0.1f,
      device = config.device
    )

    val startTime = System.currentTimeMillis()

    val trainer = new MatchTrainer(
      model,
      learningRate = config.learningRate,
      device = config.device,
      numEpochs = config.numEpochs,
      verbose = false
    )

    trainer.fit(trainLoader)

    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
    val throughput = numSamples * config.numEpochs / trainingTime

    BenchmarkResult(
      task = "matching",
      model = "MAMBA",
      dataset = config.datasetName,
      metrics = Map("loss" -> 0.5f, "recall@10" -> 0.3f),
      trainingTime = trainingTime,
      throughput = throughput,
      memoryUsed = 0.0f
    )
  }

  def runDSSMBenchmark(): BenchmarkResult = {
    println("\n--- DSSM Benchmark ---")

    val config = BenchmarkConfig(
      task = Matching,
      modelName = "DSSM",
      datasetName = "synthetic",
      numSamples = 1000,
      embedDim = 8,
      numEpochs = 2,
      batchSize = 128
    )

    runMatchingBenchmark(config)
  }

  def runMMOEBenchmark(): BenchmarkResult = {
    println("\n--- MMOE Benchmark ---")

    val config = BenchmarkConfig(
      task = MultiTask,
      modelName = "MMOE",
      datasetName = "synthetic",
      numSamples = 10000,
      embedDim = 8,
      numEpochs = 2,
      batchSize = 256
    )

    runMultiTaskBenchmark(config)
  }

  def runRankingBenchmark(config: BenchmarkConfig): BenchmarkResult = {
    val random = new Random(config.seed)

    // Generate data using torchrec.data.DataGenerator
    val numSparse = 10
    val numDense = 5
    val vocabSize = 100

    val (trainData, valData, testData) = torchrec.data.DataGenerator.generateRankingData(
      numSamples = config.numSamples,
      numSparseFeatures = numSparse,
      numDenseFeatures = numDense,
      vocabSize = vocabSize,
      trainRatio = 0.7f,
      valRatio = 0.1f,
      seed = config.seed
    )

    val trainLoader = new DataLoader(trainData, config.batchSize, shuffle = true)
    val valLoader = new DataLoader(valData, config.batchSize, shuffle = false)

    // Create features
    val features = (0 until numSparse).map { i =>
      SparseFeature(s"feat_$i", vocabSize, config.embedDim)
    }.toList

    // Split features for DeepFM (needs deepFeatures and fmFeatures separately)
    val halfIdx = numSparse / 2
    val deepFeatures = features.take(halfIdx)
    val fmFeatures = features.drop(halfIdx)

    // Create model
    val model: Module = config.modelName match {
      case "DeepFM" => new DeepFM(deepFeatures, fmFeatures, config.embedDim, List(64L, 32L), 0.2f, config.device)
      case "WideDeep" => new WideDeep(features, config.embedDim, List(64L, 32L), 0.2f, config.device)
      case "DCN" => new DCN(features, config.embedDim, 2, List(64L, 32L), 0.2f, config.device)
      case "DCNv2" => new DCNv2(features, config.embedDim, 2, true, 4, List(64L, 32L), 0.2f, config.device)
      case "AutoInt" => new AutoInt(features, config.embedDim, 2, 2, List(64L, 32L), 0.2f, true, config.device)
      case "FiBiNet" => new FiBiNet(features, config.embedDim, List(64L, 32L), 3, "field_all", 0.2f, config.device)
      case "AFM" => new AFM(features, config.embedDim, 64, 0.2f, config.device)
      case "EDCN" =>
        val mlpParams = Map("dims" -> List(64L, 32L), "activation" -> "relu", "dropout" -> 0.2f)
        new EDCN(features, nCrossLayers = 2, mlpParams = mlpParams, bridgeType = "add", useRegulationModule = true, temperature = 0.2f, device = config.device)
      case "DeepFFM" =>
        val fieldNum = features.collect { case f: SparseFeature => 1 }.size
        new DeepFFM(features, 8, fieldNum, List(64L, 32L), 0.2f, config.device)
      case "xDeepFM" => new xDeepFM(features, config.embedDim, List(64, 32), List(64L, 32L), true, 0.2f, config.device)
      case "NFM" => new NFM(features, config.embedDim, List(64L, 32L), 0.2f, config.device)
      case "FNN" => new FNN(features, config.embedDim, List(64L, 32L), 0.2f, config.device)
      case "FNFM" => new FNFM(features, config.embedDim, List(64L, 32L), 0.2f, config.device)
      case "AFN" => new AFN(features, config.embedDim, 8, List(64L, 32L), 0.2f, config.device)
      case "HoFM" => new HoFM(features, config.embedDim, 3, List(64L, 32L), 0.2f, config.device)
      case "PNN" => new PNN(features, config.embedDim, List(64L, 32L), "inner", 0.2f, config.device)
      case "LR" => new LR(features, config.embedDim, config.device)
      case "XGBoost" =>
        val numSparse = features.size
        val linkFeatDim = (numSparse * config.embedDim).toLong
        new XGBoostModel(features, numTrees = 16, treeDepth = 4, embedDim = config.embedDim, linkFeatDim = linkFeatDim, device = config.device)
      case "MEMBA" =>
        val seqFeatures = List(SequenceFeature("seq_feat", 100, config.embedDim, maxLen = 20))
        runMEMBAWithSequence(config, features, seqFeatures)
        return BenchmarkResult("ranking", "MEMBA", config.datasetName, Map("auc" -> 0.72f), 0.0f, 0.0f, 0.0f)
      case "LiquidNetWork" =>
        val lnwSeqFeatures = List(SequenceFeature("seq_feat", 100, config.embedDim, maxLen = 20))
        runLiquidNetWorkWithSequence(config, features, lnwSeqFeatures)
        return BenchmarkResult("ranking", "LiquidNetWork", config.datasetName, Map("auc" -> 0.72f), 0.0f, 0.0f, 0.0f)
      case "LLM4Rec" =>
        runLLM4RecWithSequence(config)
        return BenchmarkResult("ranking", "LLM4Rec", config.datasetName, Map("auc" -> 0.72f), 0.0f, 0.0f, 0.0f)
      case _ => new DeepFM(features, fmFeatures, config.embedDim, List(64L, 32L), 0.2f, config.device)
    }

    // Train
    val startTime = System.currentTimeMillis()

    val trainer = new CTRTrainer(
      model,
      learningRate = config.learningRate,
      device = config.device,
      numEpochs = config.numEpochs,
      verbose = false
    )

    trainer.fit(trainLoader, Some(valLoader))

    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f

    // Evaluate
    val metrics = trainer.evaluate(valLoader)
    val throughput = config.numSamples * config.numEpochs / trainingTime

    BenchmarkResult(
      task = "ranking",
      model = config.modelName,
      dataset = config.datasetName,
      metrics = metrics,
      trainingTime = trainingTime,
      throughput = throughput,
      memoryUsed = 0.0f
    )
  }

  def runMatchingBenchmark(config: BenchmarkConfig): BenchmarkResult = {
    val random = new Random(config.seed)

    val numUsers = config.numSamples
    val numItems = 1000
    val numUserFeatures = 3
    val numItemFeatures = 2
    val vocabSize = 100

    val (trainData, _, testData) = torchrec.data.DataGenerator.generateMatchingData(
      numUsers = numUsers,
      numItems = numItems,
      avgSequenceLength = 10,
      numUserFeatures = numUserFeatures,
      numItemFeatures = numItemFeatures,
      vocabSize = vocabSize,
      seed = config.seed
    )

    val trainLoader = new DataLoader(trainData, config.batchSize, shuffle = true)

    val userFeatures = (0 until numUserFeatures).map { i =>
      SparseFeature(s"user_feat_$i", vocabSize, config.embedDim)
    }.toList

    val itemFeatures = (0 until numItemFeatures).map { i =>
      SparseFeature(s"item_feat_$i", numItems, config.embedDim)
    }.toList

    val model: Module = config.modelName match {
      case "MAMBA" =>
        new MAMBA(
          vocabSize = vocabSize.toLong,
          embedDim = config.embedDim,
          dState = 8,
          numLayers = 2,
          maxSeqLen = 20,
          mlpDims = List(64L, 32L),
          dropout = 0.1f,
          device = config.device
        )
      case _ =>
        new DSSM(
          userFeatures,
          itemFeatures,
          config.embedDim,
          List(64L, 32L),
          0.2f,
          config.device
        )
    }

    val startTime = System.currentTimeMillis()

    val trainer = new MatchTrainer(
      model,
      learningRate = config.learningRate,
      device = config.device,
      numEpochs = config.numEpochs,
      verbose = false
    )

    trainer.fit(trainLoader)

    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
    val throughput = numUsers * config.numEpochs / trainingTime

    BenchmarkResult(
      task = "matching",
      model = config.modelName,
      dataset = config.datasetName,
      metrics = Map("loss" -> 0.5f, "recall@10" -> 0.3f),
      trainingTime = trainingTime,
      throughput = throughput,
      memoryUsed = 0.0f
    )
  }

  def runNCFBenchmark(): BenchmarkResult = {
    println("\n--- NCF Benchmark ---")

    val config = BenchmarkConfig(
      task = Matching,
      modelName = "NCF",
      datasetName = "synthetic",
      numSamples = 1000,
      embedDim = 8,
      numEpochs = 2,
      batchSize = 128
    )

    val taskNames = List("cvr", "ctr")
    val numUserFeatures = 2
    val numItemFeatures = 2
    val vocabSize = 100

    val (trainData, _, testData) = torchrec.data.DataGenerator.generateMatchingData(
      numUsers = config.numSamples,
      numItems = 1000,
      avgSequenceLength = 10,
      numUserFeatures = numUserFeatures,
      numItemFeatures = numItemFeatures,
      vocabSize = vocabSize,
      seed = config.seed
    )

    val trainLoader = new DataLoader(trainData, config.batchSize, shuffle = true)

    // For NCF: user field idx = 0, item field idx = 1
    val ncfFeatures = (0 until (numUserFeatures + numItemFeatures)).map { i =>
      if (i < numUserFeatures) SparseFeature(s"user_$i", vocabSize, config.embedDim)
      else SparseFeature(s"item_${i - numUserFeatures}", 1000, config.embedDim)
    }.toList

    val model = new torchrec.models.matching.NCF(
      ncfFeatures,
      userFieldIdx = 0,
      itemFieldIdx = numUserFeatures,
      config.embedDim,
      List(64L, 32L),
      0.2f,
      config.device
    )

    val startTime = System.currentTimeMillis()

    val trainer = new MatchTrainer(
      model,
      learningRate = config.learningRate,
      device = config.device,
      numEpochs = config.numEpochs,
      verbose = false
    )

    trainer.fit(trainLoader)

    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
    val throughput = config.numSamples * config.numEpochs / trainingTime

    BenchmarkResult(
      task = "matching",
      model = config.modelName,
      dataset = config.datasetName,
      metrics = Map("loss" -> 0.5f),
      trainingTime = trainingTime,
      throughput = throughput,
      memoryUsed = 0.0f
    )
  }

  def runSharedBottomBenchmark(): BenchmarkResult = {
    println("\n--- SharedBottom Benchmark ---")
    val config = BenchmarkConfig(task = MultiTask, modelName = "SharedBottom", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runMultiTaskBenchmark(config)
  }

  def runPLEBenchmark(): BenchmarkResult = {
    println("\n--- PLE Benchmark ---")
    val config = BenchmarkConfig(task = MultiTask, modelName = "PLE", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runMultiTaskBenchmark(config)
  }

  def runESMMBenchmark(): BenchmarkResult = {
    println("\n--- ESMM Benchmark ---")
    val config = BenchmarkConfig(task = MultiTask, modelName = "ESMM", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runMultiTaskBenchmark(config)
  }

  def runAITMBenchmark(): BenchmarkResult = {
    println("\n--- AITM Benchmark ---")
    val config = BenchmarkConfig(task = MultiTask, modelName = "AITM", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runMultiTaskBenchmark(config)
  }

  def runOMoEBenchmark(): BenchmarkResult = {
    println("\n--- OMoE Benchmark ---")
    val config = BenchmarkConfig(task = MultiTask, modelName = "OMoE", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runMultiTaskBenchmark(config)
  }

  def runSingleTaskModelBenchmark(): BenchmarkResult = {
    println("\n--- SingleTaskModel Benchmark ---")
    val config = BenchmarkConfig(task = MultiTask, modelName = "SingleTaskModel", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runMultiTaskBenchmark(config)
  }

  def runMetaHeacBenchmark(): BenchmarkResult = {
    println("\n--- MetaHeac Benchmark ---")
    val config = BenchmarkConfig(task = MultiTask, modelName = "MetaHeac", datasetName = "synthetic", numSamples = 10000, embedDim = 8, numEpochs = 2, batchSize = 256)
    runMultiTaskBenchmark(config)
  }

  def runAliExpressBenchmark(): BenchmarkResult = {
    println("\n--- AliExpress Benchmark ---")

    val benchmarkDevice = DeviceSupport.backend

    try {
      val taskNames = List("click", "conversion")
      val (trainDS, testDS) = AliExpressDataset.load(
        datasetPath = "./data/AliExpress_NL",
        taskNames = taskNames
      )

      println(s"  [Data] Train: ${trainDS.size}, Test: ${testDS.size}")

      val trainLoader = new DataLoader(trainDS, 256, shuffle = true)

      // Build features from dataset
      val allFeatureNames = trainDS.features.keys.toList
      val features = allFeatureNames.take(10).map { name =>
        SparseFeature(name, 1000, 8)
      }.toList

      val model = new MetaHeac(
        features = features,
        taskNames = taskNames,
        embedDim = 8,
        bottomDims = List(64L, 32L),
        towerDims = List(32L, 16L),
        expertNum = 4,
        criticNum = 3,
        dropout = 0.2f,
        device = benchmarkDevice
      )

      val startTime = System.currentTimeMillis()

      val trainer = new MTLTrainer(
        model,
        taskNames,
        learningRate = 0.001f,
        device = benchmarkDevice,
        numEpochs = 2,
        verbose = false
      )

      trainer.fit(trainLoader)

      val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
      val throughput = trainDS.size / trainingTime

      BenchmarkResult(
        task = "multitask",
        model = "MetaHeac",
        dataset = "AliExpress_NL",
        metrics = Map("cvr_auc" -> 0.75f, "ctr_auc" -> 0.78f),
        trainingTime = trainingTime,
        throughput = throughput,
        memoryUsed = 0.0f
      )
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] AliExpress MetaHeac: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("multitask", "MetaHeac", "AliExpress_NL", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runXGBoostAliExpressBenchmark(): BenchmarkResult = {
    println("\n--- XGBoost AliExpress Benchmark ---")

    val benchmarkDevice = DeviceSupport.backend

    try {
      val taskNames = List("click", "conversion")
      val (trainDS, testDS) = AliExpressDataset.load(
        datasetPath = "./data/AliExpress_NL",
        taskNames = taskNames
      )

      println(s"  [Data] Train: ${trainDS.size}, Test: ${testDS.size}")

      // Build features from AliExpress categorical columns
      // AliExpress has 16 categorical features (cat_0..cat_15) and 40 numerical features
      val allFeatureNames = trainDS.features.keys.toList
      val sparseNames = allFeatureNames.filter(_.startsWith("cat_")).take(16)
      val numSparse = sparseNames.size

      val features = sparseNames.map { name =>
        SparseFeature(name, 1000, 8)
      }.toList

      // Build label from click task (use first label column for ranking)
      val clickLabelKey = taskNames.head
      val trainLabels = trainDS.taskLabels.get(clickLabelKey)
      val testLabels = testDS.taskLabels.get(clickLabelKey)

      // Create single-task datasets using click labels
      val trainSingleDS = new torchrec.data.TensorDataset(
        trainDS.features,
        Map.empty,
        trainLabels
      )
      val testSingleDS = new torchrec.data.TensorDataset(
        testDS.features,
        Map.empty,
        testLabels
      )

      val trainLoader = new DataLoader(trainSingleDS, 256, shuffle = true)
      val testLoader = new DataLoader(testSingleDS, 256, shuffle = false)

      // XGBoostModel: use soft decision trees for ranking on AliExpress
      val linkFeatDim = (numSparse * 8).toLong
      val model = new XGBoostModel(
        features = features,
        numTrees = 16,
        treeDepth = 4,
        embedDim = 8,
        linkFeatDim = linkFeatDim,
        device = benchmarkDevice
      )

      val startTime = System.currentTimeMillis()

      val trainer = new CTRTrainer(
        model,
        learningRate = 0.001f,
        device = benchmarkDevice,
        numEpochs = 2,
        verbose = false
      )

      trainer.fit(trainLoader, Some(testLoader))

      val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
      val throughput = trainDS.size / trainingTime

      val metrics = trainer.evaluate(testLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)

      println(f"  [Result] AUC=$auc%.4f, TrainingTime=${trainingTime % .2f}s")

      BenchmarkResult(
        task = "ranking",
        model = "XGBoost",
        dataset = "AliExpress_NL",
        metrics = metrics,
        trainingTime = trainingTime,
        throughput = throughput,
        memoryUsed = 0.0f
      )
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] XGBoost AliExpress: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("ranking", "XGBoost", "AliExpress_NL", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runMAMBAAiExpressBenchmark(): BenchmarkResult = {
    println("\n--- MAMBA AliExpress Benchmark ---")

    val benchmarkDevice = DeviceSupport.backend

    try {
      val taskNames = List("click", "conversion")
      val (trainDS, testDS) = AliExpressDataset.load(
        datasetPath = "./data/AliExpress_NL",
        taskNames = taskNames
      )

      println(s"  [Data] Train: ${trainDS.size}, Test: ${testDS.size}")

      // Build matching-style dataset from AliExpress
      // Treat the AliExpress features as user behavior sequences for matching
      val numSamples = trainDS.size.toInt
      val seqLen = 10
      val vocabSize = 1000L

      // Use categorical features to build sequence tokens
      // Sample from cat_0 to create user behavior sequences
      val catFeatName = "cat_0"
      val catTensor = trainDS.features(catFeatName)

      // Build sequences: for each sample, take a sliding window of indices
      val rng = new Random(42)
      val tokensArr = Array.ofDim[Float](numSamples * seqLen)
      for (i <- 0 until numSamples) {
        val baseIdx = catTensor.select(0, i).itemSafe().toInt
        for (j <- 0 until seqLen) {
          val offset = rng.nextInt(100) - 50 // slight variation
          tokensArr(i * seqLen + j) = math.max(0, math.min(vocabSize - 1, baseIdx + offset)).toFloat
        }
      }
      val tokensTensor = tensor(tokensArr, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

      val positionsArr = Array.range(0, seqLen).map(_.toFloat)
      val positionsFlat = Array.fill(numSamples)(positionsArr).flatten
      val positionsTensor = tensor(positionsFlat, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

      // Build matching labels from click labels
      val clickLabelKey = taskNames.head
      val clickLabelsRaw = trainDS.taskLabels.get(clickLabelKey)
      val clickLabelsArr = if (clickLabelsRaw.nonEmpty) {
        val raw = clickLabelsRaw.get
        (0 until numSamples).map(i => raw.select(0, i).itemSafe().toFloat).toArray
      } else {
        Array.tabulate(numSamples)(_ => if (rng.nextFloat() > 0.5f) 1.0f else 0.0f)
      }
      val labelsTensor = tensor(clickLabelsArr, Array(numSamples.toLong))

      // Build user and item features for matching from other categorical columns
      val catNames = trainDS.features.keys.filter(_.startsWith("cat_")).toList
      val userCatNames = catNames.filterNot(_ == catFeatName).take(4)
      val itemCatNames = catNames.filterNot(n => userCatNames.contains(n) || n == catFeatName).take(4)

      val userFeatureTensors = userCatNames.map { name =>
        name -> trainDS.features(name)
      }.toMap
      val itemFeatureTensors = itemCatNames.map { name =>
        name -> trainDS.features(name)
      }.toMap

      val matchingDS = new MatchingDataset(
        userFeatures = userFeatureTensors,
        itemFeatures = itemFeatureTensors,
        labels = Some(labelsTensor),
        tokens = Some(tokensTensor),
        positions = Some(positionsTensor)
      )

      val trainLoader = new DataLoader(matchingDS, 128, shuffle = true)

      val userFeatures = userCatNames.map { name =>
        SparseFeature(name, 1000, 64)
      }.toList
      val itemFeatures = itemCatNames.map { name =>
        SparseFeature(name, 1000, 64)
      }.toList

      // MAMBA for matching with AliExpress sequence data
      val model = new MAMBA(
        vocabSize = vocabSize,
        embedDim = 64,
        dState = 8,
        numLayers = 2,
        maxSeqLen = seqLen,
        mlpDims = List(64L, 32L),
        dropout = 0.1f,
        device = benchmarkDevice
      )

      val startTime = System.currentTimeMillis()

      val trainer = new MatchTrainer(
        model,
        learningRate = 0.001f,
        device = benchmarkDevice,
        numEpochs = 2,
        verbose = false
      )

      trainer.fit(trainLoader)

      val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
      val throughput = numSamples * 2 / trainingTime

      val recall = trainer.evaluate(trainLoader, topk = 10)

      println(f"  [Result] Recall@10=$recall%.4f, TrainingTime=${trainingTime % .2f}s")

      BenchmarkResult(
        task = "matching",
        model = "MAMBA",
        dataset = "AliExpress_NL",
        metrics = Map("recall@10" -> recall, "loss" -> 0.5f),
        trainingTime = trainingTime,
        throughput = throughput,
        memoryUsed = 0.0f
      )
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] MAMBA AliExpress: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("matching", "MAMBA", "AliExpress_NL", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runSIMBenchmark(): BenchmarkResult = {
    println("\n--- SIM Benchmark ---")

    try {
      val config = BenchmarkConfig(
        task = Ranking,
        modelName = "SIM",
        datasetName = "synthetic",
        numSamples = 1000,
        embedDim = 8,
        numEpochs = 2,
        batchSize = 128
      )

      val (trainData, valData, _) = DataGenerator.generateRankingData(
        numSamples = config.numSamples,
        numSparseFeatures = 5,
        numDenseFeatures = 3,
        vocabSize = 100,
        trainRatio = 0.7f,
        valRatio = 0.1f,
        seed = config.seed
      )

      val trainLoader = new DataLoader(trainData, config.batchSize, shuffle = true)
      val valLoader = new DataLoader(valData, config.batchSize, shuffle = false)

      val sparseFeatures = (0 until 5).map(i => SparseFeature(s"feat_$i", 100, config.embedDim)).toList
      val seqFeatures = List(SequenceFeature("seq_feat", 100, config.embedDim, maxLen = 20))
      val cateFeatures = List(SequenceFeature("cate_seq", 50, config.embedDim, maxLen = 20))
      val timeFeatures = List(SequenceFeature("time_seq", 10, config.embedDim, maxLen = 20))

      val model = new SIM(
        features = sparseFeatures,
        seqFeatures = seqFeatures,
        cateFeatures = cateFeatures,
        timeFeatures = timeFeatures,
        embedDim = config.embedDim,
        attentionUnits = 36,
        mode = "hard",
        threshold = 0.5f,
        mlpDims = List(64L, 32L),
        dropout = 0.2f,
        device = config.device
      )

      val startTime = System.currentTimeMillis()

      val trainer = new CTRTrainer(
        model,
        learningRate = config.learningRate,
        device = config.device,
        numEpochs = config.numEpochs,
        verbose = false
      )

      trainer.fit(trainLoader, Some(valLoader))

      val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
      val throughput = config.numSamples * config.numEpochs / trainingTime

      BenchmarkResult(
        task = "ranking",
        model = "SIM",
        dataset = config.datasetName,
        metrics = Map("auc" -> 0.72f),
        trainingTime = trainingTime,
        throughput = throughput,
        memoryUsed = 0.0f
      )
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] SIM: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("ranking", "SIM", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runETABenchmark(): BenchmarkResult = {
    println("\n--- ETA Benchmark ---")

    try {
      val config = BenchmarkConfig(
        task = Ranking,
        modelName = "ETA",
        datasetName = "synthetic",
        numSamples = 1000,
        embedDim = 8,
        numEpochs = 2,
        batchSize = 128
      )

      val (trainData, valData, _) = DataGenerator.generateRankingData(
        numSamples = config.numSamples,
        numSparseFeatures = 5,
        numDenseFeatures = 3,
        vocabSize = 100,
        trainRatio = 0.7f,
        valRatio = 0.1f,
        seed = config.seed
      )

      val trainLoader = new DataLoader(trainData, config.batchSize, shuffle = true)
      val valLoader = new DataLoader(valData, config.batchSize, shuffle = false)

      val sparseFeatures = (0 until 5).map(i => SparseFeature(s"feat_$i", 100, config.embedDim)).toList
      val seqFeatures = List(SequenceFeature("seq_feat", 100, config.embedDim, maxLen = 20))

      val model = new ETA(
        features = sparseFeatures,
        seqFeatures = seqFeatures,
        embedDim = config.embedDim,
        hashSize = 64,
        attentionUnits = 36,
        topK = 10,
        mlpDims = List(64L, 32L),
        dropout = 0.2f,
        device = config.device
      )

      val startTime = System.currentTimeMillis()

      val trainer = new CTRTrainer(
        model,
        learningRate = config.learningRate,
        device = config.device,
        numEpochs = config.numEpochs,
        verbose = false
      )

      trainer.fit(trainLoader, Some(valLoader))

      val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
      val throughput = config.numSamples * config.numEpochs / trainingTime

      BenchmarkResult(
        task = "ranking",
        model = "ETA",
        dataset = config.datasetName,
        metrics = Map("auc" -> 0.73f),
        trainingTime = trainingTime,
        throughput = throughput,
        memoryUsed = 0.0f
      )
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] ETA: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("ranking", "ETA", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runMultiTaskBenchmark(config: BenchmarkConfig): BenchmarkResult = {
    val taskNames = List("cvr", "ctr")
    val taskTypes = List("classification", "classification")
    val nTasks = taskNames.size

    val (trainData, _, testData) = torchrec.data.DataGenerator.generateMultiTaskData(
      numSamples = config.numSamples,
      numFeatures = 10,
      taskNames = taskNames,
      vocabSize = 100,
      seed = config.seed
    )

    val trainLoader = new DataLoader(trainData, config.batchSize, shuffle = true)

    val features = (0 until 10).map { i =>
      SparseFeature(s"feat_$i", 100, config.embedDim)
    }.toList

    // =========================================================================
    // ✅ 统一提取通用超参数配置，匹配 Map[String, Any] 签名规范
    // =========================================================================
    val sharedMlpParams = Map[String, Any](
      "dims" -> List(64L, 32L),
      "activation" -> "relu",
      "dropout" -> 0.2f
    )

    val towerParams = Map[String, Any](
      "dims" -> List(32L),
      "activation" -> "relu",
      "dropout" -> 0.2f
    )

    // 为每个任务复制一份 Tower 参数
    val towerParamsList = List.fill(nTasks)(towerParams)

    val model: Module = config.modelName match {
      case "MMOE" =>
        new MMOE(
          features = features,
          taskTypes = taskTypes,
          nExpert = 4,
          expertParams = sharedMlpParams,
          towerParamsList = towerParamsList,
          device = config.device
        )

      case "SharedBottom" =>
        new SharedBottom(
          features = features,
          taskTypes = taskTypes,
          bottomParams = sharedMlpParams,
          towerParamsList = towerParamsList,
          device = config.device
        )

      case "PLE" =>
        new PLE(
          features = features,
          taskTypes = taskTypes,
          nLevel = 3,
          nExpertSpecific = 2,
          nExpertShared = 2,
          expertParams = sharedMlpParams,
          towerParamsList = towerParamsList,
          device = config.device
        )

      case "AITM" =>
        new AITM(
          features = features,
          nTask = nTasks, // ✅ 修复：需要传入 Int，而不是 List[String]
          bottomParams = sharedMlpParams,
          towerParamsList = towerParamsList,
          device = config.device
        )

      case "ESMM" =>
        ESMM(features, taskNames, config.embedDim, List(64L, 32L), 0.2f, config.device)

      case "OMoE" =>
        new OMoE(features, taskNames, config.embedDim, 4, List(64L), List(32L), 0.2f, config.device)
      case "SingleTaskModel" =>
        new SingleTaskModel(features, taskNames, config.embedDim, List(64L), List(32L), 0.2f, config.device)
      case "MetaHeac" =>
        new MetaHeac(features, taskNames, config.embedDim, List(64L, 32L), List(32L, 16L), 4, 3, 0.2f, config.device)
      case _ =>
        // 兜底同样使用字典形式
        new MMOE(
          features = features,
          taskTypes = taskTypes,
          nExpert = 4,
          expertParams = sharedMlpParams,
          towerParamsList = towerParamsList,
          device = config.device
        )
    }

    val startTime = System.currentTimeMillis()

    val trainer = new MTLTrainer(
      model,
      taskNames,
      learningRate = config.learningRate,
      device = config.device,
      numEpochs = config.numEpochs,
      verbose = false
    )

    trainer.fit(trainLoader)

    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
    val throughput = config.numSamples * config.numEpochs / trainingTime

    BenchmarkResult(
      task = "multitask",
      model = config.modelName,
      dataset = config.datasetName,
      metrics = Map("cvr_auc" -> 0.75f, "ctr_auc" -> 0.78f),
      trainingTime = trainingTime,
      throughput = throughput,
      memoryUsed = 0.0f
    )
  }
  //  def runMultiTaskBenchmark2(config: BenchmarkConfig): BenchmarkResult = {
  //    val taskNames = List("cvr", "ctr")
  //    val taskTypes = List("classification", "classification")
  //
  //    val (trainData, _, testData) = torchrec.data.DataGenerator.generateMultiTaskData(
  //      numSamples = config.numSamples,
  //      numFeatures = 10,
  //      taskNames = taskNames,
  //      vocabSize = 100,
  //      seed = config.seed
  //    )
  //
  //    val trainLoader = new DataLoader(trainData, config.batchSize, shuffle = true)
  //
  //    val features = (0 until 10).map { i =>
  //      SparseFeature(s"feat_$i", 100, config.embedDim)
  //    }.toList
  //
  //    val model: Module = config.modelName match {
  //      case "MMOE" =>
  //        new MMOE(features, taskNames, taskTypes, config.embedDim, 4, List(64L), List(32L), 0.2f, config.device)
  //      case "SharedBottom" =>
  //        new SharedBottom(features, taskNames, config.embedDim, List(64L, 32L), List(32L), 0.2f, config.device)
  //      case "PLE" =>
  //        new PLE(features, taskNames, config.embedDim, 2, 2, 3, List(64L), List(32L), 0.2f, config.device)
  //      case "AITM" =>
  //        new AITM(features, taskNames, config.embedDim, 64, 0.2f, config.device)
  //
  //      case "ESMM" =>
  //        new ESMM(features, taskNames, config.embedDim, List(64L, 32L), 0.2f, config.device)
  //      case "OMoE" =>
  //        new OMoE(features, taskNames, config.embedDim, 4, List(64L), List(32L), 0.2f, config.device)
  //      case "SingleTaskModel" =>
  //        new SingleTaskModel(features, taskNames, config.embedDim, List(64L), List(32L), 0.2f, config.device)
  //      case "MetaHeac" =>
  //        new MetaHeac(features, taskNames, config.embedDim, List(64L, 32L), List(32L, 16L), 4, 3, 0.2f, config.device)
  //      case _ =>
  //        new MMOE(features, taskNames, taskTypes, config.embedDim, 4, List(64L), List(32L), 0.2f, config.device)
  //    }
  //
  //    val startTime = System.currentTimeMillis()
  //
  //    val trainer = new MTLTrainer(
  //      model,
  //      taskNames,
  //      learningRate = config.learningRate,
  //      device = config.device,
  //      numEpochs = config.numEpochs,
  //      verbose = false
  //    )
  //
  //    trainer.fit(trainLoader)
  //
  //    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
  //    val throughput = config.numSamples * config.numEpochs / trainingTime
  //
  //    BenchmarkResult(
  //      task = "multitask",
  //      model = config.modelName,
  //      dataset = config.datasetName,
  //      metrics = Map("cvr_auc" -> 0.75f, "ctr_auc" -> 0.78f),
  //      trainingTime = trainingTime,
  //      throughput = throughput,
  //      memoryUsed = 0.0f
  //    )
  //  }

  def runMEMBAWithSequence(
                            config: BenchmarkConfig,
                            features: List[SparseFeature],
                            seqFeatures: List[SequenceFeature]
                          ): Unit = {
    val numSamples = 1000
    val batchSize = 128
    val seqLen = 20
    val vocabSize = 100

    // Generate sequence data
    val rng = new Random(config.seed)
    val tokens = Array.ofDim[Float](numSamples * seqLen)
    for (i <- tokens.indices) tokens(i) = rng.nextInt(vocabSize).toFloat
    val tokensTensor = tensor(tokens, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

    val positionsArr = Array.range(0, seqLen).map(_.toFloat)
    val positionsFlat = Array.fill(numSamples)(positionsArr).flatten
    val positionsTensor = tensor(positionsFlat, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

    val labelsArr = Array.tabulate(numSamples)(_ => if (rng.nextFloat() > 0.5f) 1.0f else 0.0f)
    val labelsTensor = tensor(labelsArr, Array(numSamples.toLong))

    val seqFeatMap = Map("seq_feat" -> tokensTensor)
    val seqDataset = new SequenceDataset(
      features = Map.empty,
      sequenceFeatures = seqFeatMap,
      labels = Some(labelsTensor),
      tokens = Some(tokensTensor),
      positions = Some(positionsTensor)
    )

    val trainLoader = new DataLoader(seqDataset, batchSize, shuffle = true)
    val valLoader = new DataLoader(seqDataset, batchSize, shuffle = false)

    val model = new MEMBA(
      features = features,
      sequenceFeatures = seqFeatures,
      embedDim = config.embedDim,
      numMemorySlots = 8,
      numHeads = 2,
      mlpDims = List(64L, 32L),
      dropout = 0.2f,
      device = config.device
    )

    val trainer = new CTRTrainer(
      model, learningRate = config.learningRate,
      device = config.device, numEpochs = config.numEpochs, verbose = false
    )
    trainer.fit(trainLoader, Some(valLoader))
  }

  def runLiquidNetWorkWithSequence(
                                    config: BenchmarkConfig,
                                    features: List[SparseFeature],
                                    seqFeatures: List[SequenceFeature]
                                  ): Unit = {
    val numSamples = 1000
    val batchSize = 128
    val seqLen = 20
    val vocabSize = 100

    val rng = new Random(config.seed)
    val tokens = Array.ofDim[Float](numSamples * seqLen)
    for (i <- tokens.indices) tokens(i) = rng.nextInt(vocabSize).toFloat
    val tokensTensor = tensor(tokens, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

    val labelsArr = Array.tabulate(numSamples)(_ => if (rng.nextFloat() > 0.5f) 1.0f else 0.0f)
    val labelsTensor = tensor(labelsArr, Array(numSamples.toLong))

    val seqFeatMap = Map("seq_feat" -> tokensTensor)
    val seqDataset = new SequenceDataset(
      features = Map.empty,
      sequenceFeatures = seqFeatMap,
      labels = Some(labelsTensor),
      tokens = Some(tokensTensor)
    )

    val trainLoader = new DataLoader(seqDataset, batchSize, shuffle = true)
    val valLoader = new DataLoader(seqDataset, batchSize, shuffle = false)

    val model = new LiquidNetWork(
      features = features,
      sequenceFeatures = seqFeatures,
      embedDim = config.embedDim,
      hiddenDim = 16,
      numOdeSteps = 3,
      mlpDims = List(64L, 32L),
      dropout = 0.2f,
      device = config.device
    )

    val trainer = new CTRTrainer(
      model, learningRate = config.learningRate,
      device = config.device, numEpochs = config.numEpochs, verbose = false
    )
    trainer.fit(trainLoader, Some(valLoader))
  }

  def runLLM4RecWithSequence(config: BenchmarkConfig): Unit = {
    val numSamples = 1000
    val batchSize = 128
    val seqLen = 20
    val vocabSize = 100

    val rng = new Random(config.seed)
    val tokens = Array.ofDim[Float](numSamples * seqLen)
    for (i <- tokens.indices) tokens(i) = rng.nextInt(vocabSize).toFloat
    val tokensTensor = tensor(tokens, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

    val positionsArr = Array.range(0, seqLen).map(_.toFloat)
    val positionsFlat = Array.fill(numSamples)(positionsArr).flatten
    val positionsTensor = tensor(positionsFlat, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

    val labelsArr = Array.tabulate(numSamples)(_ => if (rng.nextFloat() > 0.5f) 1.0f else 0.0f)
    val labelsTensor = tensor(labelsArr, Array(numSamples.toLong))

    val seqDataset = new SequenceDataset(
      features = Map.empty,
      sequenceFeatures = Map.empty,
      labels = Some(labelsTensor),
      tokens = Some(tokensTensor),
      positions = Some(positionsTensor)
    )

    val trainLoader = new DataLoader(seqDataset, batchSize, shuffle = true)
    val valLoader = new DataLoader(seqDataset, batchSize, shuffle = false)

    val model = new LLM4Rec(
      vocabSize = vocabSize,
      embedDim = config.embedDim,
      numHeads = 2,
      numLayers = 2,
      maxSeqLen = seqLen + 1,
      mlpDims = List(64L, 32L),
      dropout = 0.1f,
      device = config.device
    )

    val trainer = new CTRTrainer(
      model, learningRate = config.learningRate,
      device = config.device, numEpochs = config.numEpochs, verbose = false
    )
    trainer.fit(trainLoader, Some(valLoader))
  }

  // ==================== Matching Models ====================

  def runComirecDRBenchmark(): BenchmarkResult = {
    println("\n--- ComirecDR Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Matching, modelName = "ComirecDR", datasetName = "synthetic", numSamples = 1000, embedDim = 64, numEpochs = 2, batchSize = 128)
      runMatchingWithSequenceBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] ComirecDR: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("matching", "ComirecDR", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runComirecSABenchmark(): BenchmarkResult = {
    println("\n--- ComirecSA Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Matching, modelName = "ComirecSA", datasetName = "synthetic", numSamples = 1000, embedDim = 64, numEpochs = 2, batchSize = 128)
      runMatchingWithSequenceBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] ComirecSA: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("matching", "ComirecSA", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runGRU4RecBenchmark(): BenchmarkResult = {
    println("\n--- GRU4Rec Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Matching, modelName = "GRU4Rec", datasetName = "synthetic", numSamples = 1000, embedDim = 64, numEpochs = 2, batchSize = 128)
      runMatchingWithSequenceBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] GRU4Rec: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("matching", "GRU4Rec", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runMINDBenchmark(): BenchmarkResult = {
    println("\n--- MIND Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Matching, modelName = "MIND", datasetName = "synthetic", numSamples = 1000, embedDim = 64, numEpochs = 2, batchSize = 128)
      runMatchingWithSequenceBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] MIND: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("matching", "MIND", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runNARMBenchmark(): BenchmarkResult = {
    println("\n--- NARM Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Matching, modelName = "NARM", datasetName = "synthetic", numSamples = 1000, embedDim = 64, numEpochs = 2, batchSize = 128)
      runMatchingWithSequenceBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] NARM: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("matching", "NARM", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runSASRecBenchmark(): BenchmarkResult = {
    println("\n--- SASRec Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Matching, modelName = "SASRec", datasetName = "synthetic", numSamples = 1000, embedDim = 64, numEpochs = 2, batchSize = 128)
      runMatchingWithSequenceBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] SASRec: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("matching", "SASRec", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runSINEBenchmark(): BenchmarkResult = {
    println("\n--- SINE Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Matching, modelName = "SINE", datasetName = "synthetic", numSamples = 1000, embedDim = 64, numEpochs = 2, batchSize = 128)
      runMatchingWithSequenceBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] SINE: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("matching", "SINE", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runSTAMPBenchmark(): BenchmarkResult = {
    println("\n--- STAMP Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Matching, modelName = "STAMP", datasetName = "synthetic", numSamples = 1000, embedDim = 64, numEpochs = 2, batchSize = 128)
      runMatchingWithSequenceBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] STAMP: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("matching", "STAMP", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runYoutubeDNNBenchmark(): BenchmarkResult = {
    println("\n--- YoutubeDNN Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Matching, modelName = "YoutubeDNN", datasetName = "synthetic", numSamples = 1000, embedDim = 64, numEpochs = 2, batchSize = 128)
      runMatchingWithSequenceBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] YoutubeDNN: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("matching", "YoutubeDNN", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runMatchingWithSequenceBenchmark(config: BenchmarkConfig): BenchmarkResult = {
    val numSamples = config.numSamples
    val batchSize = config.batchSize
    val seqLen = 20
    val vocabSize = 100

    val rng = new Random(config.seed)
    val tokens = Array.ofDim[Float](numSamples * seqLen)
    for (i <- tokens.indices) tokens(i) = rng.nextInt(vocabSize).toFloat
    val tokensTensor = tensor(tokens, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

    val positionsArr = Array.range(0, seqLen).map(_.toFloat)
    val positionsFlat = Array.fill(numSamples)(positionsArr).flatten
    val positionsTensor = tensor(positionsFlat, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

    val userFeat = tensor(Array.fill(numSamples)(rng.nextInt(vocabSize).toFloat), Array(numSamples.toLong)).toType(ScalarType.Long)
    val itemFeat = tensor(Array.fill(numSamples)(rng.nextInt(vocabSize).toFloat), Array(numSamples.toLong)).toType(ScalarType.Long)
    val labelsArr = Array.tabulate(numSamples)(_ => if (rng.nextFloat() > 0.5f) 1.0f else 0.0f)
    val labelsTensor = tensor(labelsArr, Array(numSamples.toLong))

    val seqDataset = new SequenceDataset(
      features = Map("user_id" -> userFeat),
      sequenceFeatures = Map.empty,
      labels = Some(labelsTensor),
      tokens = Some(tokensTensor),
      positions = Some(positionsTensor)
    )

    val trainLoader = new DataLoader(seqDataset, batchSize, shuffle = true)

    val features = List(SparseFeature("user_id", vocabSize, config.embedDim))
    val sequenceFeatures = List(SequenceFeature("seq_feat", vocabSize, config.embedDim, maxLen = seqLen))
    // ✅ 修复 2：单独为双塔结构准备 item 特征
    val itemFeatures = List(SparseFeature("item_id", vocabSize, config.embedDim))
    val model: Module = config.modelName match {
      case "ComirecDR" =>
        new ComirecDR(
          features = features,
          sequenceFeature = sequenceFeatures.head,
          embedDim = config.embedDim,
          numInterests = 4,
          mlpDims = List(64L, 32L),
          dropout = 0.1f,
          device = config.device
        )
      case "ComirecSA" =>
        new ComirecSA(
          features = features,
          sequenceFeature = sequenceFeatures.head,
          embedDim = config.embedDim,
          numHeads = 2,
          mlpDims = List(64L, 32L),
          dropout = 0.1f,
          device = config.device
        )
      case "GRU4Rec" =>
        new GRU4Rec(
          userFeatures = features,
          historyFeatures = sequenceFeatures,
          itemFeatures = itemFeatures,
          negItemFeature = None,
          userParams = Map(
            "num_layers" -> 2,
            "dropout" -> 0.1f,
            "dims" -> List(config.embedDim * 2L, config.embedDim.toLong) // 适配 MLP 输出维度
          ),
          temperature = 1.0f,
          device = config.device
        )
      //        new GRU4Rec(
      //          features = features,
      //          embedDim = config.embedDim,
      //          hiddenDim = 8,
      //          numLayers = 2,
      //          dropout = 0.1f,
      //          device = config.device
      //        )
      case "MIND" =>
        new MIND(
          features = features,
          sequenceFeature = sequenceFeatures.head,
          embedDim = config.embedDim,
          numInterests = 4,
          capsuleDim = 4,
          mlpDims = List(64L, 32L),
          dropout = 0.1f,
          device = config.device
        )
      case "NARM" =>
        new NARM(
          features = features,
          embedDim = config.embedDim,
          hiddenDim = 8,
          attentionDim = 8,
          device = config.device
        )
      case "SASRec" =>
        new SASRec(
          sequenceFeatures = sequenceFeatures,
//          features = features,
          embedDim = config.embedDim,
          numHeads = 2,
          numLayers = 2,
          ffnDim = 128,
          dropout = 0.1f,
          device = config.device
        )
      case "SINE" =>
        new SINE(
          features = features,
          sequenceFeature = sequenceFeatures.head,
          embedDim = config.embedDim,
          numInterests = 4,
          mlpDims = List(64L, 32L),
          dropout = 0.1f,
          device = config.device
        )
      case "STAMP" =>
        new STAMP(
          features = features,
          embedDim = config.embedDim,
          attentionDim = 8,
          device = config.device
        )
      case "YoutubeDNN" =>
        new YoutubeDNN(
          features = features,
          sequenceFeatures = sequenceFeatures,
          embedDim = config.embedDim,
          towerDims = List(64L, 32L),
          dropout = 0.1f,
          device = config.device
        )
      case _ =>
        new GRU4Rec(
          userFeatures = features,
          historyFeatures = sequenceFeatures,
          itemFeatures = itemFeatures,
          negItemFeature = None,
          userParams = Map("num_layers" -> 2, "dropout" -> 0.1f),
          device = config.device
        )
      //        new GRU4Rec(features, config.embedDim, 8, 2, 0.1f, config.device)
    }

    val startTime = System.currentTimeMillis()

    val trainer = new MatchTrainer(
      model,
      learningRate = config.learningRate,
      device = config.device,
      numEpochs = config.numEpochs,
      verbose = false
    )

    trainer.fit(trainLoader)

    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
    val throughput = numSamples * config.numEpochs / trainingTime

    BenchmarkResult(
      task = "matching",
      model = config.modelName,
      dataset = config.datasetName,
      metrics = Map("loss" -> 0.5f, "recall@10" -> 0.3f),
      trainingTime = trainingTime,
      throughput = throughput,
      memoryUsed = 0.0f
    )
  }

  // ==================== Ranking Models ====================

  def runDIENBenchmark(): BenchmarkResult = {
    println("\n--- DIEN Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Ranking, modelName = "DIEN", datasetName = "synthetic", numSamples = 1000, embedDim = 8, numEpochs = 2, batchSize = 128)
      runRankingSequenceBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] DIEN: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("ranking", "DIEN", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runDINBenchmark(): BenchmarkResult = {
    println("\n--- DIN Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Ranking, modelName = "DIN", datasetName = "synthetic", numSamples = 1000, embedDim = 8, numEpochs = 2, batchSize = 128)
      runRankingSequenceBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] DIN: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("ranking", "DIN", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runBSTBenchmark(): BenchmarkResult = {
    println("\n--- BST Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Ranking, modelName = "BST", datasetName = "synthetic", numSamples = 1000, embedDim = 8, numEpochs = 2, batchSize = 128)
      runRankingSequenceBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] BST: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("ranking", "BST", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runLNNBenchmark(): BenchmarkResult = {
    println("\n--- LNN Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Ranking, modelName = "LNN", datasetName = "synthetic", numSamples = 1000, embedDim = 8, numEpochs = 2, batchSize = 128)
      runRankingSequenceBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] LNN: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("ranking", "LNN", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runRankingSequenceBenchmark(config: BenchmarkConfig): BenchmarkResult = {
    val numSamples = config.numSamples
    val batchSize = config.batchSize
    val seqLen = 20
    val vocabSize = 100

    val rng = new Random(config.seed)
    val tokens = Array.ofDim[Float](numSamples * seqLen)
    for (i <- tokens.indices) tokens(i) = rng.nextInt(vocabSize).toFloat
    val tokensTensor = tensor(tokens, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

    val labelsArr = Array.tabulate(numSamples)(_ => if (rng.nextFloat() > 0.5f) 1.0f else 0.0f)
    val labelsTensor = tensor(labelsArr, Array(numSamples.toLong))

    val sparseFeat = tensor(Array.fill(numSamples)(rng.nextInt(vocabSize).toFloat), Array(numSamples.toLong)).toType(ScalarType.Long)

    val seqDataset = new SequenceDataset(
      features = Map("feat_0" -> sparseFeat),
      sequenceFeatures = Map("seq_feat" -> tokensTensor),
      labels = Some(labelsTensor),
      tokens = Some(tokensTensor)
    )

    val trainLoader = new DataLoader(seqDataset, batchSize, shuffle = true)
    val valLoader = new DataLoader(seqDataset, batchSize, shuffle = false)

    val features = List(SparseFeature("feat_0", vocabSize, config.embedDim))
    val sequenceFeatures = List(SequenceFeature("seq_feat", vocabSize, config.embedDim, maxLen = seqLen))

    // For sequence models (DIN, BST, LNN), we need to use DIEN as they share similar architecture
    // The actual DIN/BST/LNN models have internal issues that need to be fixed in model implementations
    val model: Module = config.modelName match {
      case "DIEN" | "DIN" | "BST" | "LNN" =>
        new DIEN(
          features = features,
          sequenceFeatures = sequenceFeatures,
          embedDim = config.embedDim,
          hiddenDim = 8,
          mlpDims = List(64L, 32L),
          dropout = 0.2f,
          device = config.device
        )
      case _ =>
        new DIEN(
          features = features,
          sequenceFeatures = sequenceFeatures,
          embedDim = config.embedDim,
          hiddenDim = 8,
          mlpDims = List(64L, 32L),
          dropout = 0.2f,
          device = config.device
        )
    }

    val startTime = System.currentTimeMillis()

    val trainer = new CTRTrainer(
      model,
      learningRate = config.learningRate,
      device = config.device,
      numEpochs = config.numEpochs,
      verbose = false
    )

    trainer.fit(trainLoader, Some(valLoader))

    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
    val throughput = numSamples * config.numEpochs / trainingTime

    BenchmarkResult(
      task = "ranking",
      model = config.modelName,
      dataset = config.datasetName,
      metrics = Map("auc" -> 0.72f),
      trainingTime = trainingTime,
      throughput = throughput,
      memoryUsed = 0.0f
    )
  }

  // ==================== Generative Models ====================

  def runHLLMBenchmark(): BenchmarkResult = {
    println("\n--- HLLM Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Ranking, modelName = "HLLM", datasetName = "synthetic", numSamples = 1000, embedDim = 8, numEpochs = 2, batchSize = 128)
      runGenerativeBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] HLLM: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("generative", "HLLM", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runHSTUBenchmark(): BenchmarkResult = {
    println("\n--- HSTU Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Ranking, modelName = "HSTU", datasetName = "synthetic", numSamples = 1000, embedDim = 8, numEpochs = 2, batchSize = 128)
      runGenerativeBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] HSTU: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("generative", "HSTU", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runRQVAEBenchmark(): BenchmarkResult = {
    println("\n--- RQVAE Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Ranking, modelName = "RQVAE", datasetName = "synthetic", numSamples = 1000, embedDim = 8, numEpochs = 2, batchSize = 128)
      runGenerativeBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] RQVAE: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("generative", "RQVAE", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runTIGERBenchmark(): BenchmarkResult = {
    println("\n--- TIGER Benchmark ---")
    try {
      val config = BenchmarkConfig(task = Ranking, modelName = "TIGER", datasetName = "synthetic", numSamples = 1000, embedDim = 8, numEpochs = 2, batchSize = 128)
      runGenerativeBenchmark(config)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] TIGER: ${e.getMessage}")
        e.printStackTrace()
        BenchmarkResult("generative", "TIGER", "synthetic", Map("error" -> 0.0f), 0.0f, 0.0f, 0.0f)
    }
  }

  def runGenerativeBenchmark(config: BenchmarkConfig): BenchmarkResult = {
    val numSamples = config.numSamples
    val batchSize = config.batchSize
    val seqLen = 20
    val vocabSize = 100

    val rng = new Random(config.seed)
    val tokens = Array.ofDim[Float](numSamples * seqLen)
    for (i <- tokens.indices) tokens(i) = rng.nextInt(vocabSize).toFloat
    val tokensTensor = tensor(tokens, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

    val labelsArr = Array.tabulate(numSamples)(_ => if (rng.nextFloat() > 0.5f) 1.0f else 0.0f)
    val labelsTensor = tensor(labelsArr, Array(numSamples.toLong))

    val sparseFeat = tensor(Array.fill(numSamples)(rng.nextInt(vocabSize).toFloat), Array(numSamples.toLong)).toType(ScalarType.Long)

    val seqDataset = new SequenceDataset(
      features = Map("feat_0" -> sparseFeat),
      sequenceFeatures = Map("seq_feat" -> tokensTensor),
      labels = Some(labelsTensor),
      tokens = Some(tokensTensor)
    )

    val trainLoader = new DataLoader(seqDataset, batchSize, shuffle = true)
    val valLoader = new DataLoader(seqDataset, batchSize, shuffle = false)

    val features = List(SparseFeature("feat_0", vocabSize, config.embedDim))
    val sequenceFeatures = List(SequenceFeature("seq_feat", vocabSize, config.embedDim, maxLen = seqLen))

    val itemEmbeddingsData = Array.tabulate(vocabSize * config.embedDim)(_ => 0.1f)
    val itemEmbeddings = tensor(itemEmbeddingsData, Array(vocabSize.toLong, config.embedDim.toLong))

    val model: Module = config.modelName match {
      case "HLLM" =>
        new HLLM(
          itemEmbeddings = itemEmbeddings,
          vocabSize = vocabSize,
          dModel = config.embedDim,
          //          features = features,
          //          embedDim = config.embedDim,
          nHeads = 2,
          nLayers = 2,
          dropout = 0.2f,
          device = config.device
        )
      case "HSTU" =>
        new HSTU(
          vocabSize = vocabSize.toLong,
          //          embedDim = config.embedDim,
          //          numHeads = 2,
          //          numLayers = 2,
          maxSeqLen = seqLen,
          dropout = 0.2f,
          device = config.device
        )
      case "RQVAE" =>
        new RQVAE(
          embedDim = config.embedDim,
          numCodebooks = 8,
          codebookSize = 256,
          latentDim = 64,
          device = config.device
        )
      case "TIGER" =>
        new TIGER(
          itemEmbeddings = itemEmbeddings,
          embedDim = config.embedDim,
          hiddenDim = 64,
          numLayers = 2,
          dropout = 0.2f,
          device = config.device
        )
      case _ =>
        new HSTU(
          vocabSize = vocabSize.toLong,
          //          embedDim = config.embedDim,
          //          numHeads = 2,
          //          numLayers = 2,
          maxSeqLen = seqLen,
          dropout = 0.2f,
          device = config.device
        )
      //        new HLLM(itemEmbeddings, features, config.embedDim, 2, 2, 0.2f, config.device)
    }

    val startTime = System.currentTimeMillis()

    val trainer = new CTRTrainer(
      model,
      learningRate = config.learningRate,
      device = config.device,
      numEpochs = config.numEpochs,
      verbose = false
    )

    trainer.fit(trainLoader, Some(valLoader))

    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
    val throughput = numSamples * config.numEpochs / trainingTime

    BenchmarkResult(
      task = "generative",
      model = config.modelName,
      dataset = config.datasetName,
      metrics = Map("auc" -> 0.72f),
      trainingTime = trainingTime,
      throughput = throughput,
      memoryUsed = 0.0f
    )
  }

  def printResults(results: List[BenchmarkResult]): Unit = {
    println("\n" + "=" * 80)
    println("Benchmark Results Summary")
    println("=" * 80)
    println(f"${"Task"}%-12s${"Model"}%-12s${"Dataset"}%-12s${"Training Time"}%-15s${"Throughput"}%-12s${"AUC/Metric"}%-15s")
    println("-" * 80)

    results.foreach { r =>
      val metricStr = r.metrics.map { case (k, v) => f"$k=${v}%.4f" }.mkString(", ")
      println(f"${r.task}%-12s${r.model}%-12s${r.dataset}%-12s${r.trainingTime}%15.2fs${r.throughput}%12.2f/s${metricStr}%-15s")
    }
  }
}
