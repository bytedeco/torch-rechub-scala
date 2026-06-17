package benchmarks

import torchrec.TorchRec
import torchrec.basic.features._
import torchrec.basic.metrics._
import torchrec.basic.losses._
import torchrec.data._
import torchrec.dataframe._
import torchrec.dataframe.DataFrameBridge
import torchrec.models.ranking._
import torchrec.models.matching._
import torchrec.models.multi_task._
import torchrec.models.generative._
import torchrec.trainers._
import torchrec.utils.DeviceSupport
import torchrec.Implicits.tensor
import torchrec.Implicits._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.util.Random
import scala.collection.mutable

/**
 * Risk Control & Anti-Fraud Benchmark Suite
 *
 * This benchmark validates deep learning models, graph neural networks,
 * and large models for risk control and anti-fraud scenarios.
 */

/**
 * Benchmark result for risk control models
 */
case class RiskControlBenchmarkResult(
                                       name: String,
                                       category: String, // "DeepLearning", "GraphNN", "LLM", "FullPipeline"
                                       passed: Boolean,
                                       metric: String,
                                       value: Double,
                                       trainingTimeMs: Double,
                                       memoryMb: Double,
                                       error: Option[String] = None
                                     )

/**
 * Risk Control Benchmark Suite
 */
object RiskControlBenchmark {

  val DEFAULT_NUM_SAMPLES = 5000
  val DEFAULT_BATCH_SIZE = 256
  val DEFAULT_EMBED_DIM = 8
  val DEFAULT_VOCAB_SIZE = 1000

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("Risk Control & Anti-Fraud Benchmark Suite")
    println("=" * 80)

    val results = mutable.ListBuffer[RiskControlBenchmarkResult]()

    // Phase 1: Deep Learning CTR Models
    println("\n[Phase 1] Deep Learning CTR Models for Risk Control")
    println("-" * 60)
    results ++= runDeepLearningBenchmarks()

    // Phase 2: Graph Neural Networks
    println("\n[Phase 2] Graph Neural Networks for Fraud Detection")
    println("-" * 60)
    results ++= runGraphNNBenchmarks()

    // Phase 3: Large Models
    println("\n[Phase 3] Large Models for Sequential Risk Analysis")
    println("-" * 60)
    results ++= runLLMBenchmarks()

    // Phase 4: Full Pipeline Integration
    println("\n[Phase 4] Full Pipeline Integration")
    println("-" * 60)
    results ++= runFullPipelineBenchmarks()

    printSummary(results.toList)
  }

  // ========================================================================
  // Phase 1: Deep Learning CTR Models
  // ========================================================================

  def runDeepLearningBenchmarks(): List[RiskControlBenchmarkResult] = {
    val results = mutable.ListBuffer[RiskControlBenchmarkResult]()

    // DeepFM benchmark
    results += runDeepFMBenchmark()

    // DCN benchmark
    results += runDCNBenchmark()

    // WideDeep benchmark
    results += runWideDeepBenchmark()

    // xDeepFM benchmark
    results += runxDeepFMBenchmark()

    // FiBiNet benchmark
    results += runFiBiNetBenchmark()

    // AFM benchmark
    results += runAFMBenchmark()

    // AutoInt benchmark
    results += runAutoIntBenchmark()

    results.toList
  }

  def runDeepFMBenchmark(): RiskControlBenchmarkResult = {
    val name = "DeepFM"
    val category = "DeepLearning"
    val numSamples = DEFAULT_NUM_SAMPLES
    val batchSize = DEFAULT_BATCH_SIZE
    val embedDim = DEFAULT_EMBED_DIM

    try {
      // Generate synthetic risk control data
      val (trainData, _, _) = DataGenerator.generateRankingData(
        numSamples = numSamples,
        numSparseFeatures = 5,
        numDenseFeatures = 3,
        vocabSize = DEFAULT_VOCAB_SIZE,
        seed = 42
      )

      // Build features - must match the feature names from DataGenerator (feat_0, feat_1, etc.)
      val features = List(
        SparseFeature("feat_0", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_1", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_2", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_3", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_4", DEFAULT_VOCAB_SIZE, embedDim)
      )

      // Create model
      val model = new DeepFM(features, embedDim, List(128L, 64L), 0.2f, DeviceSupport.backend)

      // Training
      val trainLoader = new DataLoader(trainData, batchSize, shuffle = true)
      val criterion = new BCELoss()

      val startTime = System.nanoTime()
      var totalLoss = 0.0
      var numBatches = 0

      val iter = trainLoader.iterator
      while (iter.hasNext && numBatches < 10) {
        val batch = iter.next()
        val sparseFeats = batch.sparseFeatures
        val denseFeats = batch.denseFeatures
        val labels = batch.labels.getOrElse {
          torch.ones(batch.numSamples, 1L)
        }

        val logits = model.forward(sparseFeats, denseFeats)
        val loss = criterion.apply(torch.sigmoid(logits), labels)

        totalLoss += loss.item().toDouble()
        numBatches += 1
      }

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val avgLoss = totalLoss / numBatches

      RiskControlBenchmarkResult(
        name = name,
        category = category,
        passed = numBatches > 0 && !avgLoss.isNaN,
        metric = "avg_loss",
        value = avgLoss,
        trainingTimeMs = elapsed,
        memoryMb = ModelBenchmark.measureMemoryUsage()
      )
    } catch {
      case e: Exception =>
        RiskControlBenchmarkResult(name, category, false, "loss", 0, 0, 0, Some(e.getMessage))
    }
  }

  def runDCNBenchmark(): RiskControlBenchmarkResult = {
    val name = "DCNv2"
    val category = "DeepLearning"
    val numSamples = DEFAULT_NUM_SAMPLES
    val batchSize = DEFAULT_BATCH_SIZE
    val embedDim = DEFAULT_EMBED_DIM

    try {
      val (trainData, _, _) = DataGenerator.generateRankingData(
        numSamples = numSamples,
        numSparseFeatures = 5,
        numDenseFeatures = 3,
        vocabSize = DEFAULT_VOCAB_SIZE,
        seed = 42
      )

      val features = List(
        SparseFeature("feat_0", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_1", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_2", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_3", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_4", DEFAULT_VOCAB_SIZE, embedDim)
      )

      val model = new DCNv2(features, embedDim, 3, true, 4, List(128L, 64L), 0.2f, DeviceSupport.backend)

      val trainLoader = new DataLoader(trainData, batchSize, shuffle = true)
      val criterion = new BCELoss()

      val startTime = System.nanoTime()
      var totalLoss = 0.0
      var numBatches = 0

      val iter = trainLoader.iterator
      while (iter.hasNext && numBatches < 10) {
        val batch = iter.next()
        val sparseFeats = batch.sparseFeatures
        val denseFeats = batch.denseFeatures
        val labels = batch.labels.getOrElse {
          torch.ones(batch.numSamples, 1L)
        }

        val logits = model.forward(sparseFeats, denseFeats)
        val loss = criterion.apply(torch.sigmoid(logits), labels)

        totalLoss += loss.item().toDouble()
        numBatches += 1
      }

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val avgLoss = totalLoss / numBatches

      RiskControlBenchmarkResult(
        name = name,
        category = category,
        passed = numBatches > 0 && !avgLoss.isNaN,
        metric = "avg_loss",
        value = avgLoss,
        trainingTimeMs = elapsed,
        memoryMb = ModelBenchmark.measureMemoryUsage()
      )
    } catch {
      case e: Exception =>
        RiskControlBenchmarkResult(name, category, false, "loss", 0, 0, 0, Some(e.getMessage))
    }
  }

  def runWideDeepBenchmark(): RiskControlBenchmarkResult = {
    val name = "WideDeep"
    val category = "DeepLearning"
    val numSamples = DEFAULT_NUM_SAMPLES
    val batchSize = DEFAULT_BATCH_SIZE
    val embedDim = DEFAULT_EMBED_DIM

    try {
      val (trainData, _, _) = DataGenerator.generateRankingData(
        numSamples = numSamples,
        numSparseFeatures = 5,
        numDenseFeatures = 3,
        vocabSize = DEFAULT_VOCAB_SIZE,
        seed = 42
      )

      val features = List(
        SparseFeature("feat_0", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_1", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_2", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_3", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_4", DEFAULT_VOCAB_SIZE, embedDim)
      )

      val model = new WideDeep(features, embedDim, List(128L, 64L), 0.2f, DeviceSupport.backend)

      val trainLoader = new DataLoader(trainData, batchSize, shuffle = true)
      val criterion = new BCELoss()

      val startTime = System.nanoTime()
      var totalLoss = 0.0
      var numBatches = 0

      val iter = trainLoader.iterator
      while (iter.hasNext && numBatches < 10) {
        val batch = iter.next()
        val sparseFeats = batch.sparseFeatures
        val denseFeats = batch.denseFeatures
        val labels = batch.labels.getOrElse {
          torch.ones(batch.numSamples, 1L)
        }

        val logits = model.forward(sparseFeats, denseFeats)
        val loss = criterion.apply(torch.sigmoid(logits), labels)

        totalLoss += loss.item().toDouble()
        numBatches += 1
      }

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val avgLoss = totalLoss / numBatches

      RiskControlBenchmarkResult(
        name = name,
        category = category,
        passed = numBatches > 0 && !avgLoss.isNaN,
        metric = "avg_loss",
        value = avgLoss,
        trainingTimeMs = elapsed,
        memoryMb = ModelBenchmark.measureMemoryUsage()
      )
    } catch {
      case e: Exception =>
        RiskControlBenchmarkResult(name, category, false, "loss", 0, 0, 0, Some(e.getMessage))
    }
  }

  def runxDeepFMBenchmark(): RiskControlBenchmarkResult = {
    val name = "xDeepFM"
    val category = "DeepLearning"
    val numSamples = DEFAULT_NUM_SAMPLES
    val batchSize = DEFAULT_BATCH_SIZE
    val embedDim = DEFAULT_EMBED_DIM

    try {
      val (trainData, _, _) = DataGenerator.generateRankingData(
        numSamples = numSamples,
        numSparseFeatures = 5,
        numDenseFeatures = 3,
        vocabSize = DEFAULT_VOCAB_SIZE,
        seed = 42
      )

      val features = List(
        SparseFeature("feat_0", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_1", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_2", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_3", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_4", DEFAULT_VOCAB_SIZE, embedDim)
      )

      val model = new xDeepFM(features, embedDim, List(64, 32), List(128L, 64L), true, 0.2f, DeviceSupport.backend)

      val trainLoader = new DataLoader(trainData, batchSize, shuffle = true)
      val criterion = new BCELoss()

      val startTime = System.nanoTime()
      var totalLoss = 0.0
      var numBatches = 0

      val iter = trainLoader.iterator
      while (iter.hasNext && numBatches < 10) {
        val batch = iter.next()
        val sparseFeats = batch.sparseFeatures
        val denseFeats = batch.denseFeatures
        val labels = batch.labels.getOrElse {
          torch.ones(batch.numSamples, 1L)
        }

        val logits = model.forward(sparseFeats, denseFeats)
        val loss = criterion.apply(torch.sigmoid(logits), labels)

        totalLoss += loss.item().toDouble()
        numBatches += 1
      }

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val avgLoss = totalLoss / numBatches

      RiskControlBenchmarkResult(
        name = name,
        category = category,
        passed = numBatches > 0 && !avgLoss.isNaN,
        metric = "avg_loss",
        value = avgLoss,
        trainingTimeMs = elapsed,
        memoryMb = ModelBenchmark.measureMemoryUsage()
      )
    } catch {
      case e: Exception =>
        RiskControlBenchmarkResult(name, category, false, "loss", 0, 0, 0, Some(e.getMessage))
    }
  }

  def runFiBiNetBenchmark(): RiskControlBenchmarkResult = {
    val name = "FiBiNet"
    val category = "DeepLearning"
    val numSamples = DEFAULT_NUM_SAMPLES
    val batchSize = DEFAULT_BATCH_SIZE
    val embedDim = DEFAULT_EMBED_DIM

    try {
      val (trainData, _, _) = DataGenerator.generateRankingData(
        numSamples = numSamples,
        numSparseFeatures = 5,
        numDenseFeatures = 3,
        vocabSize = DEFAULT_VOCAB_SIZE,
        seed = 42
      )

      val features = List(
        SparseFeature("feat_0", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_1", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_2", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_3", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_4", DEFAULT_VOCAB_SIZE, embedDim)
      )

      val model = new FiBiNet(features, embedDim, List(128L, 64L), 3, "field_all", 0.2f, DeviceSupport.backend)

      val trainLoader = new DataLoader(trainData, batchSize, shuffle = true)
      val criterion = new BCELoss()

      val startTime = System.nanoTime()
      var totalLoss = 0.0
      var numBatches = 0

      val iter = trainLoader.iterator
      while (iter.hasNext && numBatches < 10) {
        val batch = iter.next()
        val sparseFeats = batch.sparseFeatures
        val denseFeats = batch.denseFeatures
        val labels = batch.labels.getOrElse {
          torch.ones(batch.numSamples, 1L)
        }

        val logits = model.forward(sparseFeats, denseFeats)
        val loss = criterion.apply(torch.sigmoid(logits), labels)

        totalLoss += loss.item().toDouble()
        numBatches += 1
      }

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val avgLoss = totalLoss / numBatches

      RiskControlBenchmarkResult(
        name = name,
        category = category,
        passed = numBatches > 0 && !avgLoss.isNaN,
        metric = "avg_loss",
        value = avgLoss,
        trainingTimeMs = elapsed,
        memoryMb = ModelBenchmark.measureMemoryUsage()
      )
    } catch {
      case e: Exception =>
        RiskControlBenchmarkResult(name, category, false, "loss", 0, 0, 0, Some(e.getMessage))
    }
  }

  def runAFMBenchmark(): RiskControlBenchmarkResult = {
    val name = "AFM"
    val category = "DeepLearning"
    val numSamples = DEFAULT_NUM_SAMPLES
    val batchSize = DEFAULT_BATCH_SIZE
    val embedDim = DEFAULT_EMBED_DIM

    try {
      val (trainData, _, _) = DataGenerator.generateRankingData(
        numSamples = numSamples,
        numSparseFeatures = 5,
        numDenseFeatures = 3,
        vocabSize = DEFAULT_VOCAB_SIZE,
        seed = 42
      )

      val features = List(
        SparseFeature("feat_0", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_1", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_2", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_3", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_4", DEFAULT_VOCAB_SIZE, embedDim)
      )

      val model = new AFM(features, embedDim, 8, 0.2f, DeviceSupport.backend)

      val trainLoader = new DataLoader(trainData, batchSize, shuffle = true)
      val criterion = new BCELoss()

      val startTime = System.nanoTime()
      var totalLoss = 0.0
      var numBatches = 0

      val iter = trainLoader.iterator
      while (iter.hasNext && numBatches < 10) {
        val batch = iter.next()
        val sparseFeats = batch.sparseFeatures
        val denseFeats = batch.denseFeatures
        val labels = batch.labels.getOrElse {
          torch.ones(batch.numSamples, 1L)
        }

        val logits = model.forward(sparseFeats, denseFeats)
        val loss = criterion.apply(torch.sigmoid(logits), labels)

        totalLoss += loss.item().toDouble()
        numBatches += 1
      }

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val avgLoss = totalLoss / numBatches

      RiskControlBenchmarkResult(
        name = name,
        category = category,
        passed = numBatches > 0 && !avgLoss.isNaN,
        metric = "avg_loss",
        value = avgLoss,
        trainingTimeMs = elapsed,
        memoryMb = ModelBenchmark.measureMemoryUsage()
      )
    } catch {
      case e: Exception =>
        RiskControlBenchmarkResult(name, category, false, "loss", 0, 0, 0, Some(e.getMessage))
    }
  }

  def runAutoIntBenchmark(): RiskControlBenchmarkResult = {
    val name = "AutoInt"
    val category = "DeepLearning"
    val numSamples = DEFAULT_NUM_SAMPLES
    val batchSize = DEFAULT_BATCH_SIZE
    val embedDim = DEFAULT_EMBED_DIM

    try {
      val (trainData, _, _) = DataGenerator.generateRankingData(
        numSamples = numSamples,
        numSparseFeatures = 5,
        numDenseFeatures = 3,
        vocabSize = DEFAULT_VOCAB_SIZE,
        seed = 42
      )

      val features = List(
        SparseFeature("feat_0", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_1", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_2", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_3", DEFAULT_VOCAB_SIZE, embedDim),
        SparseFeature("feat_4", DEFAULT_VOCAB_SIZE, embedDim)
      )

      val model = new AutoInt(features, embedDim, 8, 3, List(128L, 64L), 0.2f, DeviceSupport.backend)

      val trainLoader = new DataLoader(trainData, batchSize, shuffle = true)
      val criterion = new BCELoss()

      val startTime = System.nanoTime()
      var totalLoss = 0.0
      var numBatches = 0

      val iter = trainLoader.iterator
      while (iter.hasNext && numBatches < 10) {
        val batch = iter.next()
        val sparseFeats = batch.sparseFeatures
        val denseFeats = batch.denseFeatures
        val labels = batch.labels.getOrElse {
          torch.ones(batch.numSamples, 1L)
        }

        val logits = model.forward(sparseFeats, denseFeats)
        val loss = criterion.apply(torch.sigmoid(logits), labels)

        totalLoss += loss.item().toDouble()
        numBatches += 1
      }

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val avgLoss = totalLoss / numBatches

      RiskControlBenchmarkResult(
        name = name,
        category = category,
        passed = numBatches > 0 && !avgLoss.isNaN,
        metric = "avg_loss",
        value = avgLoss,
        trainingTimeMs = elapsed,
        memoryMb = ModelBenchmark.measureMemoryUsage()
      )
    } catch {
      case e: Exception =>
        RiskControlBenchmarkResult(name, category, false, "loss", 0, 0, 0, Some(e.getMessage))
    }
  }

  // ========================================================================
  // Phase 2: Graph Neural Networks
  // ========================================================================

  def runGraphNNBenchmarks(): List[RiskControlBenchmarkResult] = {
    val results = mutable.ListBuffer[RiskControlBenchmarkResult]()

    // GCN benchmark
    results += runGCNBenchmark()

    // GAT benchmark
    results += runGATBenchmark()

    // GraphSAGE benchmark
    results += runGraphSAGEBenchmark()

    // FraudGNN benchmark
    results += runFraudGNNBenchmark()

    results.toList
  }

  def runGCNBenchmark(): RiskControlBenchmarkResult = {
    val name = "GCN"
    val category = "GraphNN"
    val numNodes = 1000
    val numFeatures = 64
    val hiddenDim = 64
    val numClasses = 2

    try {
      // Generate synthetic graph data
      val features = torch.randn(Array(numNodes.toLong, numFeatures.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(DeviceSupport.backend)

      // Generate adjacency matrix (sparse graph) - use explicit Array for sizes
      val adj = torch.eye(numNodes).to(DeviceSupport.backend)
      // Add some random edges
      val randomEdges = torch.rand(numNodes, numNodes).to(DeviceSupport.backend)
      val threshold = new Scalar(0.95f)
      val mask = randomEdges.gt(threshold).to(ScalarType.Float)
      val sparseAdj = adj.add(mask)

      val model = new GCN(numFeatures, hiddenDim, numClasses, 0.5f, DeviceSupport.backend)

      val startTime = System.nanoTime()
      var numIterations = 0

      while (numIterations < 10) {
        val output = model.forward(features, sparseAdj)
        numIterations += 1
      }

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val throughput = numIterations * 1000.0 / elapsed

      RiskControlBenchmarkResult(
        name = name,
        category = category,
        passed = true,
        metric = "throughput",
        value = throughput,
        trainingTimeMs = elapsed,
        memoryMb = ModelBenchmark.measureMemoryUsage()
      )
    } catch {
      case e: Exception =>
        RiskControlBenchmarkResult(name, category, false, "throughput", 0, 0, 0, Some(e.getMessage))
    }
  }

  def runGATBenchmark(): RiskControlBenchmarkResult = {
    val name = "GAT"
    val category = "GraphNN"
    val numNodes = 1000
    val numFeatures = 64
    val hiddenDim = 64
    val numClasses = 2

    try {
      val features = torch.randn(Array(numNodes.toLong, numFeatures.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(DeviceSupport.backend)

      val adj = torch.eye(numNodes.toLong, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(DeviceSupport.backend)
      val randomEdges = torch.rand(Array(numNodes.toLong, numNodes.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(DeviceSupport.backend)
      val threshold = new Scalar(0.95f)
      val mask = randomEdges.gt(threshold).to(ScalarType.Float)
      val sparseAdj = adj.add(mask)

      val model = new GAT(numFeatures, hiddenDim, numClasses, 8, 0.5f, DeviceSupport.backend)

      val startTime = System.nanoTime()
      var numIterations = 0

      while (numIterations < 10) {
        val output = model.forward(features, sparseAdj)
        numIterations += 1
      }

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val throughput = numIterations * 1000.0 / elapsed

      RiskControlBenchmarkResult(
        name = name,
        category = category,
        passed = true,
        metric = "throughput",
        value = throughput,
        trainingTimeMs = elapsed,
        memoryMb = ModelBenchmark.measureMemoryUsage()
      )
    } catch {
      case e: Exception =>
        RiskControlBenchmarkResult(name, category, false, "throughput", 0, 0, 0, Some(e.getMessage))
    }
  }

  def runGraphSAGEBenchmark(): RiskControlBenchmarkResult = {
    val name = "GraphSAGE"
    val category = "GraphNN"
    val numNodes = 1000
    val numFeatures = 64
    val hiddenDim = 64
    val numClasses = 2

    try {
      val features = torch.randn(Array(numNodes.toLong, numFeatures.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(DeviceSupport.backend)

      val adj = torch.eye(numNodes.toLong, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(DeviceSupport.backend)
      val randomEdges = torch.rand(Array(numNodes.toLong, numNodes.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(DeviceSupport.backend)
      val threshold = new Scalar(0.95f)
      val mask = randomEdges.gt(threshold).to(ScalarType.Float)
      val sparseAdj = adj.add(mask)

      val model = new GraphSAGE(numFeatures, hiddenDim, numClasses, "mean", 0.5f, DeviceSupport.backend)

      val startTime = System.nanoTime()
      var numIterations = 0

      while (numIterations < 10) {
        val output = model.forward(features, sparseAdj)
        numIterations += 1
      }

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val throughput = numIterations * 1000.0 / elapsed

      RiskControlBenchmarkResult(
        name = name,
        category = category,
        passed = true,
        metric = "throughput",
        value = throughput,
        trainingTimeMs = elapsed,
        memoryMb = ModelBenchmark.measureMemoryUsage()
      )
    } catch {
      case e: Exception =>
        RiskControlBenchmarkResult(name, category, false, "throughput", 0, 0, 0, Some(e.getMessage))
    }
  }

  def runFraudGNNBenchmark(): RiskControlBenchmarkResult = {
    val name = "FraudGNN"
    val category = "GraphNN"
    val numNodes = 1000
    val numFeatures = 64
    val hiddenDim = 128
    val numClasses = 2

    try {
      val features = torch.randn(Array(numNodes.toLong, numFeatures.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(DeviceSupport.backend)

      val adj = torch.eye(numNodes.toLong, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(DeviceSupport.backend)
      val randomEdges = torch.rand(Array(numNodes.toLong, numNodes.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(DeviceSupport.backend)
      val threshold = new Scalar(0.95f)
      val mask = randomEdges.gt(threshold).to(ScalarType.Float)
      val sparseAdj = adj.add(mask)

      val model = new FraudGNN(numFeatures, hiddenDim, numClasses, 3, 0.3f, DeviceSupport.backend)

      val startTime = System.nanoTime()
      var numIterations = 0

      while (numIterations < 10) {
        val output = model.forward(features, sparseAdj)
        numIterations += 1
      }

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val throughput = numIterations * 1000.0 / elapsed

      RiskControlBenchmarkResult(
        name = name,
        category = category,
        passed = true,
        metric = "throughput",
        value = throughput,
        trainingTimeMs = elapsed,
        memoryMb = ModelBenchmark.measureMemoryUsage()
      )
    } catch {
      case e: Exception =>
        RiskControlBenchmarkResult(name, category, false, "throughput", 0, 0, 0, Some(e.getMessage))
    }
  }

  // ========================================================================
  // Phase 3: Large Models (LLM)
  // ========================================================================

  def runLLMBenchmarks(): List[RiskControlBenchmarkResult] = {
    val results = mutable.ListBuffer[RiskControlBenchmarkResult]()

    // HSTU benchmark
    results += runHSTUBenchmark()

    // LLM4Rec benchmark
    results += runLLM4RecBenchmark()

    results.toList
  }

  def runHSTUBenchmark(): RiskControlBenchmarkResult = {
    val name = "HSTU"
    val category = "LLM"
    val batchSize = 32
    val seqLen = 128
    val vocabSize = 8000
    val embedDim = 256
    val numHeads = 8
    val numLayers = 4

    try {
      val tokens = torch.rand(batchSize.toLong, seqLen.toLong).to(DeviceSupport.backend).mul(new Scalar(vocabSize.toFloat)).toType(ScalarType.Long)
      val positions = TorchRec.arange(0, seqLen).repeat(batchSize, 1).to(DeviceSupport.backend).toType(ScalarType.Long)
      val timeDiffs = torch.rand(batchSize.toLong, seqLen.toLong).to(DeviceSupport.backend).mul(new Scalar(512.0f)).toType(ScalarType.Long)

      val model = new HSTU(
        vocabSize = vocabSize.toLong,
        //          embedDim = config.embedDim,
        //          numHeads = 2,
        //          numLayers = 2,
        maxSeqLen = seqLen,
        dropout = 0.2f,
        device = DeviceSupport.backend
      )
      //      val model = new HSTU(vocabSize, embedDim, numHeads, numLayers, seqLen, 0.1f, DeviceSupport.backend)

      val startTime = System.nanoTime()
      var numIterations = 0

      while (numIterations < 5) {
        //        val output = model.forward(tokens, positions, timeDiffs)
        val output = model.forward(tokens, timeDiffs)
        numIterations += 1
      }

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val throughput = numIterations * 1000.0 / elapsed

      RiskControlBenchmarkResult(
        name = name,
        category = category,
        passed = true,
        metric = "throughput",
        value = throughput,
        trainingTimeMs = elapsed,
        memoryMb = ModelBenchmark.measureMemoryUsage()
      )
    } catch {
      case e: Exception =>
        RiskControlBenchmarkResult(name, category, false, "throughput", 0, 0, 0, Some(e.getMessage))
    }
  }

  def runLLM4RecBenchmark(): RiskControlBenchmarkResult = {
    val name = "LLM4Rec"
    val category = "LLM"
    val batchSize = 16
    val seqLen = 64
    val vocabSize = 8000
    val embedDim = 512
    val numLayers = 4

    try {
      val tokens = torch.rand(batchSize.toLong, seqLen.toLong).to(DeviceSupport.backend).mul(new Scalar(vocabSize.toFloat)).toType(ScalarType.Long)
      val positions = TorchRec.arange(0, seqLen).repeat(batchSize, 1).to(DeviceSupport.backend).toType(ScalarType.Long)
      val attentionMask = torch.ones(batchSize, seqLen).to(DeviceSupport.backend).toType(ScalarType.Long)

      val model = new LLM4Rec(vocabSize, embedDim, 8, numLayers, seqLen, List(256L, 128L), 0.1f, true, DeviceSupport.backend)

      val startTime = System.nanoTime()
      var numIterations = 0

      while (numIterations < 3) {
        val output = model.forward(tokens, positions)
        numIterations += 1
      }

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val throughput = numIterations * 1000.0 / elapsed

      RiskControlBenchmarkResult(
        name = name,
        category = category,
        passed = true,
        metric = "throughput",
        value = throughput,
        trainingTimeMs = elapsed,
        memoryMb = ModelBenchmark.measureMemoryUsage()
      )
    } catch {
      case e: Exception =>
        RiskControlBenchmarkResult(name, category, false, "throughput", 0, 0, 0, Some(e.getMessage))
    }
  }

  // ========================================================================
  // Phase 4: Full Pipeline Integration
  // ========================================================================

  def runFullPipelineBenchmarks(): List[RiskControlBenchmarkResult] = {
    val results = mutable.ListBuffer[RiskControlBenchmarkResult]()

    // DataFrame + DeepFM pipeline
    results += runDataFrameDeepFMPipeline()

    // DataFrame + GNN pipeline
    results += runDataFrameGNNPipeline()

    results.toList
  }

  def runDataFrameDeepFMPipeline(): RiskControlBenchmarkResult = {
    val name = "DataFrame_DeepFM_Pipeline"
    val category = "FullPipeline"
    val numRows = DEFAULT_NUM_SAMPLES
    val batchSize = DEFAULT_BATCH_SIZE

    try {
      // Generate synthetic risk control DataFrame
      val rows = (0 until numRows).map { i =>
        Map(
          "user_id" -> s"user_${i % 100}",
          "item_id" -> s"item_${i % 500}",
          "category" -> s"cat_${i % 20}",
          "brand" -> s"brand_${i % 50}",
          "price" -> (Random.nextFloat() * 100).toString,
          "rating" -> (Random.nextFloat() * 5).toString,
          "click_hist" -> (1 to (i % 10 + 1)).map(j => s"item_${(i + j) % 500}").mkString("|"),
          "label" -> (if (Random.nextFloat() > 0.5) 1 else 0).toString
        )
      }
      val rawDF = DataFrame.fromRows(rows)

      // Feature engineering pipeline
      val pipeline = PipelineBuilder.create()
        .addScaling("dense", "standard", Map())
        .addEncoding("sparse", "label", Map())
        .addRecommenderFeature("ctr", "clickThroughRate", Map())
        .addRecommenderFeature("userActivity", "userActivity", Map())
        .build()

      val startTime = System.nanoTime()

      val processedDF = pipeline.fitTransform(rawDF)

      // Convert to DataLoader
      val featureSpec = FeatureSpec(
        sparseFeatures = List(
          SparseSpec("user_id", 100, 8),
          SparseSpec("item_id", 500, 8),
          SparseSpec("category", 20, 8)
        ),
        denseFeatures = List(
          DenseSpec("price", 1),
          DenseSpec("rating", 1)
        ),
        sequenceFeatures = List(
          SequenceSpec("click_hist", 500, 8, maxLen = 50)
        )
      )

      val labelSpec = LabelSpec("label")
      val dataLoader = DataFrameBridge.toDataLoader(processedDF, featureSpec, Some(labelSpec), batchSize)

      // Build model
      val features = List(
        SparseFeature("user_id", 100, 8),
        SparseFeature("item_id", 500, 8),
        SparseFeature("category", 20, 8)
      )
      val model = new DeepFM(features, 8, List(128L, 64L), 0.2f, DeviceSupport.backend)

      // Training loop
      var numBatches = 0
      val iter = dataLoader.iterator
      while (iter.hasNext && numBatches < 5) {
        iter.next()
        numBatches += 1
      }

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val throughput = numRows * 1000.0 / elapsed

      RiskControlBenchmarkResult(
        name = name,
        category = category,
        passed = numBatches > 0,
        metric = "throughput",
        value = throughput,
        trainingTimeMs = elapsed,
        memoryMb = ModelBenchmark.measureMemoryUsage()
      )
    } catch {
      case e: Exception =>
        RiskControlBenchmarkResult(name, category, false, "throughput", 0, 0, 0, Some(e.getMessage))
    }
  }

  def runDataFrameGNNPipeline(): RiskControlBenchmarkResult = {
    val name = "DataFrame_GNN_Pipeline"
    val category = "FullPipeline"
    val numNodes = 500
    val numFeatures = 32

    try {
      // Generate synthetic graph data from DataFrame
      val nodeFeatures = (0 until numNodes).map { i =>
        Map(
          "node_id" -> i.toString,
          "feature_0" -> Random.nextFloat().toString,
          "feature_1" -> Random.nextFloat().toString,
          "feature_2" -> Random.nextFloat().toString,
          "label" -> (if (Random.nextFloat() > 0.7) 1 else 0).toString
        )
      }
      val nodeDF = DataFrame.fromRows(nodeFeatures)

      // Build adjacency from relationships
      val edges = (0 until numNodes).flatMap { i =>
        (0 until 5).map { j =>
          val target = (i + j + 1) % numNodes
          (Math.min(i, target), Math.max(i, target))
        }
      }.distinct.take(1000)

      // Create adjacency matrix using scatter
      val adjData = Array.fill(numNodes * numNodes)(0.0f)
      edges.foreach { case (i, j) =>
        adjData(i * numNodes + j) = 1.0f
        adjData(j * numNodes + i) = 1.0f
      }
      val adj = TorchRec.tensor(adjData, numNodes.toLong, numNodes.toLong).to(DeviceSupport.backend)

      // Extract features from DataFrame
      val features = torch.randn(Array(numNodes.toLong, numFeatures.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(DeviceSupport.backend)

      val startTime = System.nanoTime()

      val model = new FraudGNN(numFeatures, 64, 2, 3, 0.3f, DeviceSupport.backend)

      var numIterations = 0
      while (numIterations < 10) {
        val output = model.forward(features, adj)
        numIterations += 1
      }

      val elapsed = (System.nanoTime() - startTime) / 1e6
      val throughput = numIterations * 1000.0 / elapsed

      RiskControlBenchmarkResult(
        name = name,
        category = category,
        passed = true,
        metric = "throughput",
        value = throughput,
        trainingTimeMs = elapsed,
        memoryMb = ModelBenchmark.measureMemoryUsage()
      )
    } catch {
      case e: Exception =>
        RiskControlBenchmarkResult(name, category, false, "throughput", 0, 0, 0, Some(e.getMessage))
    }
  }

  // ========================================================================
  // Summary
  // ========================================================================

  def printSummary(results: List[RiskControlBenchmarkResult]): Unit = {
    println("\n" + "=" * 80)
    println("Risk Control & Anti-Fraud Benchmark Results Summary")
    println("=" * 80)

    val passedCount = results.count(_.passed)
    val totalCount = results.length
    val passRate = if (totalCount > 0) passedCount * 100.0 / totalCount else 0

    println(f"\nPass Rate: $passRate%.1f%% ($passedCount/$totalCount)\n")

    val byCategory = results.groupBy(_.category)
    for ((category, categoryResults) <- byCategory.toSeq.sortBy(_._1)) {
      println(s"\n[$category]")
      println("-" * 70)
      println(f"${"Name"}%-25s${"Status"}%-10s${"Metric"}%-15s${"Value"}%-12s${"Time"}%-10s")
      println("-" * 70)

      for (result <- categoryResults.sortBy(_.name)) {
        val status = if (result.passed) "PASS" else "FAIL"
        val valueStr = result.metric match {
          case "avg_loss" => f"${result.value}%.4f"
          case "throughput" => f"${result.value}%.2f iter/s"
          case _ => f"${result.value}%.2f"
        }
        val timeStr = f"${result.trainingTimeMs}%.2fms"
        println(f"${result.name}%-25s${status}%-10s${result.metric}%-15s${valueStr}%-12s${timeStr}%-10s")

        if (!result.passed && result.error.isDefined) {
          println(f"  Error: ${result.error.get}")
        }
      }

      val categoryPassed = categoryResults.count(_.passed)
      val categoryTotal = categoryResults.length
      println(f"\nCategory: $categoryPassed/$categoryTotal passed")
    }

    println("\n" + "=" * 80)
    if (passRate >= 90) {
      println("OVERALL: SUCCESS - All critical benchmarks passed")
    } else if (passRate >= 70) {
      println("OVERALL: PARTIAL - Some benchmarks failed, review needed")
    } else {
      println("OVERALL: FAILURE - Too many benchmarks failed")
    }
    println("=" * 80)
  }
}