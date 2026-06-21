package benchmarks

import torchrec.basic.features.*
import torchrec.data.*
import torchrec.models.ranking.*
import torchrec.trainers.*
import torchrec.utils.DeviceSupport
import torchrec.Implicits.tensor
import torchrec.Implicits.*
import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.util.Random

/**
 * 测试四个模型的简单脚本
 */
object TestModels {
  def main(args: Array[String]): Unit = {
    println("=" * 60)
    println("Testing AFM, DeepFM, DCN, EDCN")
    println("=" * 60)

    val numSamples = 1000
    val batchSize = 256
    val numSparse = 10
    val vocabSize = 100
    val embedDim = 8
    val device = DeviceSupport.backend

    val random = new Random(42)

    // Generate data
    val (trainData, valData, _) = torchrec.data.DataGenerator.generateRankingData(
      numSamples = numSamples,
      numSparseFeatures = numSparse,
      numDenseFeatures = 5,
      vocabSize = vocabSize,
      trainRatio = 0.7f,
      valRatio = 0.1f,
      seed = 42
    )

    val trainLoader = new DataLoader(trainData, batchSize, shuffle = true)

    // Create features
    val features = (0 until numSparse).map { i =>
      SparseFeature(s"feat_$i", vocabSize, embedDim)
    }.toList

    val halfIdx = numSparse / 2
    val deepFeatures = features.take(halfIdx)
    val fmFeatures = features.drop(halfIdx)


    println("\n--- Testing DCN ---")
    testDCN(features, trainLoader, embedDim, device)

    println("\n--- Testing EDCN ---")
    testEDCN(features, trainLoader, embedDim, device)
    // Test each model
    println("\n--- Testing AFM ---")
    testAFM(features, trainLoader, embedDim, device)

    println("\n--- Testing DeepFM ---")
    testDeepFM(deepFeatures, fmFeatures, trainLoader, embedDim, device)

    println("\n" + "=" * 60)
    println("All tests completed!")
    println("=" * 60)
  }

  def testAFM(features: List[SparseFeature], trainLoader: DataLoader, embedDim: Int, device: String): Unit = {
    try {
      val model = new AFM(features, embedDim, 64, 0.2f, device)
      val trainer = new CTRTrainer(model, learningRate = 1e-3f, device = device, numEpochs = 1, verbose = false)
      trainer.fit(trainLoader, None)
      println("  [PASS] AFM")
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] AFM: ${e.getClass.getSimpleName}: ${e.getMessage}")
        e.printStackTrace()
    }
  }

  def testDeepFM(deepFeatures: List[Feature], fmFeatures: List[Feature], trainLoader: DataLoader, embedDim: Int, device: String): Unit = {
    try {
      val model = new DeepFM(deepFeatures, fmFeatures, embedDim, List(64L, 32L), 0.2f, device)
      val trainer = new CTRTrainer(model, learningRate = 1e-3f, device = device, numEpochs = 1, verbose = false)
      trainer.fit(trainLoader, None)
      println("  [PASS] DeepFM")
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] DeepFM: ${e.getClass.getSimpleName}: ${e.getMessage}")
        e.printStackTrace()
    }
  }

  def testDCN(features: List[Feature], trainLoader: DataLoader, embedDim: Int, device: String): Unit = {
    try {
      val model = new DCN(features, embedDim, 2, List(64L, 32L), 0.2f, device)
      val trainer = new CTRTrainer(model, learningRate = 1e-3f, device = device, numEpochs = 1, verbose = false)
      trainer.fit(trainLoader, None)
      println("  [PASS] DCN")
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] DCN: ${e.getClass.getSimpleName}: ${e.getMessage}")
        e.printStackTrace()
    }
  }

  def testEDCN(features: List[Feature], trainLoader: DataLoader, embedDim: Int, device: String): Unit = {
    try {
      val mlpParams = Map("dims" -> List(64L, 32L), "activation" -> "relu", "dropout" -> 0.2f)
      val model = new EDCN(features, nCrossLayers = 2, mlpParams = mlpParams, bridgeType = "hadamard_product", useRegulationModule = true, temperature = 0.2f, device = device)
      val trainer = new CTRTrainer(model, learningRate = 1e-3f, device = device, numEpochs = 1, verbose = false)
      trainer.fit(trainLoader, None)
      println("  [PASS] EDCN")
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] EDCN: ${e.getClass.getSimpleName}: ${e.getMessage}")
        e.printStackTrace()
    }
  }
}