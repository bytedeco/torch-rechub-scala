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
 * 调试四个模型的维度问题
 */
object DebugModels {
  def main(args: Array[String]): Unit = {
    println("=" * 60)
    println("Debugging AFM, DeepFM, DCN, EDCN dimensions")
    println("=" * 60)

    val numSamples = 256  // Use smaller for debugging
    val batchSize = 256
    val numSparse = 10
    val vocabSize = 100
    val embedDim = 8
    val device = "cpu"  // Use CPU for easier debugging

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

    // Calculate expected dimensions
    val sparseDim = Features.calcSparseDim(features)
    println(s"numSparse=$numSparse, embedDim=$embedDim")
    println(s"Expected sparseDim = $sparseDim")
    println(s"deepFeatures.size = ${deepFeatures.size}, fmFeatures.size = ${fmFeatures.size}")

    // Test each model
    println("\n--- Debugging AFM ---")
    debugAFM(features, trainLoader, embedDim, device)

    println("\n--- Debugging DeepFM ---")
    debugDeepFM(deepFeatures, fmFeatures, trainLoader, embedDim, device)

    println("\n--- Debugging DCN ---")
    debugDCN(features, trainLoader, embedDim, device)

    println("\n--- Debugging EDCN ---")
    debugEDCN(features, trainLoader, embedDim, device)

    println("\n" + "=" * 60)
    println("Debug completed!")
    println("=" * 60)
  }

  def debugAFM(features: List[SparseFeature], trainLoader: DataLoader, embedDim: Int, device: String): Unit = {
    try {
      val numFields = features.size
      val fmDims = features.map(_.embedDim).sum
      println(s"AFM: numFields=$numFields, fmDims=$fmDims, embedDim=$embedDim")
      println(s"Expected linear input: ${numFields * embedDim}")
      println(s"Expected FM output: ($embedDim)")
      println(s"Expected attention h: ($embedDim, 1)")
      println(s"Expected attention p: ($embedDim, 1)")

      val model = new AFM(features, embedDim, 64, 0.2f, device)
      val trainer = new CTRTrainer(model, learningRate = 1e-3f, device = device, numEpochs = 1, verbose = false)
      trainer.fit(trainLoader, None)
      println("  [PASS] AFM")
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] AFM: ${e.getClass.getSimpleName}: ${e.getMessage}")
    }
  }

  def debugDeepFM(deepFeatures: List[Feature], fmFeatures: List[Feature], trainLoader: DataLoader, embedDim: Int, device: String): Unit = {
    try {
      val deepDims = deepFeatures.map(_.embedDim).sum
      val fmDimsCalc = fmFeatures.map(_.embedDim).sum
      println(s"DeepFM: deepFeatures.size=${deepFeatures.size}, fmFeatures.size=${fmFeatures.size}")
      println(s"deepDims=$deepDims, fmDims=$fmDimsCalc")
      println(s"Expected deep MLP input: $deepDims")
      println(s"Expected linear input: $fmDimsCalc")

      val model = new DeepFM(deepFeatures, fmFeatures, embedDim, List(64L, 32L), 0.2f, device)
      val trainer = new CTRTrainer(model, learningRate = 1e-3f, device = device, numEpochs = 1, verbose = false)
      trainer.fit(trainLoader, None)
      println("  [PASS] DeepFM")
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] DeepFM: ${e.getClass.getSimpleName}: ${e.getMessage}")
    }
  }

  def debugDCN(features: List[Feature], trainLoader: DataLoader, embedDim: Int, device: String): Unit = {
    try {
      val sparseDimCalc = Features.calcSparseDim(features)
      println(s"DCN: features.size=${features.size}, embedDim=$embedDim")
      println(s"Calculated sparseDim = $sparseDimCalc")
      println(s"Expected MLP input: $sparseDimCalc")
      println(s"MLP dims: List(64, 32), last=${32L}")
      println(s"Expected combo input: ${sparseDimCalc} + 32 = ${sparseDimCalc + 32}")

      val model = new DCN(features, embedDim, 2, List(64L, 32L), 0.2f, device)
      val trainer = new CTRTrainer(model, learningRate = 1e-3f, device = device, numEpochs = 1, verbose = false)
      trainer.fit(trainLoader, None)
      println("  [PASS] DCN")
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] DCN: ${e.getClass.getSimpleName}: ${e.getMessage}")
    }
  }

  def debugEDCN(features: List[Feature], trainLoader: DataLoader, embedDim: Int, device: String): Unit = {
    try {
      val dims = features.map(_.embedDim).sum
      println(s"EDCN: features.size=${features.size}, dims=$dims")

      val mlpParams = Map("dims" -> List(64L, 32L), "activation" -> "relu", "dropout" -> 0.2f)
      val model = new EDCN(features, nCrossLayers = 2, mlpParams = mlpParams, bridgeType = "hadamard_product", useRegulationModule = true, temperature = 0.2f, device = device)
      val trainer = new CTRTrainer(model, learningRate = 1e-3f, device = device, numEpochs = 1, verbose = false)
      trainer.fit(trainLoader, None)
      println("  [PASS] EDCN")
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] EDCN: ${e.getClass.getSimpleName}: ${e.getMessage}")
    }
  }
}