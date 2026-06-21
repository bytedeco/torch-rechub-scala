package examples.ranking

import torchrec.Implicits.*
import torchrec.data.*
import torchrec.basic.features.*
import torchrec.models.ranking.*
import torchrec.trainers.*
import torchrec.utils.DeviceSupport
import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch

/**
 * Criteo CTR Prediction Example
 * Demonstrates training various ranking models (DeepFM, WideDeep, DCN, DCNv2,
 * FiBiNet, EDCN, AutoInt, DeepFFM) on Criteo-like data.
 */
object CriteoExample {

  def main(args: Array[String]): Unit = {
    DeviceSupport.setDevice(DeviceSupport.DeviceType.AUTO)
    val device = DeviceSupport.backend
    println(s"[DeviceSupport] Active device: $device")

    println("=" * 70)
    println("Criteo CTR Prediction Example - Ranking Models")
    println("=" * 70)

    // Configuration
    val numSamples = 2000
    val numSparseFeatures = 13
    val numDenseFeatures = 13
    val vocabSize = 1000
    val embedDim = 16
    val mlpDims = List(256L, 128L)
    val batchSize = 256
    val learningRate = 1e-3f
    val weightDecay = 1e-4f
    val numEpochs = 3
    val seed = 2022

    val modelName = if (args.length > 0) args(0).toLowerCase else "deepfm"

    println(s"\nConfiguration:")
    println(s"  Model: $modelName")
    println(s"  Device: $device")
    println(s"  Samples: $numSamples")
    println(s"  Sparse Features: $numSparseFeatures")
    println(s"  Dense Features: $numDenseFeatures")
    println(s"  Embed Dim: $embedDim")
    println(s"  MLP Dims: $mlpDims")
    println(s"  Batch Size: $batchSize")
    println(s"  Learning Rate: $learningRate")
    println(s"  Weight Decay: $weightDecay")
    println(s"  Epochs: $numEpochs")

    // Generate synthetic Criteo-like data
    println("\n[1] Generating synthetic Criteo-like data...")
    val (trainData, valData, testData) = benchmarks.DataGenerator.generateRankingData(
      numSamples = numSamples,
      numSparseFeatures = numSparseFeatures,
      numDenseFeatures = numDenseFeatures,
      vocabSize = vocabSize,
      trainRatio = 0.7f,
      valRatio = 0.1f,
      seed = seed
    )
    println(s"  Train: ${trainData.size} samples")
    println(s"  Val: ${valData.size} samples")
    println(s"  Test: ${testData.size} samples")

    // Create data loaders
    println("\n[2] Creating data loaders...")
    val trainLoader = new DataLoader(trainData, batchSize, shuffle = true, device = device)
    val valLoader = new DataLoader(valData, batchSize, shuffle = false, device = device)
    val testLoader = new DataLoader(testData, batchSize, shuffle = false, device = device)
    println(s"  Train batches: ${trainData.size / batchSize}")
    println(s"  Val batches: ${valData.size / batchSize}")
    println(s"  Test batches: ${testData.size / batchSize}")

    // Define features
    println("\n[3] Defining features...")
    val denseFeatureNames = (0 until numDenseFeatures).map(i => s"dense_$i").toList
    val sparseFeatureNames = (0 until numSparseFeatures).map(i => s"sparse_$i").toList

    val denseFeatures = denseFeatureNames.map(name => DenseFeature(name = name))
    val sparseFeatures = sparseFeatureNames.map { name =>
      SparseFeature(name = name, vocabSize = vocabSize, embedDim = embedDim)
    }
    val allFeatures = denseFeatures ++ sparseFeatures

    // FFM features for DeepFFM
    val fieldNum = sparseFeatureNames.size
    val ffmCrossFeatures = sparseFeatureNames.map { name =>
      SparseFeature(name = name, vocabSize = vocabSize * fieldNum, embedDim = 10)
    }

    println(s"  Dense features: ${denseFeatures.size}")
    println(s"  Sparse features: ${sparseFeatures.size}")
    println(s"  Total features: ${allFeatures.size}")

    // Create model
    println(s"\n[4] Creating model: $modelName...")
    val model: Module = modelName match {
      case "widedeep" | "wide-deep" =>
        new WideDeep(
          features = allFeatures,
          embedDim = embedDim,
          mlpDims = mlpDims,
          dropout = 0.2f,
          device = device
        )

      case "deepfm" | "" =>
        val halfIdx = allFeatures.size / 2
        new DeepFM(
          deepFeatures = allFeatures.take(halfIdx),
          fmFeatures = allFeatures.drop(halfIdx),
          embedDim = embedDim,
          mlpDims = mlpDims,
          dropout = 0.2f,
          device = device
        )

      case "dcn" =>
        new DCN(
          features = allFeatures,
          embedDim = embedDim,
          numCrossLayers = 3,
          mlpDims = mlpDims,
          dropout = 0.2f,
          device = device
        )

      case "dcn_v2" | "dcnv2" =>
        new DCNv2(
          features = allFeatures,
          embedDim = embedDim,
          numCrossLayers = 3,
          useCrossNetMix = true,
          lowRank = 4,
          mlpDims = mlpDims,
          dropout = 0.2f,
          device = device
        )

      case "fibinet" | "fibi" =>
        new FiBiNet(
          features = allFeatures,
          embedDim = embedDim,
          mlpDims = mlpDims,
          reduction = 3,
          bilinearType = "field_all",
          dropout = 0.2f,
          device = device
        )

      case "edcn" =>
        val mlpParams = Map("dims" -> mlpDims, "activation" -> "relu", "dropout" -> 0.2f)
        new EDCN(
          features = allFeatures,
          nCrossLayers = 3,
          mlpParams = mlpParams,
          bridgeType = "add",
          useRegulationModule = true,
          temperature = 1.0f,
          device = device
        )

      case "autoint" =>
        new AutoInt(
          sparseFeatures = allFeatures,
          embedDim = embedDim,
          numAttnHeads = 2,
          numLayers = 2,
          mlpDims = mlpDims,
          dropout = 0.2f,
          useMlp = true,
          device = device
        )

      case "deepffm" =>
        new DeepFFM(
          features = allFeatures,
          embedDim = 10,
          fieldNum = fieldNum,
          mlpDims = mlpDims,
          dropout = 0.2f,
          device = device
        )

      case "fat_deepffm" | "fatdeepffm" =>
        new FatDeepFFM(
          features = allFeatures,
          embedDim = 10,
          mlpDims = mlpDims,
          dropout = 0.2f,
          device = device
        )

      case "xdeepfm" | "xdeep_fm" =>
        new xDeepFM(
          features = allFeatures,
          embedDim = embedDim,
          crossLayerSizes = List(128, 64),
          mlpDims = mlpDims,
          splitHalf = true,
          dropout = 0.2f,
          device = device
        )

      case "nfm" =>
        new NFM(
          features = allFeatures,
          embedDim = embedDim,
          mlpDims = mlpDims,
          dropout = 0.2f,
          device = device
        )

      case "fnn" =>
        new FNN(
          features = allFeatures,
          embedDim = embedDim,
          mlpDims = mlpDims,
          dropout = 0.2f,
          device = device
        )

      case "fnfm" =>
        new FNFM(
          features = allFeatures,
          embedDim = embedDim,
          mlpDims = mlpDims,
          dropout = 0.2f,
          device = device
        )

      case _ =>
        println(s"  Warning: Unknown model '$modelName', defaulting to DeepFM")
        val halfIdx = allFeatures.size / 2
        new DeepFM(
          deepFeatures = allFeatures.take(halfIdx),
          fmFeatures = allFeatures.drop(halfIdx),
          embedDim = embedDim,
          mlpDims = mlpDims,
          dropout = 0.2f,
          device = device
        )
    }

    // Train
    println("\n[5] Training model...")
    val startTime = System.currentTimeMillis()
    val trainer = new CTRTrainer(
      model = model,
      learningRate = learningRate,
      weightDecay = weightDecay,
      device = device,
      numEpochs = numEpochs,
      earlyStopPatience = 3,
      verbose = true
    )
    trainer.fit(trainLoader, Some(valLoader))
    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
    println(f"\n  Training completed in $trainingTime%.2f seconds")

    // Evaluate
    println("\n[6] Evaluating on test set...")
    val testMetrics = trainer.evaluate(testLoader)
    println("  Test metrics:")
    testMetrics.foreach { case (name, value) =>
      println(f"    $name: $value%.4f")
    }

    // Predict
    println("\n[7] Making predictions...")
    val predictions = trainer.predict(testLoader)
    println(s"  Predicted ${predictions.length} samples")
    if (predictions.length > 0) {
      println(s"  Sample predictions: ${predictions.take(5).map(v => f"$v%.3f").mkString(", ")}")
    }

    println("\n" + "=" * 70)
    println(s"Criteo Example ($modelName) Completed Successfully!")
    println("=" * 70)
  }
}
