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
 * DeepFM Example for CTR Prediction
 */
object DeepFMExample {

  def main(args: Array[String]): Unit = {
    DeviceSupport.setDevice(DeviceSupport.DeviceType.AUTO)
    val device = DeviceSupport.backend
    println(s"[DeviceSupport] Active device: $device")

    println("=" * 60)
    println("DeepFM CTR Prediction Example")
    println("=" * 60)

    // Configuration
    val batchSize = 256
    val embedDim = 8
    val numEpochs = 3
    val vocabSize = 100
    val numSparseFeatures = 10

    println(s"\nConfiguration:")
    println(s"  Device: $device")
    println(s"  Batch Size: $batchSize")
    println(s"  Embed Dim: $embedDim")
    println(s"  Epochs: $numEpochs")

    // Define features — names must match DataGenerator output (sparse_0, sparse_1, ...)
    println("\n[1] Defining features...")
    val features = (0 until numSparseFeatures).map { i =>
      SparseFeature(name = s"sparse_$i", vocabSize = vocabSize, embedDim = embedDim)
    }.toList
    features.foreach { f =>
      println(s"  - ${f.name}: vocab=${f.vocabSize}, embed=${f.embedDim}")
    }

    // Create model
    println("\n[2] Creating DeepFM model...")
    val halfIdx = numSparseFeatures / 2
    val model = new DeepFM(
      deepFeatures = features.take(halfIdx),
      fmFeatures = features.drop(halfIdx),
      embedDim = embedDim,
      mlpDims = List(128L, 64L),
      dropout = 0.2f,
      device = device
    )
    println(s"  Model created: DeepFM(embedDim=$embedDim)")

    // Generate synthetic data using DataGenerator
    println("\n[3] Generating synthetic data...")
    val (trainData, valData, testData) = benchmarks.DataGenerator.generateRankingData(
      numSamples = 1000,
      numSparseFeatures = numSparseFeatures,
      numDenseFeatures = 0,
      vocabSize = vocabSize,
      seed = 42
    )
    println(s"  Generated ${trainData.size} training samples")
    println(s"  Generated ${valData.size} validation samples")

    // Create data loaders — pass device so tensors land on the right place
    println("\n[4] Creating data loaders...")
    val trainLoader = new DataLoader(trainData, batchSize, shuffle = true, device = device)
    val valLoader = new DataLoader(valData, batchSize, shuffle = false, device = device)
    val testLoader = new DataLoader(testData, batchSize, shuffle = false, device = device)
    println(s"  Train batches: ${trainData.size / batchSize}")

    // Train using CTRTrainer
    println("\n[5] Training model...")
    val startTime = System.currentTimeMillis()
    val trainer = new CTRTrainer(
      model = model,
      learningRate = 1e-3f,
      device = device,
      numEpochs = numEpochs,
      verbose = true
    )
    trainer.fit(trainLoader, Some(valLoader))
    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
    println(f"\n  Training completed in $trainingTime%.2f seconds")

    // Evaluate
    println("\n[6] Evaluating...")
    val testMetrics = trainer.evaluate(testLoader)
    println("  Test metrics:")
    testMetrics.foreach { case (name, value) =>
      println(f"    $name: $value%.4f")
    }

    // Predict
    println("\n[7] Making predictions...")
    val predictions = trainer.predict(testLoader)
    println(s"  Predicted ${predictions.length} samples")
    if (predictions.nonEmpty) {
      println(s"  Sample predictions: ${predictions.take(5).map(v => f"$v%.3f").mkString(", ")}")
    }

    println("\n" + "=" * 60)
    println("DeepFM Example Completed Successfully!")
    println("=" * 60)
  }
}
