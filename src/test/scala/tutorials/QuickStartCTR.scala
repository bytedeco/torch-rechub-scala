package tutorials

import torchrec.Implicits._
import torchrec.basic.features._
import torchrec.data._
import benchmarks.DataGenerator
import torchrec.models.ranking._
import torchrec.trainers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * Quick Start Tutorial: CTR Prediction with DeepFM
 *
 * This tutorial demonstrates the basic workflow for building a CTR prediction model:
 * 1. Prepare data
 * 2. Define features
 * 3. Create model
 * 4. Train
 * 5. Evaluate
 */
object QuickStartCTR {

  def main(args: Array[String]): Unit = {
    DeviceSupport.setDevice(DeviceSupport.DeviceType.AUTO)
    val device = DeviceSupport.backend
    println(s"[DeviceSupport] Active device: $device")
    println(
      """
        |=============================================================
        | Quick Start: Click-Through Rate Prediction with DeepFM
        |=============================================================
        |
        | This tutorial walks you through the basic workflow for
        | building a CTR prediction model using DeepFM.
        |
        |""".stripMargin)

    // Step 1: Prepare data
    println("Step 1: Preparing data...")
    val (trainData, valData, testData) = DataGenerator.generateRankingData(
      numSamples = 1000,
      numSparseFeatures = 3,
      numDenseFeatures = 0,
      vocabSize = 100,
      seed = 2024,
      featureNames = Seq("user_id", "item_id", "category")
    )
    println(f"  Generated ${trainData.size}%,d training samples")
    println(f"  Generated ${valData.size}%,d validation samples")

    // Step 2: Define features
    println("\nStep 2: Defining features...")
    val features = List(
      SparseFeature("user_id", 100, 8),
      SparseFeature("item_id", 100, 8),
      SparseFeature("category", 100, 4)
    )
    features.foreach(f => println(f"  - ${f.name}: vocab=${f.vocabSize}, embed=${f.embedDim}"))

    // Step 3: Create model
    println("\nStep 3: Creating DeepFM model...")
    val model = new DeepFM(
      features = features,
      embedDim = 8,
      mlpDims = List(64L, 32L),
      dropout = 0.2f,
      device = device
    )
    println("  Model: DeepFM(embed_dim=8, mlp_dims=[64, 32])")

    // Step 4: Create data loaders
    println("\nStep 4: Creating data loaders...")
    val trainLoader = new DataLoader(trainData, batchSize = 128, shuffle = true, device = device)
    val valLoader = new DataLoader(valData, batchSize = 128, shuffle = false, device = device)
    println(f"  Train batches: ${trainData.size / 128}")
    println(f"  Val batches: ${valData.size / 128}")

    // Step 5: Train
    println("\nStep 5: Training...")
    val trainer = new CTRTrainer(
      model = model,
      learningRate = 1e-3f,
      device = device,
      numEpochs = 3,
      verbose = true
    )
    trainer.fit(trainLoader, Some(valLoader))

    // Step 6: Evaluate
    println("\nStep 6: Evaluating...")
    val metrics = trainer.evaluate(valLoader)
    println("  Validation Metrics:")
    metrics.foreach { case (name, value) =>
      println(f"    $name: $value%.4f")
    }

    println(
      """
        |
        |=============================================================
        | Quick Start Complete!
        |=============================================================
        |
        | Next steps:
        | - Try other models (DCN, WideDeep, DIN)
        | - Use real datasets (Criteo, Avazu)
        | - Tune hyperparameters
        |
        |""".stripMargin)
  }
}