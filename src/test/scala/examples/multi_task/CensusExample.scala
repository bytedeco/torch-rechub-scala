package examples.multi_task

import torchrec.Implicits._
import torchrec.data._
import torchrec.data.DataGenerator
import torchrec.models.multi_task._
import torchrec.trainers._
import torchrec.basic.features._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * Census Multi-Task Learning Example
 * Demonstrates training multi-task models (SharedBottom, ESMM, MMOE, PLE, AITM)
 * on Census-like data with multiple prediction tasks.
 *
 * Task 1 (CVR): Income prediction - main task
 * Task 2 (CTR): Marital status prediction - auxiliary task
 */
object CensusExample {

  def main(args: Array[String]): Unit = {
    println("=" * 70)
    println("Census Multi-Task Learning Example")
    println("=" * 70)

    // Configuration
    val numSamples = 2000
    val numFeatures = 30
    val taskNames = List("cvr", "ctr") // cvr = income, ctr = marital status
    val vocabSize = 100
    val embedDim = 8
    val batchSize = 256
    val learningRate = 1e-3f
    val weightDecay = 1e-4f
    val numEpochs = 3
    val device = "cpu"
    val seed = 2022

    // Parse model name from args (default: mmoe)
    val modelName = if (args.length > 0) args(0).toLowerCase else "mmoe"

    println(s"\nConfiguration:")
    println(s"  Model: $modelName")
    println(s"  Samples: $numSamples")
    println(s"  Features: $numFeatures")
    println(s"  Tasks: ${taskNames.mkString(", ")}")
    println(s"  Embed Dim: $embedDim")
    println(s"  Batch Size: $batchSize")
    println(s"  Learning Rate: $learningRate")
    println(s"  Weight Decay: $weightDecay")
    println(s"  Epochs: $numEpochs")

    // 1. Generate synthetic Census-like multi-task data
    println("\n[1] Generating synthetic Census-like multi-task data...")
    val (trainData, valData, testData) = DataGenerator.generateMultiTaskData(
      numSamples = numSamples,
      numFeatures = numFeatures,
      taskNames = taskNames,
      vocabSize = vocabSize,
      trainRatio = 0.7f,
      valRatio = 0.1f,
      seed = seed
    )
    println(s"  Train: ${trainData.size} samples")
    println(s"  Val: ${valData.size} samples")
    println(s"  Test: ${testData.size} samples")
    println(s"  Tasks: ${taskNames.mkString(", ")}")

    // 2. Create data loaders
    println("\n[2] Creating data loaders...")
    val trainLoader = new DataLoader(trainData, batchSize, shuffle = true)
    val valLoader = new DataLoader(valData, batchSize, shuffle = false)
    val testLoader = new DataLoader(testData, batchSize, shuffle = false)
    println(s"  Train batches: ${trainData.size / batchSize}")
    println(s"  Val batches: ${valData.size / batchSize}")
    println(s"  Test batches: ${testData.size / batchSize}")

    // 3. Define features (sparse features for Census data)
    println("\n[3] Defining features...")
    val featureNames = (0 until numFeatures).map(i => s"feat_$i").toList
    val features = featureNames.map { name =>
      SparseFeature(name = name, vocabSize = vocabSize, embedDim = embedDim)
    }
    println(s"  Sparse features: ${features.size}")

    // Task types (all classification)
    val taskTypes = taskNames.map(_ => "classification")
    println(s"  Task types: ${taskTypes.mkString(", ")}")

    // Shared MLP params
    val sharedMlpParams = Map[String, Any](
      "dims" -> List(128L, 64L),
      "activation" -> "relu",
      "dropout" -> 0.2f
    )
    val towerParams = Map[String, Any](
      "dims" -> List(64L),
      "activation" -> "relu",
      "dropout" -> 0.2f
    )
    val towerParamsList = List(towerParams, towerParams)

    // 4. Create model based on modelName
    println(s"\n[4] Creating model: $modelName...")
    val model: Module = modelName match {
      case "sharedbottom" | "shared_bottom" =>
        new SharedBottom(
          features = features,
          taskTypes = taskTypes,
          bottomParams = sharedMlpParams,
          towerParamsList = towerParamsList,
          device = device
        )

      case "esmm" =>
        ESMM(
          features = features,
          taskNames = taskNames,
          embedDim = embedDim,
          towerDims = List(128L, 64L),
          dropout = 0.2f,
          device = device
        )

      case "mmoe" =>
        new MMOE(
          features = features,
          taskTypes = taskTypes,
          nExpert = 4,
          expertParams = sharedMlpParams,
          towerParamsList = towerParamsList,
          device = device
        )

      case "ple" =>
        new PLE(
          features = features,
          taskTypes = taskTypes,
          nLevel = 3,
          nExpertSpecific = 2,
          nExpertShared = 2,
          expertParams = sharedMlpParams,
          towerParamsList = towerParamsList,
          device = device
        )

      case "aitm" =>
        new AITM(
          features = features,
          nTask = taskNames.size,
          bottomParams = sharedMlpParams,
          towerParamsList = towerParamsList,
          device = device
        )

      case "omoe" =>
        new OMoE(
          features = features,
          taskNames = taskNames,
          embedDim = embedDim,
          numExperts = 4,
          expertDims = List(128L),
          towerDims = List(64L),
          dropout = 0.2f,
          device = device
        )

      case "singletask" | "single_task" =>
        new SingleTaskModel(
          features = features,
          taskNames = taskNames,
          embedDim = embedDim,
          bottomDims = List(128L),
          towerDims = List(64L),
          dropout = 0.2f,
          device = device
        )

      case _ =>
        println(s"  Warning: Unknown model '$modelName', defaulting to MMOE")
        new MMOE(
          features = features,
          taskTypes = taskTypes,
          nExpert = 4,
          expertParams = sharedMlpParams,
          towerParamsList = towerParamsList,
          device = device
        )
    }

    // Count parameters
    println(s"  Model: $modelName")

    // 5. Train model
    println("\n[5] Training multi-task model...")
    val startTime = System.currentTimeMillis()

    val trainer = new MTLTrainer(
      model = model,
      taskNames = taskNames,
      learningRate = learningRate,
      weightDecay = weightDecay,
      device = device,
      numEpochs = numEpochs,
      earlyStopPatience = 5,
      taskWeights = None, // Use equal weights
      verbose = true
    )

    trainer.fit(trainLoader, Some(valLoader))

    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
    println(f"\n  Training completed in $trainingTime%.2f seconds")

    // 6. Evaluate on test set
    println("\n[6] Evaluating on test set...")
    val testMetrics = trainer.evaluate(testLoader)
    println("  Test metrics per task:")
    testMetrics.foreach { case (taskName, auc) =>
      println(f"    $taskName AUC: $auc%.4f")
    }

    // Calculate average AUC
    val avgAUC = testMetrics.values.sum / testMetrics.size
    println(f"  Average AUC: $avgAUC%.4f")

    println("\n" + "=" * 70)
    println(s"Census Multi-Task Example ($modelName) Completed Successfully!")
    println("=" * 70)
  }
}