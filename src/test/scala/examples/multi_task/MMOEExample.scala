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
 * MMOE Example for Multi-Task Learning
 */
object MMOEExample {

  def main(args: Array[String]): Unit = {
    println("=" * 60)
    println("MMOe Multi-Task Learning Example")
    println("=" * 60)

    // Configuration
    val numSamples = 3000
    val numFeatures = 10
    val vocabSize = 100
    val embedDim = 8
    val numExperts = 4
    val expertDims = List(128L)
    val towerDims = List(64L)
    val batchSize = 256
    val learningRate = 1e-3f
    val numEpochs = 5
    val device = "cpu"

    // Task names and types
    val taskNames = List("cvr", "ctr")
    val taskTypes = List("classification", "classification")

    println(s"\nConfiguration:")
    println(s"  Samples: $numSamples")
    println(s"  Features: $numFeatures")
    println(s"  Tasks: ${taskNames.mkString(", ")}")
    println(s"  Experts: $numExperts")
    println(s"  Expert Dims: $expertDims")

    // 1. Generate multi-task data
    println("\n[1] Generating synthetic multi-task data...")
    val (trainData, valData, testData) = DataGenerator.generateMultiTaskData(
      numSamples = numSamples,
      numFeatures = numFeatures,
      taskNames = taskNames,
      vocabSize = vocabSize,
      seed = 42
    )
    println(s"  Train: ${trainData.size} samples")
    println(s"  Val: ${valData.size} samples")
    println(s"  Test: ${testData.size} samples")

    // 2. Create data loaders
    println("\n[2] Creating data loaders...")
    val trainLoader = new DataLoader(trainData, batchSize, shuffle = true)
    val valLoader = new DataLoader(valData, batchSize, shuffle = false)

    // 3. Define features
    println("\n[3] Defining features...")
    val features = (0 until numFeatures).map { i =>
      SparseFeature(s"feat_$i", vocabSize, embedDim)
    }.toList
    println(s"  Created ${features.size} sparse features")

    // 4. Create model
    println("\n[4] Creating MMOE model...")
    val expertParams = Map[String, Any](
      "dims" -> expertDims,
      "activation" -> "relu",
      "dropout" -> 0.2f
    )
    val towerParams = Map[String, Any](
      "dims" -> towerDims,
      "activation" -> "relu",
      "dropout" -> 0.2f
    )
    val model = new MMOE(
      features = features,
      taskTypes = taskTypes,
      nExpert = numExperts,
      expertParams = expertParams,
      towerParamsList = List(towerParams, towerParams),
      device = device
    )
    println(s"  Model created: MMOE(num_experts=$numExperts, expert_dims=$expertDims)")

    // 5. Train model
    println("\n[5] Training model...")
    val startTime = System.currentTimeMillis()

    val taskWeights = Map("cvr" -> 1.0f, "ctr" -> 1.0f)

    val trainer = new MTLTrainer(
      model = model,
      taskNames = taskNames,
      learningRate = learningRate,
      device = device,
      numEpochs = numEpochs,
      earlyStopPatience = 3,
      taskWeights = Some(taskWeights),
      verbose = true
    )

    trainer.fit(trainLoader, Some(valLoader))

    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
    println(f"\n  Training completed in $trainingTime%.2f seconds")

    // 6. Evaluate
    println("\n[6] Evaluating...")
    val metrics = trainer.evaluate(valLoader)
    println("  Task metrics:")
    metrics.foreach { case (task, auc) =>
      println(f"    $task AUC: $auc%.4f")
    }

    println("\n" + "=" * 60)
    println("MMOE Example Completed Successfully!")
    println("=" * 60)
  }
}