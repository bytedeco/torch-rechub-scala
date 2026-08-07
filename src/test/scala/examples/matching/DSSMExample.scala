package examples.matching

import torchrec.Implicits.*
import torchrec.data.*
import torchrec.models.matching.*
import torchrec.trainers.*
import torchrec.basic.features.SparseFeature
import torchrec.utils.DeviceSupport
import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.nn.modules._
import org.bytedeco.pytorch.nn.modules.container._
import org.bytedeco.pytorch.nn.options._
import org.bytedeco.pytorch.optim._
import org.bytedeco.pytorch.data.datasets._
import org.bytedeco.pytorch.data.options._
import org.bytedeco.pytorch.data.sampler._
import org.bytedeco.pytorch.distributed._
import org.bytedeco.pytorch.global.torch

/**
 * DSSM Example for User-Item Matching
 */
object DSSMExample {

  def main(args: Array[String]): Unit = {
    DeviceSupport.setDevice(DeviceSupport.DeviceType.AUTO)
    val device = DeviceSupport.backend
    println(s"[DeviceSupport] Active device: $device")

    println("=" * 60)
    println("DSSM Two-Tower Matching Example")
    println("=" * 60)

    // Configuration
    val numUsers = 5000
    val numItems = 1000
    val numUserFeatures = 3
    val numItemFeatures = 2
    val vocabSize = 100
    val embedDim = 8
    val towerDims = List(128L, 64L)
    val batchSize = 128
    val learningRate = 1e-3f
    val numEpochs = 5

    println(s"\nConfiguration:")
    println(s"  Device: $device")
    println(s"  Users: $numUsers")
    println(s"  Items: $numItems")
    println(s"  Embed Dim: $embedDim")
    println(s"  Tower Dims: $towerDims")

    // Generate matching data
    println("\n[1] Generating synthetic matching data...")
    val (trainData, _, testData) = benchmarks.DataGenerator.generateMatchingData(
      numUsers = numUsers,
      numItems = numItems,
      avgSequenceLength = 10,
      numUserFeatures = numUserFeatures,
      numItemFeatures = numItemFeatures,
      vocabSize = vocabSize,
      seed = 42
    )
    println(s"  Train users: ${trainData.size}")
    println(s"  Items: $numItems")

    // Create data loaders
    println("\n[2] Creating data loaders...")
    val trainLoader = new DataLoader(trainData, batchSize, shuffle = true, device = device)
    val testLoader = new DataLoader(testData, batchSize, shuffle = false, device = device)

    // Define features
    println("\n[3] Defining features...")
    val userFeatures = (0 until numUserFeatures).map { i =>
      SparseFeature(s"user_$i", vocabSize, embedDim)
    }.toList
    val itemFeatures = (0 until numItemFeatures).map { i =>
      SparseFeature(s"item_$i", numItems, embedDim)
    }.toList

    println("  User features:")
    userFeatures.foreach { f => println(s"    - ${f.name}: vocab=${f.vocabSize}") }
    println("  Item features:")
    itemFeatures.foreach { f => println(s"    - ${f.name}: vocab=${f.vocabSize}") }

    // Create model
    println("\n[4] Creating DSSM model...")
    val model = new DSSM(
      userFeatures = userFeatures,
      itemFeatures = itemFeatures,
      embedDim = embedDim,
      towerDims = towerDims,
      dropout = 0.2f,
      device = device
    )
    println(s"  Model created: DSSM(user_tower=$towerDims, item_tower=$towerDims)")

    // Train
    println("\n[5] Training model...")
    val startTime = System.currentTimeMillis()
    val trainer = new MatchTrainer(
      model = model,
      learningRate = learningRate,
      device = device,
      numEpochs = numEpochs,
      verbose = true
    )
    trainer.fit(trainLoader)
    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
    println(f"\n  Training completed in $trainingTime%.2f seconds")

    // Extract embeddings
    println("\n[6] Extracting user and item embeddings...")
    val userEmbeds = trainer.inferenceEmbedding(testLoader, "user")
    val itemEmbeds = trainer.inferenceEmbedding(testLoader, "item")
    if (userEmbeds.nonEmpty) {
      println(s"  User embeddings shape: ${userEmbeds.head.shape.mkString(", ")}")
    }
    if (itemEmbeds.nonEmpty) {
      println(s"  Item embeddings shape: ${itemEmbeds.head.shape.mkString(", ")}")
    }

    // Evaluate
    println("\n[7] Evaluating...")
    val recall = trainer.evaluate(testLoader, topk = 10)
    println(f"  Recall@10: $recall%.4f")

    val fullMetrics = trainer.evaluateFull(testLoader, topk = 10)
    println("  Full metrics:")
    fullMetrics.foreach { case (name, value) => println(f"    $name: $value%.4f") }

    println("\n" + "=" * 60)
    println("DSSM Example Completed Successfully!")
    println("=" * 60)
  }
}
