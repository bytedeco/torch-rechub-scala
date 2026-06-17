package examples.matching

import torchrec.Implicits._
import torchrec.data._
import torchrec.data.DataGenerator
import torchrec.models.matching._
import torchrec.trainers._
import torchrec.basic.features._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * MovieLens Matching Example
 * Demonstrates training DSSM, MIND, ComirecDR, and YoutubeDNN matching models
 * on MovieLens-like data for user-item retrieval.
 */
object MovieLensExample {

  def main(args: Array[String]): Unit = {
    println("=" * 70)
    println("MovieLens Matching Example - Two-Tower Retrieval Models")
    println("=" * 70)

    // Configuration
    val numUsers = 5000
    val numItems = 1000
    val numUserFeatures = 3
    val numItemFeatures = 2
    val avgSeqLen = 10
    val vocabSize = 1000
    val embedDim = 16
    val towerDims = List(256L, 128L, 64L)
    val batchSize = 256
    val learningRate = 1e-4f
    val weightDecay = 1e-6f
    val numEpochs = 3
    val device = "cpu"
    val seed = 2022
    val numInterests = 4

    // Parse model name from args (default: dssm)
    val modelName = if (args.length > 0) args(0).toLowerCase else "dssm"

    println(s"\nConfiguration:")
    println(s"  Model: $modelName")
    println(s"  Users: $numUsers")
    println(s"  Items: $numItems")
    println(s"  Avg Sequence Length: $avgSeqLen")
    println(s"  Embed Dim: $embedDim")
    println(s"  Tower Dims: $towerDims")
    println(s"  Batch Size: $batchSize")
    println(s"  Learning Rate: $learningRate")
    println(s"  Epochs: $numEpochs")

    // 1. Generate synthetic MovieLens-like matching data
    println("\n[1] Generating synthetic MovieLens-like matching data...")
    val (trainData, _, testData) = DataGenerator.generateMatchingData(
      numUsers = numUsers,
      numItems = numItems,
      avgSequenceLength = avgSeqLen,
      numUserFeatures = numUserFeatures,
      numItemFeatures = numItemFeatures,
      vocabSize = vocabSize,
      trainRatio = 0.8f,
      seed = seed
    )
    println(s"  Train users: ${trainData.size}")
    println(s"  Items: $numItems")

    // 2. Create data loaders
    println("\n[2] Creating data loaders...")
    val trainLoader = new DataLoader(trainData, batchSize, shuffle = true)
    val testLoader = new DataLoader(testData, batchSize, shuffle = false)
    println(s"  Train batches: ${trainData.size / batchSize}")
    println(s"  Test batches: ${testData.size / batchSize}")

    // 3. Define features
    println("\n[3] Defining features...")

    // User features
    val userFeatureNames = List("user_feat_0", "user_feat_1", "user_feat_2")
    val userSparseFeatures = userFeatureNames.map { name =>
      SparseFeature(name = name, vocabSize = vocabSize, embedDim = embedDim)
    }

    // Sequence feature for user history
    val sequenceFeature = SequenceFeature(
      name = "history",
      vocabSize = numItems,
      embedDim = embedDim,
      pooling = "mean",
      maxLen = avgSeqLen + 5
    )

    // Item features
    val itemFeatureNames = List("item_feat_0", "item_feat_1")
    val itemSparseFeatures = itemFeatureNames.map { name =>
      SparseFeature(name = name, vocabSize = vocabSize, embedDim = embedDim)
    }

    println(s"  User sparse features: ${userSparseFeatures.size}")
    println(s"  User sequence feature: ${sequenceFeature.name}")
    println(s"  Item sparse features: ${itemSparseFeatures.size}")

    // 4. Create model based on modelName
    println(s"\n[4] Creating model: $modelName...")
    val model: Module = modelName match {
      case "dssm" =>
        new DSSM(
          userFeatures = userSparseFeatures,
          itemFeatures = itemSparseFeatures,
          embedDim = embedDim,
          towerDims = towerDims,
          dropout = 0.2f,
          device = device
        )

      case "mind" =>
        new MIND(
          features = userSparseFeatures,
          sequenceFeature = sequenceFeature,
          embedDim = embedDim,
          numInterests = numInterests,
          capsuleDim = 4,
          mlpDims = towerDims,
          dropout = 0.2f,
          device = device
        )

      case "comirecdr" | "comirec_dr" =>
        new ComirecDR(
          features = userSparseFeatures,
          sequenceFeature = sequenceFeature,
          embedDim = embedDim,
          numInterests = numInterests,
          mlpDims = towerDims,
          dropout = 0.2f,
          device = device
        )

      case "youtubednn" | "youtube_dnn" =>
        new YoutubeDNN(
          features = userSparseFeatures,
          sequenceFeatures = List(sequenceFeature),
          embedDim = embedDim,
          towerDims = towerDims,
          dropout = 0.2f,
          device = device
        )

      case _ =>
        println(s"  Warning: Unknown model '$modelName', defaulting to DSSM")
        new DSSM(
          userFeatures = userSparseFeatures,
          itemFeatures = itemSparseFeatures,
          embedDim = embedDim,
          towerDims = towerDims,
          dropout = 0.2f,
          device = device
        )
    }

    // Count parameters
    println(s"  Model: $modelName")

    // 5. Train model
    println("\n[5] Training model...")
    val startTime = System.currentTimeMillis()

    val trainer = new MatchTrainer(
      model = model,
      learningRate = learningRate,
      weightDecay = weightDecay,
      device = device,
      mode = 2, // listwise training
      temperature = 0.02f,
      numEpochs = numEpochs,
      verbose = true
    )

    trainer.fit(trainLoader, None)

    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
    println(f"\n  Training completed in $trainingTime%.2f seconds")

    // 6. Extract embeddings
    println("\n[6] Extracting user and item embeddings...")
    val userEmbeds = trainer.inferenceEmbedding(testLoader, "user")
    val itemEmbeds = trainer.inferenceEmbedding(testLoader, "item")
    if (userEmbeds.nonEmpty) {
      println(s"  User embeddings shape: ${userEmbeds.head.shape.mkString(", ")}")
    }
    if (itemEmbeds.nonEmpty) {
      println(s"  Item embeddings shape: ${itemEmbeds.head.shape.mkString(", ")}")
    }

    // 7. Evaluate
    println("\n[7] Evaluating retrieval performance...")
    val recallAt10 = trainer.evaluate(testLoader, topk = 10)
    val recallAt20 = trainer.evaluate(testLoader, topk = 20)
    val recallAt50 = trainer.evaluate(testLoader, topk = 50)
    println(f"  Recall@10: $recallAt10%.4f")
    println(f"  Recall@20: $recallAt20%.4f")
    println(f"  Recall@50: $recallAt50%.4f")

    println("\n" + "=" * 70)
    println(s"MovieLens Example ($modelName) Completed Successfully!")
    println("=" * 70)
  }
}
