package tutorials

import torchrec.Implicits._
import torchrec.data.DataGenerator
import torchrec.basic.features._
import torchrec.data._
import torchrec.models.matching._
import torchrec.trainers._

/**
 * Tutorial: Two-Tower Matching with DSSM
 *
 * DSSM (Deep Structured Semantic Model) uses a dual-tower architecture
 * for efficient user-item matching in large-scale retrieval.
 */
object MatchingDSSM {

  def main(args: Array[String]): Unit = {
    println(
      """
        |=============================================================
        | Tutorial: Two-Tower Matching with DSSM
        |=============================================================
        |
        | DSSM learns separate representations for users and items,
        | enabling efficient nearest-neighbor search for retrieval.
        |
        |""".stripMargin)

    // Generate matching data first so we can build feature definitions that match
    println("Defining features and generating matching data...")
    val numUsers = 500
    val numItems = 1000
    val genNumUserFeatures = 3
    val genNumItemFeatures = 2
    val genVocabSize = 100

    println("\nGenerating matching data...")
    val (trainData, _, testData) = DataGenerator.generateMatchingData(
      numUsers = numUsers,
      numItems = numItems,
      avgSequenceLength = 10,
      numUserFeatures = genNumUserFeatures,
      numItemFeatures = genNumItemFeatures,
      vocabSize = genVocabSize
    )
    println(f"  Train: ${trainData.size}%,d users")
    println(f"  Items: ${numItems}%,d")

    // Build feature lists that match the generated dataset keys
    val embedDim = 16
    val userFeatures = (0 until genNumUserFeatures).map { i =>
      SparseFeature(s"user_feat_$i", genVocabSize, embedDim)
    }.toList ++ List(SequenceFeature("history", numItems, embedDim))

    val itemFeatures = (0 until genNumItemFeatures).map { i =>
      SparseFeature(s"item_feat_$i", genVocabSize, embedDim)
    }.toList

    println("\nUser features:")
    userFeatures.foreach(f => println(s"  - ${f.name}: vocab=${f.vocabSize}"))
    println("\nItem features:")
    itemFeatures.foreach(f => println(s"  - ${f.name}: vocab=${f.vocabSize}"))

    // Create DSSM model
    println("\nCreating DSSM model...")
    val model = new DSSM(
      userFeatures = userFeatures,
      itemFeatures = itemFeatures,
      embedDim = embedDim,
      towerDims = List(256L, 128L, 64L),
      dropout = 0.2f
    )
    println(s"  Model: DSSM(embed_dim=$embedDim, tower_dims=[256, 128, 64])")

    // Create data loaders
    val trainLoader = new DataLoader(trainData, batchSize = 128, shuffle = true)
    val testLoader = new DataLoader(testData, batchSize = 128, shuffle = false)

    // Create trainer
    println("\nCreating MatchTrainer...")
    val trainer = new MatchTrainer(
      model = model,
      learningRate = 1e-3f,
      mode = 2, // listwise
      numEpochs = 3,
      verbose = true
    )

    // Train
    println("\nTraining...")
    trainer.fit(trainLoader)

    // Extract embeddings
    println("\nExtracting user embeddings for inference...")
    val userEmbeds = trainer.inferenceEmbedding(testLoader, "user")
    println(f"  User embeddings shape: ${userEmbeds.head.shape.mkString(", ")}")

    println(
      """
        |
        |=============================================================
        | Tutorial Complete!
        |=============================================================
        |
        | Next: Build ANN index for fast retrieval
        |""".stripMargin)
  }
}