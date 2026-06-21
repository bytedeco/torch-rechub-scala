package tutorials

import benchmarks.DataGenerator
import torchrec.basic.features.SparseFeature
import torchrec.data.DataLoader
import torchrec.models.matching.DSSM
import torchrec.trainers.MatchTrainer

object Tutorial02MatchingDSSMReplica {
  def main(args: Array[String]): Unit = {
    val (trainData, _, testData) = DataGenerator.generateMatchingData(
      numUsers = 2000,
      numItems = 1000,
      avgSequenceLength = 10,
      numUserFeatures = 3,
      numItemFeatures = 2,
      vocabSize = 100,
      seed = 2024
    )

    val userFeatures = List(
      SparseFeature("user_0", 100, 16),
      SparseFeature("user_1", 100, 16),
      SparseFeature("user_2", 100, 16)
    )
    val itemFeatures = List(
      SparseFeature("item_0", 100, 16),
      SparseFeature("item_1", 100, 16)
    )

    val model = new DSSM(userFeatures, itemFeatures, embedDim = 16, towerDims = List(128L, 64L), dropout = 0.2f, device = "cpu")
    val trainer = new MatchTrainer(model, learningRate = 1e-3f, device = "cpu", numEpochs = 2, verbose = true)

    val trainLoader = new DataLoader(trainData, batchSize = 128, shuffle = true, device = "cpu")
    val testLoader = new DataLoader(testData, batchSize = 128, shuffle = false, device = "cpu")
    trainer.fit(trainLoader)

    val recall = trainer.evaluate(testLoader, topk = 10)
    println(f"[TUTORIAL02_DSSM] recall@10=$recall%.4f")
    require(recall >= 0.0f && recall <= 1.0f, f"Invalid Recall@10: $recall%.4f")
  }
}

