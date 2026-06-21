package tutorials

import torchrec.data.DataLoader
import torchrec.models.ranking.DeepFM
import torchrec.trainers.CTRTrainer

object TutorialAmazonBooksReplica {
  def main(args: Array[String]): Unit = {
    val csv = if (args.nonEmpty) args(0) else "/home/muller/IdeaProjects/torch-rechub/examples/ranking/data/amazon-books/amazon_books_sample.csv"
    val split = AmazonBooksBeautyCtrSupport.load(csv)
    val half = split.features.size / 2
    val model = new DeepFM(split.features.take(half), split.features.drop(half), embedDim = 8, mlpDims = List(64L, 32L), dropout = 0.2f)
    val trainer = new CTRTrainer(model, learningRate = 1e-3f, numEpochs = 2, earlyStopPatience = 2, verbose = true)

    val train = new DataLoader(split.train, 256, shuffle = true)
    val valid = new DataLoader(split.valid, 256, shuffle = false)
    val test = new DataLoader(split.test, 256, shuffle = false)

    trainer.fit(train, Some(valid))
    val m = trainer.evaluate(test)
    val auc = m.getOrElse("AUC", 0.0f)
    println(f"[AMAZON_BOOKS_REPLICA] auc=$auc%.4f metrics=$m")
    require(auc >= 0.0f && auc <= 1.0f, f"Invalid AUC value: $auc%.4f")
    require(m.contains("LogLoss") && m.contains("Accuracy"), "Missing core metrics")
  }
}

