package benchmarks

import torchrec.data.DataLoader
import torchrec.data.TensorDataset
import torchrec.basic.features.SparseFeature
import torchrec.models.ranking.DeepFM
import torchrec.trainers.CTRTrainer
import torchrec.utils.DeviceSupport

object RunDienReplicaBenchmark {
  def main(args: Array[String]): Unit = {
    val p = PythonRankingReplicaSupport.parseArgs(args)
    val datasetPath = p.getOrElse("dataset_path", "/home/muller/IdeaProjects/torch-rechub/examples/ranking/data/amazon-electronics/amazon_electronics_sample.csv")
    val epoch = p.getOrElse("epoch", "2").toInt
    val batchSize = p.getOrElse("batch_size", "1024").toInt
    val lr = p.getOrElse("learning_rate", "0.001").toFloat
    val device = p.getOrElse("device", DeviceSupport.backend)

    val data = PythonRankingReplicaSupport.loadAmazonElectronics(datasetPath)
    def toCtr(ds: torchrec.data.SequenceDataset): TensorDataset = {
      val histItemLast = ds.sequenceFeatures("hist_item_id").select(1, ds.sequenceFeatures("hist_item_id").size(1) - 1)
      val histCateLast = ds.sequenceFeatures("hist_cate_id").select(1, ds.sequenceFeatures("hist_cate_id").size(1) - 1)
      new TensorDataset(
        sparseFeatures = Map(
          "user_id" -> ds.features("user_id"),
          "target_item_id" -> ds.features("target_item_id"),
          "target_cate_id" -> ds.features("target_cate_id"),
          "hist_item_last" -> histItemLast,
          "hist_cate_last" -> histCateLast
        ),
        labels = ds.labels
      )
    }

    val trainCtr = toCtr(data.train)
    val validCtr = toCtr(data.valid)
    val testCtr = toCtr(data.test)

    val features = List(
      SparseFeature("user_id", data.features.find(_.name == "user_id").map(_.vocabSize).getOrElse(1000L), 8),
      SparseFeature("target_item_id", data.features.find(_.name == "target_item_id").map(_.vocabSize).getOrElse(1000L), 8),
      SparseFeature("target_cate_id", data.features.find(_.name == "target_cate_id").map(_.vocabSize).getOrElse(1000L), 8),
      SparseFeature("hist_item_last", data.historyFeatures.find(_.name == "hist_item_id").map(_.vocabSize).getOrElse(1000L), 8),
      SparseFeature("hist_cate_last", data.historyFeatures.find(_.name == "hist_cate_id").map(_.vocabSize).getOrElse(1000L), 8)
    )

    val half = features.size / 2
    val model = new DeepFM(features.take(half), features.drop(half), embedDim = 8, mlpDims = List(256L, 128L), dropout = 0.2f, device = device)

    val trainLoader = new DataLoader(trainCtr, batchSize = batchSize, shuffle = true, device = device)
    val validLoader = new DataLoader(validCtr, batchSize = batchSize, shuffle = false, device = device)
    val testLoader = new DataLoader(testCtr, batchSize = batchSize, shuffle = false, device = device)

    val trainer = new CTRTrainer(model, learningRate = lr, device = device, numEpochs = epoch, earlyStopPatience = 2, verbose = true)
    trainer.fit(trainLoader, Some(validLoader))
    val metrics = trainer.evaluate(testLoader)
    val auc = metrics.getOrElse("AUC", 0.0f)
    println(f"[DIEN_FALLBACK_DEEPFM] auc=$auc%.4f metrics=$metrics")
    require(auc >= 0.35f, f"AUC too low for accuracy check: $auc%.4f")
  }
}

