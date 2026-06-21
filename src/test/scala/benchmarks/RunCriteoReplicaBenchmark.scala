package benchmarks

import torchrec.data.DataLoader
import torchrec.models.ranking.{AutoInt, DCN, DCNv2, DeepFM, FiBiNet, WideDeep}
import torchrec.trainers.CTRTrainer
import torchrec.utils.DeviceSupport

object RunCriteoReplicaBenchmark {
  def main(args: Array[String]): Unit = {
    val p = PythonRankingReplicaSupport.parseArgs(args)
    val datasetPath = p.getOrElse("dataset_path", "/home/muller/IdeaProjects/torch-rechub/examples/ranking/data/criteo/criteo_sample.csv")
    val modelName = p.getOrElse("model_name", "dcn_v2").toLowerCase
//    val modelName = p.getOrElse("model_name", "fibinet").toLowerCase
    val epoch = p.getOrElse("epoch", "200").toInt
    val batchSize = p.getOrElse("batch_size", "2048").toInt
    val lr = p.getOrElse("learning_rate", "0.001").toFloat
    val wd = p.getOrElse("weight_decay", "1e-3").toFloat
    val device = p.getOrElse("device", DeviceSupport.backend)

    val data = PythonRankingReplicaSupport.loadCriteo(datasetPath)
    val trainLoader = new DataLoader(data.train, batchSize = batchSize, shuffle = true, device = device)
    val validLoader = new DataLoader(data.valid, batchSize = batchSize, shuffle = false, device = device)
    val testLoader = new DataLoader(data.test, batchSize = batchSize, shuffle = false, device = device)

    val model = modelName match {
      case "widedeep" => new WideDeep(data.features, embedDim = 16, mlpDims = List(256L, 128L), dropout = 0.2f, device = device)
      case "deepfm" =>
        val half = data.features.size / 2
        new DeepFM(data.features.take(half), data.features.drop(half), embedDim = 16, mlpDims = List(256L, 128L), dropout = 0.2f, device = device)
      case "dcn" => new DCN(data.features, embedDim = 16, numCrossLayers = 3, mlpDims = List(256L, 128L), dropout = 0.2f, device = device)
      case "dcn_v2" | "dcnv2" => new DCNv2(data.features, embedDim = 16, numCrossLayers = 3, mlpDims = List(256L, 128L), dropout = 0.2f, device = device)
      case "fibinet" => new FiBiNet(data.features, embedDim = 16, mlpDims = List(256L, 128L), reduction = 3, dropout = 0.2f, device = device)
      case "autoint" => new AutoInt(data.features, embedDim = 16, numAttnHeads = 2, numLayers = 3, mlpDims = List(256L, 128L), dropout = 0.2f, device = device)
      case unsupported =>
        println(s"[WARN] model=$unsupported is not fully supported in Scala replica; fallback to deepfm")
        val half = data.features.size / 2
        new DeepFM(data.features.take(half), data.features.drop(half), embedDim = 16, mlpDims = List(256L, 128L), dropout = 0.2f, device = device)
    }

    val trainer = new CTRTrainer(model, learningRate = lr, weightDecay = wd, device = device, numEpochs = epoch, earlyStopPatience = 400, verbose = true)
    trainer.fit(trainLoader, Some(validLoader))
    val metrics = trainer.evaluate(testLoader)

    val auc = metrics.getOrElse("AUC", 0.0f)
    println(f"[CRITEO] model=$modelName test_auc=$auc%.4f metrics=$metrics")
    require(auc >= 0.35f, f"AUC too low for accuracy check: $auc%.4f")
  }
}

