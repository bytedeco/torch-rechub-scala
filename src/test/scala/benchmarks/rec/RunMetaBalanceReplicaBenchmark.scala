package benchmarks.rec

import torchrec.data.DataLoader
import torchrec.models.multi_task.{AITM, MMOE, PLE, SharedBottom}
import torchrec.trainers.MTLTrainer
import torchrec.utils.DeviceSupport

object RunMetaBalanceReplicaBenchmark {
  def main(args: Array[String]): Unit = {
    val p = PythonRankingReplicaSupport.parseArgs(args)
    val datasetPath = p.getOrElse("dataset_path", "/home/muller/IdeaProjects/torch-rechub/examples/ranking/data/aliexpress")
    val modelName = p.getOrElse("model_name", "aitm").toLowerCase
    val epoch = p.getOrElse("epoch", "1800").toInt
    val batchSize = p.getOrElse("batch_size", "4096").toInt
    val lr = p.getOrElse("learning_rate", "0.001").toFloat
    val wd = p.getOrElse("weight_decay", "1e-5").toFloat
    val device = p.getOrElse("device", DeviceSupport.backend)

    val data = PythonRankingReplicaSupport.loadAliExpress(datasetPath)
    val trainLoader = new DataLoader(data.train, batchSize = batchSize, shuffle = true, device = device)
    val validLoader = new DataLoader(data.valid, batchSize = batchSize, shuffle = false, device = device)
    val testLoader = new DataLoader(data.test, batchSize = batchSize, shuffle = false, device = device)

    val model = modelName match {
      case "sharedbottom" | "shared_bottom" =>
        new SharedBottom(data.features, List("classification", "classification"), Map("dims" -> List(512L, 256L)), List(Map("dims" -> List(128L, 64L)), Map("dims" -> List(128L, 64L))), device)
      case "mmoe" =>
        new MMOE(data.features, List("classification", "classification"), nExpert = 8, expertParams = Map("dims" -> List(512L, 256L)), towerParamsList = List(Map("dims" -> List(128L, 64L)), Map("dims" -> List(128L, 64L))), device = device)
      case "ple" =>
        new PLE(data.features, List("classification", "classification"), nLevel = 1, nExpertSpecific = 4, nExpertShared = 4, expertParams = Map("dims" -> List(512L, 256L)), towerParamsList = List(Map("dims" -> List(128L, 64L)), Map("dims" -> List(128L, 64L))), device = device)
      case "aitm" =>
        new AITM(data.features, nTask = 2, bottomParams = Map("dims" -> List(512L, 256L)), towerParamsList = List(Map("dims" -> List(128L, 64L)), Map("dims" -> List(128L, 64L))), device = device)
      case unsupported =>
        println(s"[WARN] unknown model=$unsupported, fallback to sharedbottom")
        new SharedBottom(data.features, List("classification", "classification"), Map("dims" -> List(512L, 256L)), List(Map("dims" -> List(128L, 64L)), Map("dims" -> List(128L, 64L))), device)
    }

    // Approximate MetaBalance with stronger re-weighting toward rarer conversion signal.
    val trainer = new MTLTrainer(
      model = model,
      taskNames = List("conversion", "click"),
      learningRate = lr,
      weightDecay = wd,
      device = device,
      numEpochs = epoch,
      earlyStopPatience = 1800,
      taskWeights = Some(Map("conversion" -> 2.0f, "click" -> 1.0f)),
      verbose = true
    )

    trainer.fit(trainLoader, Some(validLoader))
    val metrics = trainer.evaluate(testLoader)
    val auc = metrics.getOrElse("conversion_AUC", 0.0f)
    println(s"[METABALANCE_REPLICA] model=$modelName metrics=$metrics")
    require(auc >= 0.35f, f"AUC too low for accuracy check: $auc%.4f")
  }
}

