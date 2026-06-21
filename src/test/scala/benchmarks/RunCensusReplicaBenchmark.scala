package benchmarks

import torchrec.data.DataLoader
import torchrec.models.multi_task.{AITM, ESMM, MMOE, PLE, SharedBottom}
import torchrec.trainers.MTLTrainer
import torchrec.utils.DeviceSupport

object RunCensusReplicaBenchmark {
  def main(args: Array[String]): Unit = {
    val p = PythonRankingReplicaSupport.parseArgs(args)
    val datasetPath = p.getOrElse("dataset_path", "/home/muller/IdeaProjects/torch-rechub/examples/ranking/data/census-income")
    val modelName = p.getOrElse("model_name", "sharedbottom").toLowerCase
    val epoch = p.getOrElse("epoch", "200").toInt
    val batchSize = p.getOrElse("batch_size", "1024").toInt
    val lr = p.getOrElse("learning_rate", "0.001").toFloat
    val wd = p.getOrElse("weight_decay", "1e-4").toFloat
    val device = p.getOrElse("device", DeviceSupport.backend)

    val data = PythonRankingReplicaSupport.loadCensus(datasetPath)
    val trainLoader = new DataLoader(data.train, batchSize = batchSize, shuffle = true, device = device)
    val validLoader = new DataLoader(data.valid, batchSize = batchSize, shuffle = false, device = device)
    val testLoader = new DataLoader(data.test, batchSize = batchSize, shuffle = false, device = device)

    val (model, taskNames) = modelName match {
      case "sharedbottom" | "shared_bottom" =>
        (
          new SharedBottom(
            data.allFeatures,
            taskTypes = List("classification", "classification"),
            bottomParams = Map("dims" -> List(117L)),
            towerParamsList = List(Map("dims" -> List(8L)), Map("dims" -> List(8L))),
            device = device
          ),
          List("cvr_label", "ctr_label")
        )
      case "esmm" =>
        (
          new ESMM(data.userFeatures, data.itemFeatures, cvrParams = Map("dims" -> List(16L, 8L)), ctrParams = Map("dims" -> List(16L, 8L)), device = device),
          List("cvr_label", "ctr_label", "ctcvr_label")
        )
      case "mmoe" =>
        (
          new MMOE(
            data.allFeatures,
            taskTypes = List("classification", "classification"),
            nExpert = 8,
            expertParams = Map("dims" -> List(16L)),
            towerParamsList = List(Map("dims" -> List(8L)), Map("dims" -> List(8L))),
            device = device
          ),
          List("cvr_label", "ctr_label")
        )
      case "ple" =>
        (
          new PLE(
            data.allFeatures,
            taskTypes = List("classification", "classification"),
            nLevel = 1,
            nExpertSpecific = 2,
            nExpertShared = 1,
            expertParams = Map("dims" -> List(16L)),
            towerParamsList = List(Map("dims" -> List(8L)), Map("dims" -> List(8L))),
            device = device
          ),
          List("cvr_label", "ctr_label")
        )
      case "aitm" =>
        (
          new AITM(
            data.allFeatures,
            nTask = 2,
            bottomParams = Map("dims" -> List(32L, 16L)),
            towerParamsList = List(Map("dims" -> List(8L)), Map("dims" -> List(8L))),
            device = device
          ),
          List("cvr_label", "ctr_label")
        )
      case "omoe" =>
        (
          new torchrec.models.multi_task.OMoE(
            features = data.allFeatures,
            taskNames = List("cvr_label", "ctr_label"),
            embedDim = 8,
            numExperts = 4,
            expertDims = List(64L),
            towerDims = List(32L),
            dropout = 0.2f,
            device = device
          ),
          List("cvr_label", "ctr_label")
        )
      case unsupported =>
        println(s"[WARN] unknown model=$unsupported, fallback to SharedBottom")
        (
          new SharedBottom(data.allFeatures, List("classification", "classification"), Map("dims" -> List(117L)), List(Map("dims" -> List(8L)), Map("dims" -> List(8L))), device),
          List("cvr_label", "ctr_label")
        )
    }

    val trainer = new MTLTrainer(model, taskNames = taskNames, learningRate = lr, weightDecay = wd, device = device, numEpochs = epoch, earlyStopPatience = 800, verbose = true)
    trainer.fit(trainLoader, Some(validLoader))
    val metrics = trainer.evaluate(testLoader)

    val key = s"${taskNames.head}_AUC"
    val auc = metrics.getOrElse(key, 0.0f)
    println(s"[CENSUS] model=$modelName metrics=$metrics")
    require(auc >= 0.35f, f"AUC too low for accuracy check: $auc%.4f")
  }
}

