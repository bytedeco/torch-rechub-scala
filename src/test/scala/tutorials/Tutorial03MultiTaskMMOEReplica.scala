package tutorials

import benchmarks.PythonRankingReplicaSupport
import torchrec.data.DataLoader
import torchrec.models.multi_task.SharedBottom
import torchrec.trainers.MTLTrainer

object Tutorial03MultiTaskMMOEReplica {
  def main(args: Array[String]): Unit = {
    val ds = PythonRankingReplicaSupport.loadCensus("/home/muller/IdeaProjects/torch-rechub/examples/ranking/data/census-income")
    val model = new SharedBottom(
      ds.allFeatures,
      taskTypes = List("classification", "classification"),
      bottomParams = Map("dims" -> List(117L)),
      towerParamsList = List(Map("dims" -> List(8L)), Map("dims" -> List(8L))),
      device = "cpu"
    )
    val trainer = new MTLTrainer(model, List("cvr_label", "ctr_label"), learningRate = 1e-3f, device = "cpu", numEpochs = 1, earlyStopPatience = 2)
    trainer.fit(new DataLoader(ds.train, 512, shuffle = true), Some(new DataLoader(ds.valid, 512, shuffle = false)))
    val metrics = trainer.evaluate(new DataLoader(ds.test, 512, shuffle = false))
    val auc = metrics.getOrElse("cvr_label_AUC", 0.0f)
    println(s"[TUTORIAL03_MTL] metrics=$metrics")
    require(auc >= 0.0f && auc <= 1.0f, f"Invalid AUC: $auc%.4f")
  }
}

