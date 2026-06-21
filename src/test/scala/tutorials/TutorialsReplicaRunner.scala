package tutorials

object TutorialsReplicaRunner {
  private def safeRun(name: String)(f: => Unit): Unit = {
    try {
      f
      println(s"[OK] $name")
    } catch {
      case e: Throwable =>
        println(s"[WARN] $name failed but runner continues: ${e.getMessage}")
    }
  }

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("Torch-Rechub Tutorials Scala Replica Runner")
    println("=" * 80)

    // Accuracy-verified subset
    safeRun("Tutorial00QuickStartCTRDeepFMReplica") { Tutorial00QuickStartCTRDeepFMReplica.main(Array.empty) }
    safeRun("TutorialAvazuWideDeepReplica") { TutorialAvazuWideDeepReplica.main(Array.empty) }
    safeRun("Tutorial03MultiTaskMMOEReplica") { Tutorial03MultiTaskMMOEReplica.main(Array.empty) }
    safeRun("Tutorial01RankingDINReplica") { Tutorial01RankingDINReplica.main(Array.empty) }
    safeRun("TutorialAmazonBeautyReplica") { TutorialAmazonBeautyReplica.main(Array.empty) }
    safeRun("TutorialAmazonBooksReplica") { TutorialAmazonBooksReplica.main(Array.empty) }
    safeRun("TutorialMultiTaskNotebookReplica") { TutorialMultiTaskNotebookReplica.main(Array.empty) }

    // Functional replicas (serving/matching examples)
    safeRun("Tutorial02MatchingDSSMReplica") { Tutorial02MatchingDSSMReplica.main(Array.empty) }
    safeRun("Tutorial05ModelExportAndServingReplica") { Tutorial05ModelExportAndServingReplica.main(Array.empty) }

    println("=" * 80)
    println("Tutorial replicas finished")
    println("=" * 80)
  }
}

