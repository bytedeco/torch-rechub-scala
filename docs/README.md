# Tutorials Scala Replica

This folder contains Scala replicas for `torch-rechub/tutorials` notebooks.

## Added replica files

- `Tutorial00QuickStartCTRDeepFMReplica.scala`
- `Tutorial01RankingDINReplica.scala`
- `Tutorial02MatchingDSSMReplica.scala`
- `Tutorial03MultiTaskMMOEReplica.scala`
- `Tutorial04ExperimentTrackingLightReplica.scala`
- `Tutorial05ModelExportAndServingReplica.scala`
- `TutorialDINNotebookReplica.scala`
- `TutorialDeepFMNotebookReplica.scala`
- `TutorialMatchingNotebookReplica.scala`
- `TutorialMilvusNotebookReplica.scala`
- `TutorialMultiTaskNotebookReplica.scala`
- `TutorialAvazuWideDeepReplica.scala`
- `TutorialAmazonBeautyReplica.scala`
- `TutorialAmazonBooksReplica.scala`
- `AmazonBooksBeautyCtrSupport.scala`
- `TutorialsReplicaRunner.scala`

## Datasets used

- `/home/muller/IdeaProjects/torch-rechub/examples/ranking/data/amazon-beauty`
- `/home/muller/IdeaProjects/torch-rechub/examples/ranking/data/amazon-books`
- `/home/muller/IdeaProjects/torch-rechub/examples/ranking/data/amazon-electronics`
- `/home/muller/IdeaProjects/torch-rechub/examples/ranking/data/avazu`
- `/home/muller/IdeaProjects/torch-rechub/examples/ranking/data/census-income`
- `/home/muller/IdeaProjects/torch-rechub/examples/ranking/data/criteo`

## Quick run

```bash
cd /home/muller/IdeaProjects/torch-rechub-scala
sbt --no-colors "Test/runMain tutorials.TutorialsReplicaRunner"
```

## Individual runs

```bash
cd /home/muller/IdeaProjects/torch-rechub-scala
sbt --no-colors "Test/runMain tutorials.TutorialAmazonBeautyReplica"
sbt --no-colors "Test/runMain tutorials.TutorialAmazonBooksReplica"
sbt --no-colors "Test/runMain tutorials.Tutorial00QuickStartCTRDeepFMReplica"
sbt --no-colors "Test/runMain tutorials.Tutorial03MultiTaskMMOEReplica"
```

