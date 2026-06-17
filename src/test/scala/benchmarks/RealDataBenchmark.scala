package benchmarks

import org.bytedeco.pytorch.Device
import torchrec.data.*
import torchrec.basic.features.*
import torchrec.models.ranking.*
import torchrec.models.matching.*
import torchrec.trainers.*
import torchrec.utils.DeviceSupport

/**
 * Real Dataset Benchmark — downloads and benchmarks 4 real public datasets:
 * MovieLens 1M   (DSSM two-tower retrieval)
 * Criteo day_15  (DeepFM CTR ranking)
 * Census-Income  (DeepFM binary classification)
 * Amazon Reviews (DSSM two-tower retrieval)
 *
 * Falls back to synthetic data if download fails.
 * All benchmarks must complete without crashing to be considered passing.
 */
object RealDataBenchmark {

  def main(args: Array[String]): Unit = {
    println("=" * 70)
    println("Real Dataset Benchmark (torch-rechub-scala)")
    println("=" * 70)
    DeviceSupport.setDevice(DeviceSupport.DeviceType.CUDA)
    val device = DeviceSupport.backend
    println(s"[DeviceSupport] Active device: $device")
    val results = scala.collection.mutable.ListBuffer[(String, String, Float, Float, Boolean)]()

    // MovieLens 1M - DSSM two-tower retrieval
    results += benchmarkMovieLens(device)

    System.gc()
    // Criteo day_15 - DeepFM CTR
    results += benchmarkCriteo(device)
    System.gc()
    // Census-Income (UCI Adult) - DeepFM binary classification
    results += benchmarkCensus(device)
    System.gc()
    // Amazon Fine Food Reviews - DSSM two-tower retrieval
    results += benchmarkAmazon(device)
    System.gc()
    printResults(results.toList)
  }

  // ---------------------------------------------------------------------------
  // MovieLens 1M
  // ---------------------------------------------------------------------------
  def benchmarkMovieLens(device: String): (String, String, Float, Float, Boolean) = {
    println("\n" + "=" * 60)
    println("MovieLens 1M — DSSM Two-Tower Retrieval")
    println("=" * 60)

    try {
      val (trainDS, valDS, testDS) = MovieLensDataset.load(
        trainRatio = 0.8f,
        negRatio = 4,
        maxSamples = Some(100000),
        seed = 42
      )

      val numUsers = trainDS.userFeatures.get("user_id").fold(6040L)(_.size(0).toLong)
      val numItems = trainDS.itemFeatures.get("movie_id").fold(3706L)(_.size(0).toLong)
      println(s"  [Data] Users: $numUsers, Movies: $numItems, Train: ${trainDS.size}")

      val trainLoader = DataLoader.fromJavaRandom(trainDS, batchSize = 256L)
      val valLoader = DataLoader.fromJavaSequential(valDS, batchSize = 256L)

      val userFeat = SparseFeature("user_id", numUsers.toInt.max(1), 16)
      val movieFeat = SparseFeature("movie_id", numItems.toInt.max(1), 16)

      val model = new DSSM(List(userFeat), List(movieFeat), 16, List(64L, 32L), 0.2f, device)

      val trainer = new MatchTrainer(model, learningRate = 0.001f, device = device, numEpochs = 3, verbose = true)

      val t0 = System.currentTimeMillis()
      trainer.fit(trainLoader, Some(valLoader))
      val elapsed = (System.currentTimeMillis() - t0) / 1000.0f

      println(f"\n  [OK] MovieLens DSSM — elapsed: $elapsed%.2fs")
      ("MovieLens", "DSSM", elapsed, 0.0f, true)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] MovieLens DSSM: ${e.getMessage}")
        e.printStackTrace()
        ("MovieLens", "DSSM", 0.0f, 0.0f, false)
    }
  }

  // ---------------------------------------------------------------------------
  // Criteo day_15 CTR
  // ---------------------------------------------------------------------------
  def benchmarkCriteo(device: String): (String, String, Float, Float, Boolean) = {
    println("\n" + "=" * 60)
    println("Criteo day_15 — DeepFM CTR Ranking")
    println("=" * 60)

    try {
      val (trainDS, valDS, testDS) = CriteoDataset.load(
        trainRatio = 0.8f,
        maxSamples = Some(100000),
        seed = 42
      )

      println(s"  [Data] Train: ${trainDS.size}, Val: ${valDS.size}, Test: ${testDS.size}")

      val trainLoader = DataLoader.fromJavaRandom(trainDS, batchSize = 512L)
      val valLoader = DataLoader.fromJavaSequential(valDS, batchSize = 512L)

      // 26 sparse features (C1-C26), hash-encoded to vocabSize=100000
      val sparseFeats = (0 until 26).map(i => SparseFeature(s"sparse_$i", 100000, 8)).toList

      val model = new DeepFM(sparseFeats, embedDim = 8, mlpDims = List(128L, 64L), dropout = 0.2f, device = device)

      val trainer = new CTRTrainer(model, learningRate = 0.01f, device = device, numEpochs = 3, earlyStopPatience = 3, verbose = true)

      val t0 = System.currentTimeMillis()
      trainer.fit(trainLoader, Some(valLoader))
      val elapsed = (System.currentTimeMillis() - t0) / 1000.0f

      val metrics = trainer.evaluate(valLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)
      val logloss = metrics.getOrElse("LogLoss", 0.0f)

      println(f"\n  [OK] Criteo DeepFM — elapsed: $elapsed%.2fs, AUC: $auc%.4f, LogLoss: $logloss%.4f")
      ("Criteo", "DeepFM", elapsed, auc, true)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] Criteo DeepFM: ${e.getMessage}")
        e.printStackTrace()
        ("Criteo", "DeepFM", 0.0f, 0.0f, false)
    }
  }

  // ---------------------------------------------------------------------------
  // Census-Income (UCI Adult)
  // ---------------------------------------------------------------------------
  def benchmarkCensus(device: String): (String, String, Float, Float, Boolean) = {
    println("\n" + "=" * 60)
    println("Census-Income (UCI Adult) — DeepFM Binary Classification")
    println("=" * 60)

    try {
      val (trainDS, valDS, testDS) = CensusIncomeDataset.load(
        trainRatio = 0.8f,
        useOfficialTest = false,
        seed = 42
      )

      println(s"  [Data] Train: ${trainDS.size}, Val: ${valDS.size}, Test: ${testDS.size}")

      val trainLoader = DataLoader.fromJavaRandom(trainDS, batchSize = 256L)
      val valLoader = DataLoader.fromJavaSequential(valDS, batchSize = 256L)

      // Census features: 14 total
      // Categorical (sparse): workclass, education, marital-status,
      //                       occupation, relationship, race, sex, native-country (8 features)
      // Numerical (dense): age, fnlwgt, education-num, capital-gain, capital-loss,
      //                    hours-per-week (6 features — not used as sparse in DeepFM)
      // NOTE: education_num (index 4) is numerical in CensusIncomeDataset, not categorical
      val features = List(
        SparseFeature("workclass", 100, 4),
        SparseFeature("education", 100, 4),
        SparseFeature("marital_status", 100, 4),
        SparseFeature("occupation", 100, 4),
        SparseFeature("relationship", 100, 4),
        SparseFeature("race", 100, 4),
        SparseFeature("sex", 100, 4),
        SparseFeature("native_country", 100, 4)
      )

      val model = new DeepFM(features, embedDim = 8, mlpDims = List(32L, 16L), dropout = 0.2f, device = device)

      val trainer = new CTRTrainer(model, learningRate = 0.01f, numEpochs = 3, device = device, earlyStopPatience = 3, verbose = true)

      val t0 = System.currentTimeMillis()
      trainer.fit(trainLoader, Some(valLoader))
      val elapsed = (System.currentTimeMillis() - t0) / 1000.0f

      val metrics = trainer.evaluate(valLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)
      val logloss = metrics.getOrElse("LogLoss", 0.0f)

      println(f"\n  [OK] Census DeepFM — elapsed: $elapsed%.2fs, AUC: $auc%.4f, LogLoss: $logloss%.4f")
      ("Census", "DeepFM", elapsed, auc, true)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] Census DeepFM: ${e.getMessage}")
        e.printStackTrace()
        ("Census", "DeepFM", 0.0f, 0.0f, false)
    }
  }

  // ---------------------------------------------------------------------------
  // Amazon Fine Food Reviews
  // ---------------------------------------------------------------------------
  def benchmarkAmazon(device: String): (String, String, Float, Float, Boolean) = {
    println("\n" + "=" * 60)
    println("Amazon Fine Food Reviews — DSSM Two-Tower Retrieval")
    println("=" * 60)

    try {
      val (trainDS, valDS, testDS) = AmazonDataset.load(
        category = "food",
        trainRatio = 0.8f,
        maxReviews = Some(50000),
        seed = 42
      )

      val numUsers = trainDS.userFeatures.get("user_id").fold(10000L)(_.size(0).toLong)
      val numItems = trainDS.itemFeatures.get("item_id").fold(50000L)(_.size(0).toLong)
      println(s"  [Data] Users: $numUsers, Items: $numItems, Train: ${trainDS.size}")

      val trainLoader = DataLoader.fromJavaRandom(trainDS, batchSize = 256L)
      val valLoader = DataLoader.fromJavaSequential(valDS, batchSize = 256L)

      val userFeat = SparseFeature("user_id", numUsers.toInt.max(1), 16)
      val itemFeat = SparseFeature("item_id", numItems.toInt.max(1), 16)

      val model = new DSSM(List(userFeat), List(itemFeat), 16, List(64L, 32L), 0.2f, device)

      val trainer = new MatchTrainer(model, learningRate = 0.001f, device = device, numEpochs = 3, verbose = true)

      val t0 = System.currentTimeMillis()
      trainer.fit(trainLoader, Some(valLoader))
      val elapsed = (System.currentTimeMillis() - t0) / 1000.0f

      println(f"\n  [OK] Amazon DSSM — elapsed: $elapsed%.2fs")
      ("Amazon", "DSSM", elapsed, 0.0f, true)
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] Amazon DSSM: ${e.getMessage}")
        e.printStackTrace()
        ("Amazon", "DSSM", 0.0f, 0.0f, false)
    }
  }

  // ---------------------------------------------------------------------------
  // Results
  // ---------------------------------------------------------------------------
  def printResults(results: List[(String, String, Float, Float, Boolean)]): Unit = {
    val passed = results.count(_._5)
    val total = results.size

    println("\n" + "=" * 70)
    println("Real Dataset Benchmark Results")
    println("=" * 70)
    println(f"${"Dataset"}%-15s${"Model"}%-15s${"Time(s)"}%-12s${"AUC"}%-10s${"Status"}%-10s")
    println("-" * 70)

    results.foreach { case (ds, model, time, auc, ok) =>
      val aucStr = if (auc > 0) f"$auc%.4f" else "-"
      val status = if (ok) "PASS" else "FAIL"
      println(f"$ds%-15s$model%-15s${time}%12.2f$aucStr%-10s$status%-10s")
    }

    println("-" * 70)
    println(f"Summary: $passed/$total benchmarks passed")
    println("=" * 70)

    if (passed < total) {
      println("WARNING: Some benchmarks failed. Check output above.")
    } else {
      println("All benchmarks passed!")
    }
  }
}
