package benchmarks.rec

import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.{Device, Module, Tensor}
import torchrec.Implicits.{RichTensor, tensor}
import torchrec.basic.features.SparseFeature
import torchrec.basic.metrics.AUC
import torchrec.data.{DataLoader, MultiTaskDataset}
import torchrec.models.multi_task.*
import torchrec.trainers.MTLTrainer

import scala.collection.mutable
import scala.io.Source
import scala.util.Random


//================ MULTI-TASK PIPELINE SUMMARY ================
//  [PASS] MMOE (multitask) -> fraud_AUC=0.0586, high_amount_LogLoss=0.9811, fraud_Accuracy=0.1120, high_amount_AUC=0.1901, high_amount_Accuracy=0.0560, fraud_LogLoss=0.9050
//  [PASS] OMoE (multitask) -> fraud_AUC=0.0525, high_amount_LogLoss=0.8277, fraud_Accuracy=0.1120, high_amount_AUC=0.1783, high_amount_Accuracy=0.0560, fraud_LogLoss=0.9287
//  [PASS] PLE (multitask) -> fraud_AUC=0.0480, high_amount_LogLoss=0.9372, fraud_Accuracy=0.1120, high_amount_AUC=0.1936, high_amount_Accuracy=0.0560, fraud_LogLoss=0.9296
//  [PASS] AITM (multitask) -> fraud_AUC=0.0561, high_amount_LogLoss=0.9567, fraud_Accuracy=0.1120, high_amount_AUC=0.1774, high_amount_Accuracy=0.0560, fraud_LogLoss=0.8888
//  [PASS] MetaHeac (multitask) -> fraud_AUC=0.0490, high_amount_LogLoss=0.9546, fraud_Accuracy=0.1120, high_amount_AUC=0.1876, high_amount_Accuracy=0.0560, fraud_LogLoss=0.9162
//  [PASS] ESMM (multitask) -> fraud_AUC=0.0467, high_amount_LogLoss=0.9263, fraud_Accuracy=0.1120, high_amount_AUC=0.1514, high_amount_Accuracy=0.0560, fraud_LogLoss=0.8961
//  [PASS] SharedBottom (multitask) -> fraud_AUC=0.0535, high_amount_LogLoss=0.9467, fraud_Accuracy=0.1120, high_amount_AUC=0.1684, high_amount_Accuracy=0.0560, fraud_LogLoss=0.9326
//  [PASS] SingleTaskModel (multitask) -> fraud_AUC=0.0654, high_amount_LogLoss=0.9570, fraud_Accuracy=0.1120, high_amount_AUC=0.1904, high_amount_Accuracy=0.0560, fraud_LogLoss=0.9322

/**
 * Multi-Task Pipeline Benchmark
 *
 * Tests all multi-task models: MMOE, OMoE, PLE, AITM, MetaHeac, ESMM, SharedBottom, SingleTaskModel
 * Uses the fraud dataset (fraud_data.csv) with fraud and high_amount tasks.
 */
object RunMultiTaskPipeline {

  // 设置 CUDA 内存分配配置，避免碎片化导致 OOM
  System.setProperty("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

  case class EncodedRow(
    ids: Array[Int],
    dense: Array[Float],
    label: Float,
    highAmount: Float,
    userId: Int,
    itemId: Int,
    tokens: Array[Int],
    itemVec: Array[Float]
  )

  case class FraudPrepared(
    train: Vector[EncodedRow],
    valid: Vector[EncodedRow],
    test: Vector[EncodedRow],
    numFeatures: Int,
    numBins: Int,
    seqLen: Int,
    userVocab: Int,
    itemVocab: Int
  )

  case class ModelReport(name: String, task: String, ok: Boolean, metrics: Map[String, Float], note: String)

  def main(args: Array[String]): Unit = {
    val csvPath = "/home/muller/IdeaProjects/torch-rechub-scala/src/main/resources/fraud_data.csv"
    val maxRows = 21900
    val batchSize = 256
    val epochs = 5
    val device = "cuda"
    val seed = 2026
    val bins = 128
    val seqLen = 20

    val prepared = prepareFraud(csvPath, maxRows, bins, seqLen, seed, itemEmbedDim = 16)

    val reports = mutable.ArrayBuffer[ModelReport]()

    reports ++= runMultiTaskFraud(prepared, epochs, batchSize, device)

    println("\n================ MULTI-TASK PIPELINE SUMMARY ================")
    reports.foreach { r =>
      val m = r.metrics.map { case (k, v) => s"$k=${"%.4f".format(v)}" }.mkString(", ")
      println(s"[${if (r.ok) "PASS" else "FAIL"}] ${r.name} (${r.task}) -> ${if (m.nonEmpty) m else r.note}")
      if (!r.ok) println(s"  reason: ${r.note}")
    }
  }

  private def runReport(name: String, task: String)(fn: => Map[String, Float]): ModelReport = {
    try {
      val m = fn
      ModelReport(name, task, ok = true, m, "")
    } catch {
      case e: Throwable => ModelReport(name, task, ok = false, Map.empty, Option(e.getMessage).getOrElse(e.getClass.getSimpleName))
    }
  }

  private def runMultiTaskFraud(prepared: FraudPrepared, epochs: Int, batchSize: Int, device: String): Seq[ModelReport] = {
    val reports = mutable.ArrayBuffer[ModelReport]()
    val (tr, va, te, features) = buildMultiTaskData(prepared)
    val trainLoader = new DataLoader(tr, batchSize = batchSize, shuffle = true, device = device)
    val validLoader = new DataLoader(va, batchSize = batchSize, shuffle = false, device = device)
    val testLoader = new DataLoader(te, batchSize = batchSize, shuffle = false, device = device)

    // Diagnostic: inspect a single batch to verify shapes/dtypes of features and labels
    try {
      val it = trainLoader.iterator
      if (it.hasNext) {
        val b = it.next()
        System.err.println(s"[RunMultiTaskPipeline DEBUG] batch sparse keys=${b.sparseFeatures.keys.mkString(",")} ")
        b.sparseFeatures.foreach { case (k, v) => System.err.println(s"  feat=$k shape=${v.sizes().toString} dtype=${v.dtype()}") }
        b.taskLabels.foreach { m => m.foreach { case (t, v) => System.err.println(s"  task=$t shape=${v.sizes().toString} dtype=${v.dtype()} sample=${v.squeeze().to( org.bytedeco.pytorch.global.torch.ScalarType.Float).contiguous().cpu().toFloatArray.take(5).mkString(",")}") } }
      }
    } catch { case _: Throwable => () }

    // Diagnostic: run a forward pass through a fresh MMOE to inspect outputs
    try {
      val dbgModel = new MMOE((0 until prepared.numFeatures).map(i => SparseFeature(s"feat_$i", prepared.numBins + 2, 8)).toList, taskTypes = List("classification","classification"), nExpert = 2, expertParams = Map("dims" -> List(32L)), towerParamsList = List(Map("dims" -> List(16L)), Map("dims" -> List(16L))), device = device)
      val it2 = trainLoader.iterator
      if (it2.hasNext) {
        val b = it2.next()
        val out = dbgModel.forward(b.sparseFeatures)
        try {
          System.err.println(s"[RunMultiTaskPipeline DEBUG] model output shape=${out.sizes().toString} dtype=${out.dtype()}")
          val samples = out.contiguous().cpu().toType(org.bytedeco.pytorch.global.torch.ScalarType.Float).toFloatArray.take(10)
          System.err.println(s"[RunMultiTaskPipeline DEBUG] output samples=${samples.mkString(",")}")
        } catch { case _: Throwable => () }
      }
    } catch { case e: Throwable => System.err.println(s"[RunMultiTaskPipeline DEBUG] forward failed: ${e.getMessage}") }

    def trainMtl(name: String, model: Module): ModelReport = runReport(name, "multitask") {
      // 训练前清理 CUDA 缓存，避免前面模型残留显存
      if (device == "cuda") {
        torch.emptyCache()
        System.gc()
      }

      val trainer = new MTLTrainer(model, taskNames = List("fraud", "high_amount"), learningRate = 1e-3f, device = device, numEpochs = epochs, verbose = true)
      trainer.fit(trainLoader, Some(validLoader))
      val metrics = trainer.evaluate(testLoader)

      // 训练后清理 CUDA 缓存
      if (device == "cuda") {
        torch.emptyCache()
        System.gc()
      }

      metrics
    }

    // MMOE
    reports += trainMtl("MMOE", new MMOE(features, taskTypes = List("classification", "classification"), nExpert = 4, expertParams = Map("dims" -> List(64L)), towerParamsList = List(Map("dims" -> List(32L)), Map("dims" -> List(32L))), device = device))

    // OMoE
    reports += trainMtl("OMoE", new OMoE(features, taskNames = List("fraud", "high_amount"), embedDim = 8, numExperts = 4, expertDims = List(64L), towerDims = List(32L), device = device))

    // PLE
    reports += trainMtl("PLE", new PLE(features, taskTypes = List("classification", "classification"), nLevel = 1, nExpertSpecific = 2, nExpertShared = 1, expertParams = Map("dims" -> List(64L)), towerParamsList = List(Map("dims" -> List(32L)), Map("dims" -> List(32L))), device = device))

    // AITM
    reports += trainMtl("AITM", new AITM(features, nTask = 2, bottomParams = Map("dims" -> List(64L, 32L)), towerParamsList = List(Map("dims" -> List(16L)), Map("dims" -> List(16L))), device = device))

    // MetaHeac
    reports += trainMtl("MetaHeac", new MetaHeac(features, taskNames = List("fraud", "high_amount"), embedDim = 8, bottomDims = List(64L, 32L), towerDims = List(32L, 16L), expertNum = 4, criticNum = 3, dropout = 0.2f, device = device))

    // ESMM
    reports += trainMtl("ESMM", ESMM(features, taskNames = List("fraud", "high_amount"), embedDim = 8, towerDims = List(64L, 32L), dropout = 0.2f, device = device))

    // SharedBottom
    reports += trainMtl("SharedBottom", new SharedBottom(features, taskTypes = List("classification", "classification"), bottomParams = Map("dims" -> List(64L)), towerParamsList = List(Map("dims" -> List(32L)), Map("dims" -> List(32L))), device = device))

    // SingleTaskModel
    reports += trainMtl("SingleTaskModel", new SingleTaskModel(features, taskNames = List("fraud", "high_amount"), embedDim = 8, bottomDims = List(64L), towerDims = List(32L), dropout = 0.2f, device = device))

    reports.toSeq
  }

  private def prepareFraud(path: String, maxRows: Int, numBins: Int, seqLen: Int, seed: Int, itemEmbedDim: Int): FraudPrepared = {
    val src = Source.fromFile(path)
    val lines = try src.getLines().drop(1).toVector finally src.close()
    val parsed = lines.flatMap { line =>
      val arr = line.split(",", -1)
      if (arr.length == 30) {
        val feats = arr.take(29).map(_.toDouble)
        val label = arr(29).toFloat
        Some((feats, label))
      } else None
    }

    val positives = parsed.filter(_._2 > 0.5f)
    val negatives = Random(seed).shuffle(parsed.filter(_._2 <= 0.5f))
    val keepNeg = math.min(math.max(positives.size * 8, 1), math.max(maxRows - positives.size, 1))
    val balanced = Random(seed + 1).shuffle((positives ++ negatives.take(keepNeg)).toVector).take(maxRows)

    val (trainRaw, validRaw, testRaw) = splitStratified(balanced, seed)

    val mins = Array.fill(29)(Double.MaxValue)
    val maxs = Array.fill(29)(Double.MinValue)
    trainRaw.foreach { case (f, _) =>
      f.indices.foreach(i => {
        if (f(i) < mins(i)) mins(i) = f(i)
        if (f(i) > maxs(i)) maxs(i) = f(i)
      })
    }

    def normalize(feats: Array[Double]): Array[Float] = feats.indices.map { i =>
      if (maxs(i) > mins(i)) ((feats(i) - mins(i)) / (maxs(i) - mins(i))).toFloat
      else 0.0f
    }.toArray

    def buildRows(rows: Vector[(Array[Double], Float)]): Vector[EncodedRow] = {
      rows.map { case (f, l) =>
        val ids = normalize(f).grouped(1).map(_(0) * numBins).map(_.toInt).map(i => math.abs(i) % (numBins + 2)).toArray
        EncodedRow(
          ids = ids,
          dense = normalize(f),
          label = l,
          highAmount = if (f.slice(0, 10).sum / 10 > 0.5) 1.0f else 0.0f,
          userId = math.abs(ids.hashCode) % 10000,
          itemId = math.abs(ids.take(5).hashCode) % 10000,
          tokens = ids.take(seqLen),
          itemVec = Array.fill(itemEmbedDim)(0.01f)
        )
      }
    }

    FraudPrepared(
      train = buildRows(trainRaw),
      valid = buildRows(validRaw),
      test = buildRows(testRaw),
      numFeatures = 29,
      numBins = numBins,
      seqLen = seqLen,
      userVocab = 10000,
      itemVocab = 10000
    )
  }

  private def splitStratified(data: Vector[(Array[Double], Float)], seed: Int): (Vector[(Array[Double], Float)], Vector[(Array[Double], Float)], Vector[(Array[Double], Float)]) = {
    val pos = data.filter(_._2 > 0.5f)
    val neg = data.filter(_._2 <= 0.5f)
    val rng = Random(seed)
    def split[T](v: Vector[T]): (Vector[T], Vector[T], Vector[T]) = {
      val shuffled = rng.shuffle(v)
      val n = shuffled.size
      val t = (n * 0.6).toInt
      val v2 = (n * 0.2).toInt
      (shuffled.take(t), shuffled.slice(t, t + v2), shuffled.slice(t + v2, n))
    }
    val (trp, vap, tep) = split(pos)
    val (trn, van, ten) = split(neg)
    ((trp ++ trn).sortBy(_._2), (vap ++ van).sortBy(_._2), (tep ++ ten).sortBy(_._2))
  }

  private def buildMultiTaskData(prepared: FraudPrepared): (MultiTaskDataset, MultiTaskDataset, MultiTaskDataset, List[SparseFeature]) = {
    def ds(rows: Vector[EncodedRow]): MultiTaskDataset = {
      val features = buildSparseFeatureMap(rows, prepared.numFeatures)
      val labels = Map(
        "fraud" -> tensor(rows.map(_.label).toArray, Array(rows.size.toLong)),
        "high_amount" -> tensor(rows.map(_.highAmount).toArray, Array(rows.size.toLong))
      )
      new MultiTaskDataset(features, labels)
    }
    val feats = (0 until prepared.numFeatures).map(i => SparseFeature(s"feat_$i", prepared.numBins + 2, 8)).toList
    (ds(prepared.train), ds(prepared.valid), ds(prepared.test), feats)
  }

  private def buildSparseFeatureMap(rows: Vector[EncodedRow], numFeatures: Int): Map[String, Tensor] = {
    val n = rows.size
    // Build a dense feature tensor (row-major: each row is a sample)
    val featureData = Array.ofDim[Float](n * numFeatures)
    rows.zipWithIndex.foreach { case (row, i) =>
      var j = 0
      while (j < numFeatures && j < row.ids.length) {
        featureData(i * numFeatures + j) = row.ids(j).toFloat
        j += 1
      }
      while (j < numFeatures) {
        featureData(i * numFeatures + j) = 0.0f
        j += 1
      }
    }
    // Create features as "feat_0", "feat_1", etc. like AliExpressDataset
    (0 until numFeatures).map { i =>
      val colData = (0 until n).map(row => featureData(row * numFeatures + i)).toArray
      s"feat_$i" -> tensor(colData, Array(n.toLong)).toType(org.bytedeco.pytorch.global.torch.ScalarType.Long)
    }.toMap
  }
}