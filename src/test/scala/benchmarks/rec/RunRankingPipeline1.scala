package benchmarks.rec

import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import org.bytedeco.pytorch.*
import torchrec.Implicits.{RichTensor, tensor}
import torchrec.basic.features.{Feature, SparseFeature}
import torchrec.basic.losses.BCEWithLogitsLoss
import torchrec.basic.metrics.{AUC, Accuracy, LogLoss}
import torchrec.data.{DataLoader, TensorDataset}
import torchrec.models.ranking.*
import torchrec.trainers.CTRTrainer

import scala.collection.mutable
import scala.io.Source
import scala.util.Random
//================ RANKING PIPELINE 1 SUMMARY ================
//[PASS] AFN (ranking) -> AUC=0.5675, LogLoss=0.6364, Accuracy=0.7801
//[PASS] NFM (ranking) -> AUC=0.5216, LogLoss=0.6051, Accuracy=0.7178
//[PASS] FNN (ranking) -> AUC=0.5811, LogLoss=0.6899, Accuracy=0.5602
//[PASS] PNN (ranking) -> AUC=0.4955, LogLoss=0.6625, Accuracy=0.7531
//[PASS] HoFM (ranking) -> AUC=0.4387, LogLoss=3.5138, Accuracy=0.1598
//[PASS] FNFM (ranking) -> AUC=0.4082, LogLoss=0.6646, Accuracy=0.8776
//[PASS] LR (ranking) -> AUC=0.6193, LogLoss=2.9841, Accuracy=0.6701
//[PASS] DeepFM (ranking) -> AUC=0.4934, LogLoss=5.5097, Accuracy=0.6037
//[PASS] WideDeep (ranking) -> AUC=0.5784, LogLoss=2.4611, Accuracy=0.7220
//[PASS] DCN (ranking) -> AUC=0.8880, LogLoss=0.2573, Accuracy=0.8983
//[PASS] DCNv2 (ranking) -> AUC=0.8841, LogLoss=0.2534, Accuracy=0.8963
//[PASS] AFM (ranking) -> AUC=0.5459, LogLoss=4.5618, Accuracy=0.5954
//[PASS] FiBiNet (ranking) -> AUC=0.6493, LogLoss=0.6715, Accuracy=0.7988
/**
 * Benchmark pipeline for non-sequence ranking models.
 * Tests: DeepFM, WideDeep, DCN, DCNv2, AFM, AFN, FiBiNet, NFM, FNN, PNN, HoFM, FNFM, LNN, LR
 */
object RunRankingPipeline1 {

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
    val p = PythonRankingReplicaSupport.parseArgs(args)
    val csvPath = p.getOrElse("dataset", "/home/muller/IdeaProjects/torch-rechub-scala/src/main/resources/fraud_data.csv")
    val maxRows = p.getOrElse("max_rows", "21900").toInt
    val batchSize = p.getOrElse("batch_size", "256").toInt
    val epochs = p.getOrElse("epochs", "5").toInt
    val device = p.getOrElse("device", "cuda")
    val seed = p.getOrElse("seed", "2026").toInt
    val bins = p.getOrElse("bins", "128").toInt
    val seqLen = p.getOrElse("seq_len", "20").toInt

    val prepared = prepareFraud(csvPath, maxRows, bins, seqLen, seed, itemEmbedDim = 16)
    val reports = mutable.ArrayBuffer[ModelReport]()

    // 训练前清理 CUDA 缓存
    if (device == "cuda") {
      torch.emptyCache()
      System.gc()
    }


    // ==================== AFN ====================
    reports += runReport("AFN", "ranking") {
      val features = (0 until prepared.numFeatures).map(i => SparseFeature(s"feat_$i", prepared.numBins + 2, 8)).toList
      val (tr, va, te, _) = buildRankingData(prepared, embedDim = 8)
      val model = new AFN(features, embedDim = 8, lnnDim = 8, mlpDims = List(128L, 64L, 32L), dropout = 0.2f, device = device)
      trainAFN(model, tr, epochs, batchSize, device)
      evalAFN(model, te, batchSize, device)
    }



    // ==================== NFM ====================
    reports += runReport("NFM", "ranking") {
      val features = (0 until prepared.numFeatures).map(i => SparseFeature(s"feat_$i", prepared.numBins + 2, 8)).toList
      val (tr, va, te, _) = buildRankingData(prepared, embedDim = 8)
      val model = new NFM(features, embedDim = 8, mlpDims = List(64L, 32L), dropout = 0.2f, device = device)
      trainNFM(model, tr, epochs, batchSize, device)
      evalNFM(model, te, batchSize, device)
    }

    // ==================== FNN ====================
    reports += runReport("FNN", "ranking") {
      val features = (0 until prepared.numFeatures).map(i => SparseFeature(s"feat_$i", prepared.numBins + 2, 8)).toList
      val (tr, va, te, _) = buildRankingData(prepared, embedDim = 8)
      val model = new FNN(features, embedDim = 8, mlpDims = List(128L, 64L, 32L), dropout = 0.2f, device = device)
      trainFNN(model, tr, epochs, batchSize, device)
      evalFNN(model, te, batchSize, device)
    }

    // ==================== PNN ====================
    reports += runReport("PNN", "ranking") {
      val features = (0 until prepared.numFeatures).map(i => SparseFeature(s"feat_$i", prepared.numBins + 2, 8)).toList
      val (tr, va, te, _) = buildRankingData(prepared, embedDim = 8)
      val model = new PNN(features, embedDim = 8, mlpDims = List(128L, 64L, 32L), productType = "inner", dropout = 0.2f, device = device)
      trainPNN(model, tr, epochs, batchSize, device)
      evalPNN(model, te, batchSize, device)
    }

    // ==================== HoFM ====================
    reports += runReport("HoFM", "ranking") {
      val features = (0 until prepared.numFeatures).map(i => SparseFeature(s"feat_$i", prepared.numBins + 2, 8)).toList
      val (tr, va, te, _) = buildRankingData(prepared, embedDim = 8)
      val model = new HoFM(features, embedDim = 8, order = 3, mlpDims = List(64L, 32L), dropout = 0.2f, device = device)
      trainHoFM(model, tr, epochs, batchSize, device)
      evalHoFM(model, te, batchSize, device)
    }

    // ==================== FNFM ====================
    reports += runReport("FNFM", "ranking") {
      val features = (0 until prepared.numFeatures).map(i => SparseFeature(s"feat_$i", prepared.numBins + 2, 8)).toList
      val (tr, va, te, _) = buildRankingData(prepared, embedDim = 8)
      val model = new FNFM(features, embedDim = 8, mlpDims = List(128L, 64L, 32L), dropout = 0.2f, device = device)
      trainFNFM(model, tr, epochs, batchSize, device)
      evalFNFM(model, te, batchSize, device)
    }

    // ==================== LR ====================
    reports += runReport("LR", "ranking") {
      val features = (0 until prepared.numFeatures).map(i => SparseFeature(s"feat_$i", prepared.numBins + 2, 8)).toList
      val (tr, va, te, _) = buildRankingData(prepared, embedDim = 8)
      val model = new LR(features, embedDim = 8, device = device)
      trainLR(model, tr, epochs, batchSize, device)
      evalLR(model, te, batchSize, device)
    }

    // ==================== DeepFM ====================
    reports += runReport("DeepFM", "ranking") {
      val features = (0 until prepared.numFeatures).map(i => SparseFeature(s"feat_$i", prepared.numBins + 2, 8)).toList
      val deepFeatures = features.take(20)
      val fmFeatures = features
      val (tr, va, te, _) = buildRankingData(prepared, embedDim = 8)
      val model = new DeepFM(deepFeatures, fmFeatures, embedDim = 8, mlpDims = List(128L, 64L), dropout = 0.2f, device = device)
      trainDeepFM(model, tr, epochs, batchSize, device)
      evalDeepFM(model, te, batchSize, device)
    }

    // ==================== WideDeep ====================
    reports += runReport("WideDeep", "ranking") {
      val features = (0 until prepared.numFeatures).map(i => SparseFeature(s"feat_$i", prepared.numBins + 2, 8)).toList
      val (tr, va, te, _) = buildRankingData(prepared, embedDim = 8)
      val model = new WideDeep(features, embedDim = 8, mlpDims = List(128L, 64L), dropout = 0.2f, device = device)
      trainWideDeep(model, tr, epochs, batchSize, device)
      evalWideDeep(model, te, batchSize, device)
    }

    // ==================== DCN ====================
    reports += runReport("DCN", "ranking") {
      val features = (0 until prepared.numFeatures).map(i => SparseFeature(s"feat_$i", prepared.numBins + 2, 8)).toList
      val (tr, va, te, _) = buildRankingData(prepared, embedDim = 8)
      val model = new DCN(features, embedDim = 8, numCrossLayers = 3, mlpDims = List(128L, 64L), dropout = 0.2f, device = device)
      trainDCN(model, tr, epochs, batchSize, device)
      evalDCN(model, te, batchSize, device)
    }

    // ==================== DCNv2 ====================
    reports += runReport("DCNv2", "ranking") {
      val features = (0 until prepared.numFeatures).map(i => SparseFeature(s"feat_$i", prepared.numBins + 2, 8)).toList
      val (tr, va, te, _) = buildRankingData(prepared, embedDim = 8)
      val model = new DCNv2(features, embedDim = 8, numCrossLayers = 3, useCrossNetMix = true, lowRank = 4, mlpDims = List(128L, 64L), dropout = 0.2f, device = device)
      trainDCNv2(model, tr, epochs, batchSize, device)
      evalDCNv2(model, te, batchSize, device)
    }

    // ==================== AFM ====================
    reports += runReport("AFM", "ranking") {
      val features = (0 until prepared.numFeatures).map(i => SparseFeature(s"feat_$i", prepared.numBins + 2, 8)).toList
      val (tr, va, te, _) = buildRankingData(prepared, embedDim = 8)
      val model = new AFM(features, embedDim = 8, attentionDim = 64, dropout = 0.2f, device = device)
      trainAFM(model, tr, epochs, batchSize, device)
      evalAFM(model, te, batchSize, device)
    }

    // ==================== FiBiNet ====================
    reports += runReport("FiBiNet", "ranking") {
      val features = (0 until prepared.numFeatures).map(i => SparseFeature(s"feat_$i", prepared.numBins + 2, 8)).toList
      val (tr, va, te, _) = buildRankingData(prepared, embedDim = 8)
      val model = new FiBiNet(features, embedDim = 8, mlpDims = List(128L, 64L), reduction = 3, bilinearType = "field_interaction", dropout = 0.2f, device = device)
      trainFiBiNet(model, tr, epochs, batchSize, device)
      evalFiBiNet(model, te, batchSize, device)
    }

    // 打印总结
    println("\n================ RANKING PIPELINE 1 SUMMARY ================")
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
      var i = 0
      while (i < 29) {
        mins(i) = math.min(mins(i), f(i))
        maxs(i) = math.max(maxs(i), f(i))
        i += 1
      }
    }

    val trainAmounts = trainRaw.map(_._1(28)).sorted
    val amountMedian = if (trainAmounts.nonEmpty) trainAmounts(trainAmounts.size / 2) else 0.0

    def encode(rows: Vector[(Array[Double], Float)]): Vector[EncodedRow] = rows.zipWithIndex.map { case ((f, y), idx) =>
      val ids = Array.ofDim[Int](29)
      val dense = Array.ofDim[Float](29)
      var i = 0
      while (i < 29) {
        val denom = math.max(1e-12, maxs(i) - mins(i))
        val ratio = ((f(i) - mins(i)) / denom).max(0.0).min(1.0)
        ids(i) = (ratio * (numBins - 1)).toInt + 1
        dense(i) = ratio.toFloat
        i += 1
      }
      val userId = ((ids(0) * 131 + ids(1) * 17 + ids(2) * 7) % 50000) + 1
      val itemId = ((ids(3) * 97 + ids(4) * 13 + ids(5) * 5 + ids(28)) % 20000) + 1
      val tokens = Array.tabulate(seqLen)(j => ids(j % ids.length))
      val itemVec = Array.tabulate(itemEmbedDim)(j => (((itemId * (j + 3)) % 97).toFloat / 97.0f) - 0.5f)
      EncodedRow(ids, dense, y, if (f(28) > amountMedian) 1.0f else 0.0f, userId, itemId, tokens, itemVec)
    }

    val train = encode(trainRaw)
    val valid = encode(validRaw)
    val test = encode(testRaw)

    FraudPrepared(train, valid, test, numFeatures = 29, numBins = numBins, seqLen = seqLen, userVocab = 50000, itemVocab = 20000)
  }

  private def splitStratified(data: Vector[(Array[Double], Float)], seed: Int): (Vector[(Array[Double], Float)], Vector[(Array[Double], Float)], Vector[(Array[Double], Float)]) = {
    val pos = Random(seed).shuffle(data.filter(_._2 > 0.5f))
    val neg = Random(seed + 1).shuffle(data.filter(_._2 <= 0.5f))
    def splitOne[T](xs: Vector[T]): (Vector[T], Vector[T], Vector[T]) = {
      val n = xs.size
      val tr = (n * 0.7).toInt
      val va = (n * 0.15).toInt
      (xs.take(tr), xs.slice(tr, tr + va), xs.drop(tr + va))
    }
    val (ptr, pva, pte) = splitOne(pos)
    val (ntr, nva, nte) = splitOne(neg)
    (
      Random(seed + 2).shuffle((ptr ++ ntr).toVector),
      Random(seed + 3).shuffle((pva ++ nva).toVector),
      Random(seed + 4).shuffle((pte ++ nte).toVector)
    )
  }

  private def buildSparseFeatureMap(rows: Vector[EncodedRow], nFeat: Int): Map[String, Tensor] = {
    (0 until nFeat).map { i =>
      val arr = rows.map(_.ids(i).toFloat).toArray
      s"feat_$i" -> tensor(arr, Array(rows.size.toLong)).toType(ScalarType.Long)
    }.toMap
  }

  private def buildRankingData(prepared: FraudPrepared, embedDim: Int): (TensorDataset, TensorDataset, TensorDataset, List[SparseFeature]) = {
    def ds(rows: Vector[EncodedRow]): TensorDataset = {
      val sparse = buildSparseFeatureMap(rows, prepared.numFeatures)
      val y = tensor(rows.map(_.label).toArray, Array(rows.size.toLong))
      new TensorDataset(sparseFeatures = sparse, denseFeatures = Map.empty, labels = Some(y))
    }
    val features = (0 until prepared.numFeatures).map(i => SparseFeature(s"feat_$i", prepared.numBins + 2, embedDim)).toList
    (ds(prepared.train), ds(prepared.valid), ds(prepared.test), features)
  }

  // ==================== DeepFM ====================
  private def trainDeepFM(model: DeepFM, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        b.labels.foreach { yRaw =>
          optimizer.zero_grad()
          val logits = model.forward(b.sparseFeatures)
          val bs = yRaw.size(0).toInt
          val logits2 = if (logits.dim() == 1L) logits.unsqueeze(1L) else logits
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits2, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble
          numBatches += 1
          val preds = logits2.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          val labels = y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          epochPreds.appendAll(preds)
          epochLabels.appendAll(labels)
        }
      }
      val avgLoss = totalLoss / numBatches
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute()
      aucMetric.reset()
      println(f"[DeepFM] Epoch ${e + 1}/$epochs - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalDeepFM(model: DeepFM, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      b.labels.foreach { yRaw =>
        val logits = model.forward(b.sparseFeatures)
        val p = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val y = yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        preds.appendAll(p)
        labels.appendAll(y)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray)
    ll.update(preds.toArray, labels.toArray)
    acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== WideDeep ====================
  private def trainWideDeep(model: WideDeep, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        b.labels.foreach { yRaw =>
          optimizer.zero_grad()
          val logits = model.forward(b.sparseFeatures)
          val bs = yRaw.size(0).toInt
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble
          numBatches += 1
          val preds = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          val labels = y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          epochPreds.appendAll(preds)
          epochLabels.appendAll(labels)
        }
      }
      val avgLoss = totalLoss / numBatches
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute()
      aucMetric.reset()
      println(f"[WideDeep] Epoch ${e + 1}/$epochs - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalWideDeep(model: WideDeep, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      b.labels.foreach { yRaw =>
        val logits = model.forward(b.sparseFeatures)
        val p = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val y = yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        preds.appendAll(p)
        labels.appendAll(y)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray)
    ll.update(preds.toArray, labels.toArray)
    acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== DCN ====================
  private def trainDCN(model: DCN, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        b.labels.foreach { yRaw =>
          optimizer.zero_grad()
          val logits = model.forward(b.sparseFeatures)
          val bs = yRaw.size(0).toInt
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble
          numBatches += 1
          val preds = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          val labels = y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          epochPreds.appendAll(preds)
          epochLabels.appendAll(labels)
        }
      }
      val avgLoss = totalLoss / numBatches
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute()
      aucMetric.reset()
      println(f"[DCN] Epoch ${e + 1}/$epochs - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalDCN(model: DCN, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      b.labels.foreach { yRaw =>
        val logits = model.forward(b.sparseFeatures)
        val p = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val y = yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        preds.appendAll(p)
        labels.appendAll(y)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray)
    ll.update(preds.toArray, labels.toArray)
    acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== DCNv2 ====================
  private def trainDCNv2(model: DCNv2, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        b.labels.foreach { yRaw =>
          optimizer.zero_grad()
          val logits = model.forward(b.sparseFeatures)
          val bs = yRaw.size(0).toInt
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble
          numBatches += 1
          val preds = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          val labels = y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          epochPreds.appendAll(preds)
          epochLabels.appendAll(labels)
        }
      }
      val avgLoss = totalLoss / numBatches
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute()
      aucMetric.reset()
      println(f"[DCNv2] Epoch ${e + 1}/$epochs - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalDCNv2(model: DCNv2, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      b.labels.foreach { yRaw =>
        val logits = model.forward(b.sparseFeatures)
        val p = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val y = yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        preds.appendAll(p)
        labels.appendAll(y)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray)
    ll.update(preds.toArray, labels.toArray)
    acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== AFM ====================
  private def trainAFM(model: AFM, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        b.labels.foreach { yRaw =>
          optimizer.zero_grad()
          val logits = model.forward(b.sparseFeatures)
          val bs = yRaw.size(0).toInt
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble
          numBatches += 1
          val preds = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          val labels = y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          epochPreds.appendAll(preds)
          epochLabels.appendAll(labels)
        }
      }
      val avgLoss = totalLoss / numBatches
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute()
      aucMetric.reset()
      println(f"[AFM] Epoch ${e + 1}/$epochs - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalAFM(model: AFM, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      b.labels.foreach { yRaw =>
        val logits = model.forward(b.sparseFeatures)
        val p = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val y = yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        preds.appendAll(p)
        labels.appendAll(y)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray)
    ll.update(preds.toArray, labels.toArray)
    acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== AFN ====================
  private def trainAFN(model: AFN, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        b.labels.foreach { yRaw =>
          optimizer.zero_grad()
          val logits = model.forward(b.sparseFeatures)
          val bs = yRaw.size(0).toInt
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble
          numBatches += 1
          val preds = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          val labels = y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          epochPreds.appendAll(preds)
          epochLabels.appendAll(labels)
        }
      }
      val avgLoss = totalLoss / numBatches
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute()
      aucMetric.reset()
      println(f"[AFN] Epoch ${e + 1}/$epochs - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalAFN(model: AFN, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      b.labels.foreach { yRaw =>
        val logits = model.forward(b.sparseFeatures)
        val p = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val y = yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        preds.appendAll(p)
        labels.appendAll(y)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray)
    ll.update(preds.toArray, labels.toArray)
    acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== FiBiNet ====================
  private def trainFiBiNet(model: FiBiNet, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        b.labels.foreach { yRaw =>
          optimizer.zero_grad()
          val logits = model.forward(b.sparseFeatures)
          val bs = yRaw.size(0).toInt
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble
          numBatches += 1
          val preds = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          val labels = y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          epochPreds.appendAll(preds)
          epochLabels.appendAll(labels)
        }
      }
      val avgLoss = totalLoss / numBatches
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute()
      aucMetric.reset()
      println(f"[FiBiNet] Epoch ${e + 1}/$epochs - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalFiBiNet(model: FiBiNet, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      b.labels.foreach { yRaw =>
        val logits = model.forward(b.sparseFeatures)
        val p = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val y = yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        preds.appendAll(p)
        labels.appendAll(y)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray)
    ll.update(preds.toArray, labels.toArray)
    acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== NFM ====================
  private def trainNFM(model: NFM, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        b.labels.foreach { yRaw =>
          optimizer.zero_grad()
          val logits = model.forward(b.sparseFeatures)
          val bs = yRaw.size(0).toInt
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble
          numBatches += 1
          val preds = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          val labels = y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          epochPreds.appendAll(preds)
          epochLabels.appendAll(labels)
        }
      }
      val avgLoss = totalLoss / numBatches
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute()
      aucMetric.reset()
      println(f"[NFM] Epoch ${e + 1}/$epochs - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalNFM(model: NFM, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      b.labels.foreach { yRaw =>
        val logits = model.forward(b.sparseFeatures)
        val p = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val y = yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        preds.appendAll(p)
        labels.appendAll(y)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray)
    ll.update(preds.toArray, labels.toArray)
    acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== FNN ====================
  private def trainFNN(model: FNN, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        b.labels.foreach { yRaw =>
          optimizer.zero_grad()
          val logits = model.forward(b.sparseFeatures)
          val bs = yRaw.size(0).toInt
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble
          numBatches += 1
          val preds = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          val labels = y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          epochPreds.appendAll(preds)
          epochLabels.appendAll(labels)
        }
      }
      val avgLoss = totalLoss / numBatches
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute()
      aucMetric.reset()
      println(f"[FNN] Epoch ${e + 1}/$epochs - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalFNN(model: FNN, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      b.labels.foreach { yRaw =>
        val logits = model.forward(b.sparseFeatures)
        val p = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val y = yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        preds.appendAll(p)
        labels.appendAll(y)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray)
    ll.update(preds.toArray, labels.toArray)
    acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== PNN ====================
  private def trainPNN(model: PNN, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        b.labels.foreach { yRaw =>
          optimizer.zero_grad()
          val logits = model.forward(b.sparseFeatures)
          val bs = yRaw.size(0).toInt
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble
          numBatches += 1
          val preds = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          val labels = y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          epochPreds.appendAll(preds)
          epochLabels.appendAll(labels)
        }
      }
      val avgLoss = totalLoss / numBatches
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute()
      aucMetric.reset()
      println(f"[PNN] Epoch ${e + 1}/$epochs - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalPNN(model: PNN, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      b.labels.foreach { yRaw =>
        val logits = model.forward(b.sparseFeatures)
        val p = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val y = yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        preds.appendAll(p)
        labels.appendAll(y)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray)
    ll.update(preds.toArray, labels.toArray)
    acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== HoFM ====================
  private def trainHoFM(model: HoFM, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        b.labels.foreach { yRaw =>
          optimizer.zero_grad()
          val logits = model.forward(b.sparseFeatures)
          val bs = yRaw.size(0).toInt
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble
          numBatches += 1
          val preds = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          val labels = y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          epochPreds.appendAll(preds)
          epochLabels.appendAll(labels)
        }
      }
      val avgLoss = totalLoss / numBatches
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute()
      aucMetric.reset()
      println(f"[HoFM] Epoch ${e + 1}/$epochs - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalHoFM(model: HoFM, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      b.labels.foreach { yRaw =>
        val logits = model.forward(b.sparseFeatures)
        val p = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val y = yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        preds.appendAll(p)
        labels.appendAll(y)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray)
    ll.update(preds.toArray, labels.toArray)
    acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== FNFM ====================
  private def trainFNFM(model: FNFM, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        b.labels.foreach { yRaw =>
          optimizer.zero_grad()
          val logits = model.forward(b.sparseFeatures)
          val bs = yRaw.size(0).toInt
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble
          numBatches += 1
          val preds = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          val labels = y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          epochPreds.appendAll(preds)
          epochLabels.appendAll(labels)
        }
      }
      val avgLoss = totalLoss / numBatches
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute()
      aucMetric.reset()
      println(f"[FNFM] Epoch ${e + 1}/$epochs - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalFNFM(model: FNFM, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      b.labels.foreach { yRaw =>
        val logits = model.forward(b.sparseFeatures)
        val p = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val y = yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        preds.appendAll(p)
        labels.appendAll(y)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray)
    ll.update(preds.toArray, labels.toArray)
    acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== LR ====================
  private def trainLR(model: LR, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        b.labels.foreach { yRaw =>
          optimizer.zero_grad()
          val logits = model.forward(b.sparseFeatures)
          val bs = yRaw.size(0).toInt
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble
          numBatches += 1
          val preds = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          val labels = y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
          epochPreds.appendAll(preds)
          epochLabels.appendAll(labels)
        }
      }
      val avgLoss = totalLoss / numBatches
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute()
      aucMetric.reset()
      println(f"[LR] Epoch ${e + 1}/$epochs - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalLR(model: LR, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      b.labels.foreach { yRaw =>
        val logits = model.forward(b.sparseFeatures)
        val p = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val y = yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        preds.appendAll(p)
        labels.appendAll(y)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray)
    ll.update(preds.toArray, labels.toArray)
    acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }
}