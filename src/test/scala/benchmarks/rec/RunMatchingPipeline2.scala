package benchmarks.rec

import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import org.bytedeco.pytorch.*
import torchrec.Implicits.{RichTensor, tensor}
import torchrec.basic.features.{SequenceFeature, SparseFeature}
import torchrec.basic.losses.BCEWithLogitsLoss
import torchrec.basic.metrics.{AUC, Accuracy, LogLoss}
import torchrec.data.{DataLoader, MatchingDataset, SequenceDataset, TensorDataset}
import torchrec.models.matching.*
import torchrec.trainers.MatchTrainer

import scala.collection.mutable
import scala.io.Source
import scala.util.Random

//[✓ PASS] SASRec
//  Metrics: AUC=0.8589, LogLoss=0.2577, Accuracy=0.8921
//[✓ PASS] DSSMSENET
//  Metrics: AUC=0.6024, LogLoss=0.8414, Accuracy=0.1120
//[✓ PASS] FaceBookDSSM
//  Metrics: AUC=0.5867, LogLoss=0.8262, Accuracy=0.1120
/**
 * Matching models benchmark pipeline using fraud dataset.
 * Tests: SASRec, DSSMSENET, FaceBookDSSM (runs with actual training output!)
 */
object RunMatchingPipeline2 {

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

  case class ModelReport(name: String, ok: Boolean, metrics: Map[String, Float], note: String)

  def main(args: Array[String]): Unit = {
    val p = parseArgs(args)
    val csvPath = p.getOrElse("dataset", "/home/muller/IdeaProjects/torch-rechub-scala/src/main/resources/fraud_data.csv")
    val maxRows = p.getOrElse("max_rows", "21900").toInt
    val batchSize = p.getOrElse("batch_size", "256").toInt
    val epochs = p.getOrElse("epochs", "15").toInt
    val device = p.getOrElse("device", "cuda")
    val seed = p.getOrElse("seed", "2026").toInt
    val bins = p.getOrElse("bins", "128").toInt
    val seqLen = p.getOrElse("seq_len", "20").toInt

    println(s"[Pipeline2] Loading fraud dataset from: $csvPath")
    val prepared = prepareFraud(csvPath, maxRows, bins, seqLen, seed, itemEmbedDim = 16)
    println(s"[Pipeline2] Data loaded: ${prepared.train.size} train, ${prepared.valid.size} valid, ${prepared.test.size} test")

    val reports = mutable.ArrayBuffer[ModelReport]()

    // Build shared datasets
    val matchingData = buildMatchingSequenceData(prepared, seqEmbedDim = 16)

    // ============================================================
    // SASRec
    // ============================================================
     println("\n[Pipeline2] ========== SASRec ==========")
     reports += runReport("SASRec") {
       clearMemory(device)
       val seqFeatures = List(SequenceFeature("seq_feat", prepared.numBins + 2, 8, pooling = "mean", sharedWith = None, maxLen = prepared.seqLen, paddingIdx = 0L))
       val model = new SASRec(seqFeatures, embedDim = 8, numHeads = 2, numLayers = 2, ffnDim = 128, dropout = 0.2f, device = device)
       trainSASRec(model, matchingData._1, epochs, batchSize, device)
       val metrics = evalSASRec(model, matchingData._3, batchSize, device)
       clearMemory(device)
       metrics
     }

    // ============================================================
    // DSSMSENET
    // ============================================================
    println("\n[Pipeline2] ========== DSSMSENET ==========")
    reports += runReport("DSSMSENET") {
      clearMemory(device)
      val userFeatures = List(SparseFeature("user_id", prepared.userVocab + 2, 8, sharedWith = None, paddingIdx = None))
      val itemFeatures = List(SparseFeature("item_id", prepared.itemVocab + 2, 8, sharedWith = None, paddingIdx = None))
      val model = new DSSMSENET(userFeatures, itemFeatures, userParams = Map(), itemParams = Map(), temperature = 1.0f, device = device)
      trainDSSMSENET(model, prepared, epochs, batchSize, device)
      val metrics = evalDSSMSENET(model, prepared, batchSize, device)
      clearMemory(device)
      metrics
    }

    // ============================================================
    // FaceBookDSSM
    // ============================================================
    println("\n[Pipeline2] ========== FaceBookDSSM ==========")
    reports += runReport("FaceBookDSSM") {
      clearMemory(device)
      val userFeatures = List(SparseFeature("user_id", prepared.userVocab + 2, 8, sharedWith = None, paddingIdx = None))
      val posItemFeatures = List(SparseFeature("item_id", prepared.itemVocab + 2, 8, sharedWith = None, paddingIdx = None))
      val negItemFeatures = List(SparseFeature("item_id_neg", prepared.itemVocab + 2, 8, sharedWith = None, paddingIdx = None))
      val model = new FaceBookDSSM(userFeatures, posItemFeatures, negItemFeatures, userParams = Map(), itemParams = Map(), temperature = 1.0f, device = device)
      trainFaceBookDSSM(model, prepared, epochs, batchSize, device)
      val metrics = evalFaceBookDSSM(model, prepared, batchSize, device)
      clearMemory(device)
      metrics
    }

    // ============================================================
    // Summary
    // ============================================================
    println("\n" + "="*60)
    println("MATCHING PIPELINE 2 SUMMARY")
    println("="*60)
    reports.foreach { r =>
      val m = r.metrics.map { case (k, v) => s"$k=${"%.4f".format(v)}" }.mkString(", ")
      println(s"[${if (r.ok) "✓ PASS" else "✗ FAIL"}] ${r.name}")
      if (r.metrics.nonEmpty) {
        println(s"  Metrics: $m")
      } else {
        println(s"  Note: ${r.note}")
      }
    }
  }

  private def parseArgs(args: Array[String]): Map[String, String] = {
    args.sliding(2, 2).collect {
      case Array(k, v) if k.startsWith("--") => k.stripPrefix("--") -> v
    }.toMap
  }

  private def clearMemory(device: String): Unit = {
    if (device == "cuda") {
      torch.emptyCache()
      System.gc()
    }
  }

  private def runReport(name: String)(fn: => Map[String, Float]): ModelReport = {
    try {
      val m = fn
      ModelReport(name, ok = true, m, "")
    } catch {
      case e: Throwable => ModelReport(name, ok = false, Map.empty, Option(e.getMessage).getOrElse(e.getClass.getSimpleName))
    }
  }

  // ================================================================
  // Data Preparation
  // ================================================================

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

  private def buildMatchingSequenceData(prepared: FraudPrepared, seqEmbedDim: Int): (SequenceDataset, SequenceDataset, SequenceDataset) = {
    def mk(rows: Vector[EncodedRow]): SequenceDataset = {
      val userIds = tensor(rows.map(_.userId.toFloat).toArray, Array(rows.size.toLong)).toType(ScalarType.Long)
      val flatTokens = rows.flatMap(_.tokens.map(_.toFloat)).toArray
      val tokens = tensor(flatTokens, Array(rows.size.toLong, prepared.seqLen.toLong)).toType(ScalarType.Long)
      val positions = tensor(Array.tabulate(rows.size * prepared.seqLen)(i => (i % prepared.seqLen).toFloat), Array(rows.size.toLong, prepared.seqLen.toLong)).toType(ScalarType.Long)
      val labels = tensor(rows.map(_.label).toArray, Array(rows.size.toLong))
      val flatItemVec = rows.flatMap(_.itemVec).toArray
      val itemVec = tensor(flatItemVec, Array(rows.size.toLong, seqEmbedDim.toLong)).toType(ScalarType.Float)
      new SequenceDataset(
        features = Map("user_id" -> userIds),
        sequenceFeatures = Map("seq_feat" -> tokens),
        labels = Some(labels),
        tokens = Some(tokens),
        positions = Some(positions),
        itemFeatures = Some(Map("item_vec" -> itemVec))
      )
    }
    (mk(prepared.train), mk(prepared.valid), mk(prepared.test))
  }

  private def buildMatchingDatasetWithNeg(prepared: FraudPrepared): (MatchingDataset, MatchingDataset, MatchingDataset) = {
    def mk(rows: Vector[EncodedRow]): MatchingDataset = {
      val userIds = tensor(rows.map(_.userId.toFloat).toArray, Array(rows.size.toLong)).toType(ScalarType.Long)
      val itemIds = tensor(rows.map(_.itemId.toFloat).toArray, Array(rows.size.toLong)).toType(ScalarType.Long)
      // Generate negative item IDs by adding a fixed offset and wrapping
      val negItemIds = tensor(rows.map(r => ((r.itemId + 1000 + Random.nextInt(5000)) % prepared.itemVocab + 1).toFloat).toArray, Array(rows.size.toLong)).toType(ScalarType.Long)
      val labels = tensor(rows.map(_.label).toArray, Array(rows.size.toLong))
      new MatchingDataset(
        userFeatures = Map("user_id" -> userIds),
        itemFeatures = Map("item_id" -> itemIds),
        labels = Some(labels),
        negItemFeatures = Some(Map("item_id_neg" -> negItemIds))
      )
    }
    (mk(prepared.train), mk(prepared.valid), mk(prepared.test))
  }

  // ================================================================
  // SASRec
  // ================================================================

  private def trainSASRec(model: SASRec, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
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
        (b.sequenceFeatures.get("seq_feat"), b.labels) match {
          case (Some(tokens), Some(yRaw)) =>
            val bs = yRaw.size(0).toInt
            optimizer.zero_grad()
            val logits = model.forward(tokens)
            val logitsView = logits.view(bs, 1)
            val y = yRaw.view(bs, 1).toType(ScalarType.Float)
            val loss = lossFn.apply(logitsView, y)
            loss.backward(); optimizer.step()
            totalLoss += loss.item().toDouble
            numBatches += 1
            val preds = logitsView.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
            val labels = y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
            epochPreds.appendAll(preds)
            epochLabels.appendAll(labels)
          case _ =>
        }
      }
      val avgLoss = totalLoss / numBatches
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute()
      aucMetric.reset()
      println(f"[SASRec] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }

  private def evalSASRec(model: SASRec, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      (b.sequenceFeatures.get("seq_feat"), b.labels) match {
        case (Some(tokens), Some(yRaw)) =>
          val bs = yRaw.size(0).toInt
          val logits = model.forward(tokens).view(bs, 1)
          preds.appendAll(logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
          labels.appendAll(yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        case _ =>
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray)
    ll.update(preds.toArray, labels.toArray)
    acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ================================================================
  // DSSMSENET
  // ================================================================

  private def trainDSSMSENET(model: DSSMSENET, prepared: FraudPrepared, epochs: Int, batchSize: Int, device: String): Unit = {
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val lossFn = new BCEWithLogitsLoss()
    val aucMetric = new AUC()

    for (e <- 0 until epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()

      val trainSize = prepared.train.size
      var batchIdx = 0
      while (batchIdx * batchSize < trainSize) {
        val startIdx = batchIdx * batchSize
        val endIdx = math.min(startIdx + batchSize, trainSize)
        val currentBatchSize = endIdx - startIdx

        try {
          optimizer.zero_grad()

          val batchUserIds = tensor(prepared.train.slice(startIdx, endIdx).map(_.userId.toFloat).toArray, Array(currentBatchSize.toLong)).toType(ScalarType.Long)
          val batchItemIds = tensor(prepared.train.slice(startIdx, endIdx).map(_.itemId.toFloat).toArray, Array(currentBatchSize.toLong)).toType(ScalarType.Long)
          val batchLabels = tensor(prepared.train.slice(startIdx, endIdx).map(_.label).toArray, Array(currentBatchSize.toLong))

          val logits = model.forward(Map("user_id" -> batchUserIds, "item_id" -> batchItemIds)).unsqueeze(1)

          val loss = lossFn.apply(logits, batchLabels.unsqueeze(1))

          loss.backward()
          optimizer.step()

          totalLoss += loss.item().toDouble
          numBatches += 1

          val preds = logits.sigmoid().squeeze().toType(ScalarType.Float).detach().contiguous().cpu().toFloatArray
          epochPreds.appendAll(preds)
          epochLabels.appendAll(prepared.train.slice(startIdx, endIdx).map(_.label))
        } catch {
          case ex: Exception => throw ex
        }

        batchIdx += 1
      }

      if (numBatches > 0) {
        val avgLoss = totalLoss / numBatches
        aucMetric.update(epochPreds.toArray, epochLabels.toArray)
        val trainAuc = aucMetric.compute()
        aucMetric.reset()
        println(f"[DSSMSENET] Epoch ${e + 1}/$epochs - Loss: $avgLoss%.4f - AUC: $trainAuc%.4f")
        if (device == "cuda") torch.emptyCache()
      }
    }
  }

  private def evalDSSMSENET(model: DSSMSENET, prepared: FraudPrepared, batchSize: Int, device: String): Map[String, Float] = {
    model.eval()
    val auc = new AUC()
    val ll = new LogLoss()
    val acc = new Accuracy()
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()

    val testSize = prepared.test.size
    var batchIdx = 0
    while (batchIdx * batchSize < testSize) {
      val startIdx = batchIdx * batchSize
      val endIdx = math.min(startIdx + batchSize, testSize)
      val currentBatchSize = endIdx - startIdx

      try {
        val batchUserIds = tensor(prepared.test.slice(startIdx, endIdx).map(_.userId.toFloat).toArray, Array(currentBatchSize.toLong)).toType(ScalarType.Long)
        val batchItemIds = tensor(prepared.test.slice(startIdx, endIdx).map(_.itemId.toFloat).toArray, Array(currentBatchSize.toLong)).toType(ScalarType.Long)

        val logits = model.forward(Map("user_id" -> batchUserIds, "item_id" -> batchItemIds)).unsqueeze(1)

        val predProbs = logits.sigmoid().squeeze().toType(ScalarType.Float).detach().contiguous().cpu().toFloatArray
        preds.appendAll(predProbs)
        labels.appendAll(prepared.test.slice(startIdx, endIdx).map(_.label))
      } catch {
          case ex: Exception => throw ex
        }

      batchIdx += 1
    }

    if (preds.nonEmpty) {
      auc.update(preds.toArray, labels.toArray)
      ll.update(preds.toArray, labels.toArray)
      acc.update(preds.toArray, labels.toArray)
      val aucScore = auc.compute()
      val llScore = ll.compute()
      val accScore = acc.compute()
      println(f"[DSSMSENET] Test - AUC: $aucScore%.4f, LogLoss: $llScore%.4f, Acc: $accScore%.4f")
      Map("AUC" -> aucScore, "LogLoss" -> llScore, "Accuracy" -> accScore)
    } else {
      Map.empty
    }
  }

  // ================================================================
  // FaceBookDSSM
  // ================================================================

  private def trainFaceBookDSSM(model: FaceBookDSSM, prepared: FraudPrepared, epochs: Int, batchSize: Int, device: String): Unit = {
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val lossFn = new BCEWithLogitsLoss()
    val aucMetric = new AUC()

    for (e <- 0 until epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()

      val trainSize = prepared.train.size
      var batchIdx = 0
      while (batchIdx * batchSize < trainSize) {
        val startIdx = batchIdx * batchSize
        val endIdx = math.min(startIdx + batchSize, trainSize)
        val currentBatchSize = endIdx - startIdx

         try {
           optimizer.zero_grad()

           val batchUserIds = tensor(prepared.train.slice(startIdx, endIdx).map(_.userId.toFloat).toArray, Array(currentBatchSize.toLong)).toType(ScalarType.Long)
           val batchItemIds = tensor(prepared.train.slice(startIdx, endIdx).map(_.itemId.toFloat).toArray, Array(currentBatchSize.toLong)).toType(ScalarType.Long)
           val batchNegItemIds = tensor(prepared.train.slice(startIdx, endIdx).map(r => ((r.itemId + 1000 + Random.nextInt(5000)) % 20000 + 1).toFloat).toArray, Array(currentBatchSize.toLong)).toType(ScalarType.Long)
           val batchLabels = tensor(prepared.train.slice(startIdx, endIdx).map(_.label).toArray, Array(currentBatchSize.toLong))

           val (posScore, negScore) = model.forward(Map("user_id" -> batchUserIds, "item_id" -> batchItemIds, "item_id_neg" -> batchNegItemIds))
           val logits = posScore.unsqueeze(1)

           val loss = lossFn.apply(logits, batchLabels.unsqueeze(1))

           loss.backward()
           optimizer.step()

           totalLoss += loss.item().toDouble
           numBatches += 1

           val preds = logits.sigmoid().squeeze().toType(ScalarType.Float).detach().contiguous().cpu().toFloatArray
           epochPreds.appendAll(preds)
           epochLabels.appendAll(prepared.train.slice(startIdx, endIdx).map(_.label))
        } catch {
          case ex: Exception => throw ex
        }

        batchIdx += 1
      }

      if (numBatches > 0) {
        val avgLoss = totalLoss / numBatches
        aucMetric.update(epochPreds.toArray, epochLabels.toArray)
        val trainAuc = aucMetric.compute()
        aucMetric.reset()
        println(f"[FaceBookDSSM] Epoch ${e + 1}/$epochs - Loss: $avgLoss%.4f - AUC: $trainAuc%.4f")
        if (device == "cuda") torch.emptyCache()
      }
    }
  }

  private def evalFaceBookDSSM(model: FaceBookDSSM, prepared: FraudPrepared, batchSize: Int, device: String): Map[String, Float] = {
    model.eval()
    val auc = new AUC()
    val ll = new LogLoss()
    val acc = new Accuracy()
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()

    val testSize = prepared.test.size
    var batchIdx = 0
    while (batchIdx * batchSize < testSize) {
      val startIdx = batchIdx * batchSize
      val endIdx = math.min(startIdx + batchSize, testSize)
      val currentBatchSize = endIdx - startIdx

       try {
         val batchUserIds = tensor(prepared.test.slice(startIdx, endIdx).map(_.userId.toFloat).toArray, Array(currentBatchSize.toLong)).toType(ScalarType.Long)
         val batchItemIds = tensor(prepared.test.slice(startIdx, endIdx).map(_.itemId.toFloat).toArray, Array(currentBatchSize.toLong)).toType(ScalarType.Long)
         val batchNegItemIds = tensor(prepared.test.slice(startIdx, endIdx).map(r => ((r.itemId + 1000 + Random.nextInt(5000)) % 20000 + 1).toFloat).toArray, Array(currentBatchSize.toLong)).toType(ScalarType.Long)

         val (posScore, negScore) = model.forward(Map("user_id" -> batchUserIds, "item_id" -> batchItemIds, "item_id_neg" -> batchNegItemIds))
         val logits = posScore.unsqueeze(1)

         val predProbs = logits.sigmoid().squeeze().toType(ScalarType.Float).detach().contiguous().cpu().toFloatArray
         preds.appendAll(predProbs)
         labels.appendAll(prepared.test.slice(startIdx, endIdx).map(_.label))
      } catch {
          case ex: Exception => throw ex
        }

      batchIdx += 1
    }

    if (preds.nonEmpty) {
      auc.update(preds.toArray, labels.toArray)
      ll.update(preds.toArray, labels.toArray)
      acc.update(preds.toArray, labels.toArray)
      val aucScore = auc.compute()
      val llScore = ll.compute()
      val accScore = acc.compute()
      println(f"[FaceBookDSSM] Test - AUC: $aucScore%.4f, LogLoss: $llScore%.4f, Acc: $accScore%.4f")
      Map("AUC" -> aucScore, "LogLoss" -> llScore, "Accuracy" -> accScore)
    } else {
      Map.empty
    }
  }

  // ================================================================
  // YoutubeSBC (stub functions to prevent compilation errors)
  // ================================================================

  private def trainYoutubeSBC(model: YoutubeSBC, prepared: FraudPrepared, epochs: Int, batchSize: Int, device: String): Unit = {
    println("[YoutubeSBC] Skipped - focus on core models")
  }

  private def evalYoutubeSBC(model: YoutubeSBC, prepared: FraudPrepared, batchSize: Int, device: String): Map[String, Float] = {
    Map.empty
  }

  // ================================================================
  // MAMBA (stub functions)
  // ================================================================

  private def trainMambaAsCtr(model: MAMBA, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    println("[MAMBA] Skipped - focus on core models")
  }

  private def evalMambaAsCtr(model: MAMBA, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    Map.empty
  }

  // ================================================================
  // MIND (stub functions)
  // ================================================================

  private def trainMindAsCtr(model: MIND, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    println("[MIND] Skipped - focus on core models")
  }

  private def evalMindAsCtr(model: MIND, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    Map.empty
  }
}
