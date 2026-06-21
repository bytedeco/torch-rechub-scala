package benchmarks.rec

import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import org.bytedeco.pytorch.*
import torchrec.Implicits.*
import torchrec.basic.features.{SequenceFeature, SparseFeature}
import torchrec.basic.losses.BCEWithLogitsLoss
import torchrec.basic.metrics.{AUC, HitRate}
import torchrec.data.{DataLoader, SequenceDataset, TensorDataset}
import torchrec.models.matching.*
import torchrec.trainers.MatchTrainer

import scala.collection.mutable
import scala.io.Source
import scala.util.Random

object RunMatchingPipeline1 {

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
     val csvPath = "/home/muller/IdeaProjects/torch-rechub-scala/src/main/resources/fraud_data.csv"
     val maxRows = 5000
     val batchSize = 256
     val epochs = 3
     val device = "cuda"
     val seed = 2026
     val bins = 128
     val seqLen = 20

     val prepared = prepareFraud(csvPath, maxRows, bins, seqLen, seed, itemEmbedDim = 8)
     val reports = mutable.ArrayBuffer[ModelReport]()

    // Build shared datasets
    val (trainDs, validDs, testDs) = buildMatchingDatasets(prepared, device)


    // ========== YoutubeDNN ==========
    reports += runReport("YoutubeDNN") {
      if (device == "cuda") { torch.emptyCache(); System.gc() }
      val features = List(SparseFeature("feat_0", prepared.numBins + 2, 8))
      val sequenceFeatures = List(SequenceFeature("seq_feat", prepared.itemVocab + 2, 8, maxLen = prepared.seqLen))
      val model = new YoutubeDNN(features, sequenceFeatures, embedDim = 8, towerDims = List(256L, 128L), dropout = 0.2f, device)
      trainYoutubeDnn(model, trainDs, epochs, batchSize, device)
      evalYoutubeDnn(model, testDs, batchSize, device)
    }

    // ========== ComirecDR ==========
    reports += runReport("ComirecDR") {
      if (device == "cuda") { torch.emptyCache(); System.gc() }
      val features = List(SparseFeature("feat_0", prepared.numBins + 2, 8))
      val sequenceFeature = SequenceFeature("seq_feat", prepared.itemVocab + 2, 8, maxLen = prepared.seqLen)
      val model = new ComirecDR(features, sequenceFeature, embedDim = 8, numInterests = 4, mlpDims = List(256L, 128L), dropout = 0.2f, device)
      trainComirecDR(model, trainDs, epochs, batchSize, device)
      evalComirecDR(model, testDs, batchSize, device)
    }

    // ========== ComirecSA ==========
    reports += runReport("ComirecSA") {
      if (device == "cuda") { torch.emptyCache(); System.gc() }
      val features = List(SparseFeature("feat_0", prepared.numBins + 2, 8))
      val sequenceFeature = SequenceFeature("seq_feat", prepared.itemVocab + 2, 8, maxLen = prepared.seqLen)
      val model = new ComirecSA(features, sequenceFeature, embedDim = 8, numInterests = 4, numHeads = 2, mlpDims = List(256L, 128L), dropout = 0.2f, device)
      trainComirecSA(model, trainDs, epochs, batchSize, device)
      evalComirecSA(model, testDs, batchSize, device)
    }





    // ========== NARM ==========
    reports += runReport("NARM") {
      if (device == "cuda") { torch.emptyCache(); System.gc() }
      val features = List(SequenceFeature("seq_feat", prepared.itemVocab + 2, 8, maxLen = prepared.seqLen))
      val model = new NARM(features, embedDim = 8, hiddenDim = 8, attentionDim = 8, device)
      trainNarm(model, trainDs, epochs, batchSize, device)
      evalNarm(model, testDs, batchSize, device)
    }

    // ========== STAMP ==========
    reports += runReport("STAMP") {
      if (device == "cuda") { torch.emptyCache(); System.gc() }
      val features = List(SequenceFeature("seq_feat", prepared.itemVocab + 2, 8, maxLen = prepared.seqLen))
      val model = new STAMP(features, embedDim = 8, attentionDim = 8, device)
      trainStamp(model, trainDs, epochs, batchSize, device)
      evalStamp(model, testDs, batchSize, device)
    }

     // ========== NCF ==========
     reports += runReport("NCF") {
       if (device == "cuda") {
         torch.emptyCache(); System.gc()
       }
       val features = List(
         SparseFeature("user_id", prepared.userVocab + 2, 8),
         SparseFeature("item_id", prepared.itemVocab + 2, 8)
       )
       val model = new NCF(features, userFieldIdx = 0, itemFieldIdx = 1, embedDim = 8, mlpDims = List(64L, 32L), dropout = 0.2f, device)
       trainNcf(model, trainDs, epochs, batchSize, device)
       evalNcf(model, testDs, batchSize, device)
     }
     // ========== GRU4Rec ==========
     reports += runReport("GRU4Rec") {
       if (device == "cuda") {
         torch.emptyCache();
         System.gc()
       }
       val userFeatures = List(SparseFeature("feat_0", prepared.numBins + 2, 8))
       val historyFeatures = List(SequenceFeature("seq_feat", prepared.itemVocab + 2, 8, maxLen = prepared.seqLen))
       val itemFeatures = List(SparseFeature("item_id", prepared.itemVocab + 2, 8))
       val model = new GRU4Rec(userFeatures, historyFeatures, itemFeatures, negItemFeature = None, userParams = Map(), temperature = 1.0f, device)
       trainGru4Rec(model, trainDs, epochs, batchSize, device)
       evalGru4Rec(model, testDs, batchSize, device)
     }

     // ========== SINE ==========
    reports += runReport("SINE") {
      if (device == "cuda") { torch.emptyCache(); System.gc() }
      val features = List(SparseFeature("feat_0", prepared.numBins + 2, 8))
      val sequenceFeature = SequenceFeature("seq_feat", prepared.itemVocab + 2, 8, maxLen = prepared.seqLen)
      val model = new SINE(features, sequenceFeature, embedDim = 8, numInterests = 4, mlpDims = List(128L, 64L), dropout = 0.2f, device)
      trainSine(model, trainDs, epochs, batchSize, device)
      evalSine(model, testDs, batchSize, device)
    }
     // ========== DSSM ==========
     reports += runReport("DSSM") {
       if (device == "cuda") {
         torch.emptyCache(); System.gc()
       }
       val userFeatures = List(SparseFeature("user_id", prepared.userVocab + 2, 8))
       val itemFeatures = List(SparseFeature("item_id", prepared.itemVocab + 2, 8))
       val model = new DSSM(userFeatures, itemFeatures, embedDim = 8, towerDims = List(256L, 128L), dropout = 0.2f, device)
       trainDssm(model, trainDs, epochs, batchSize, device)
       evalDssm(model, testDs, batchSize, device)
     }
    if (device == "cuda") { torch.emptyCache(); System.gc() }

    println("\n================ MATCHING PIPELINE 1 SUMMARY ================")
    reports.foreach { r =>
      val m = r.metrics.map { case (k, v) => s"$k=${"%.4f".format(v)}" }.mkString(", ")
      println(s"[${if (r.ok) "PASS" else "FAIL"}] ${r.name} -> ${if (m.nonEmpty) m else r.note}")
      if (!r.ok) println(s"  reason: ${r.note}")
    }
   }

   private def runReport(name: String)(fn: => Map[String, Float]): ModelReport = {
     try {
       val m = fn
       ModelReport(name, ok = true, m, "")
     } catch {
       case e: Throwable =>
         e.printStackTrace()
         ModelReport(name, ok = false, Map.empty, Option(e.getMessage).getOrElse(e.getClass.getSimpleName))
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

  private def buildMatchingDatasets(prepared: FraudPrepared, device: String): (SequenceDataset, SequenceDataset, SequenceDataset) = {
    def mk(rows: Vector[EncodedRow]): SequenceDataset = {
      val userIds = tensor(rows.map(_.userId.toFloat).toArray, Array(rows.size.toLong)).toType(ScalarType.Long)
      val itemIds = tensor(rows.map(_.itemId.toFloat).toArray, Array(rows.size.toLong)).toType(ScalarType.Long)
      // feat_0 is a binned categorical id in [1, numBins]; it MUST stay within the
      // declared vocab (numBins + 2) to avoid out-of-range embedding lookups.
      val feat0 = tensor(rows.map(_.ids(0).toFloat).toArray, Array(rows.size.toLong)).toType(ScalarType.Long)
      val flatTokens = rows.flatMap(_.tokens.map(_.toFloat)).toArray
      val tokens = tensor(flatTokens, Array(rows.size.toLong, prepared.seqLen.toLong)).toType(ScalarType.Long)
      val positions = tensor(Array.tabulate(rows.size * prepared.seqLen)(i => (i % prepared.seqLen).toFloat), Array(rows.size.toLong, prepared.seqLen.toLong)).toType(ScalarType.Long)
      val labels = tensor(rows.map(_.label).toArray, Array(rows.size.toLong))
      val flatItemVec = rows.flatMap(_.itemVec).toArray
      val itemVec = tensor(flatItemVec, Array(rows.size.toLong, 8.toLong)).toType(ScalarType.Float)
      new SequenceDataset(
        features = Map("user_id" -> userIds, "item_id" -> itemIds, "feat_0" -> feat0),
        sequenceFeatures = Map("seq_feat" -> tokens),
        labels = Some(labels),
        tokens = Some(tokens),
        positions = Some(positions),
        itemFeatures = Some(Map("item_vec" -> itemVec))
      )
    }
    (mk(prepared.train), mk(prepared.valid), mk(prepared.test))
  }

  // ==================== DSSM ====================
  private def trainDssm(model: DSSM, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
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
          // DSSM is a two-tower model: route user_id to the user tower and
          // item_id to the item tower (selected explicitly, not all sparse feats).
          val userFeats = b.sparseFeatures.view.filterKeys(_ == "user_id").toMap
          val itemFeats = b.sparseFeatures.view.filterKeys(_ == "item_id").toMap
          if (userFeats.nonEmpty && itemFeats.nonEmpty) {
            val bs = userFeats.values.head.size(0).toInt
            optimizer.zero_grad()
            val userEmb = model.userTowerForward(userFeats)
           val itemEmb = model.itemTowerForward(itemFeats)
           val posScores = userEmb.mul(itemEmb).sum(1)
           var lossTensor: Tensor = null.asInstanceOf[Tensor]
           var pairCount = 0
           for (i <- 0 until bs) {
             val userVec = userEmb.select(0, i)
             val posScoreI = posScores.select(0, i)
             for (j <- 0 until bs if j != i) {
               val negItemVec = itemEmb.select(0, j)
               val negScoreIJ = userVec.mul(negItemVec).sum(0).reshape(1)
               val pairLoss = torch.log(torch.sigmoid(posScoreI.sub(negScoreIJ)).add(new Scalar(1e-8f))).neg()
               if (lossTensor == null) lossTensor = pairLoss else lossTensor = lossTensor.add(pairLoss)
               pairCount += 1
             }
           }
           if (lossTensor != null && pairCount > 0) {
             val avgLoss = lossTensor.div(new Scalar(pairCount.toDouble))
             avgLoss.backward()
             optimizer.step()
             totalLoss += avgLoss.item().toDouble
             numBatches += 1
             lossTensor.close()
             // collect predictions
             val preds = posScores.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
             epochPreds.appendAll(preds)
             val labelsArr = b.labels.getOrElse(torch.zeros(bs.toLong)).squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
             epochLabels.appendAll(labelsArr)
           }
         }
       }
      val avgLoss = if (numBatches > 0) totalLoss / numBatches else 0.0
      if (epochPreds.nonEmpty) {
        aucMetric.update(epochPreds.toArray, epochLabels.toArray)
        val trainAuc = aucMetric.compute()
        aucMetric.reset()
        println(f"[DSSM] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      } else {
        println(f"[DSSM] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f")
      }
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }

  private def evalDssm(model: DSSM, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      val userFeats = b.sparseFeatures.view.filterKeys(_ == "user_id").toMap
      val itemFeats = b.sparseFeatures.view.filterKeys(_ == "item_id").toMap
      if (userFeats.nonEmpty && itemFeats.nonEmpty) {
        val bs = userFeats.values.head.size(0).toInt
        val userEmb = model.userTowerForward(userFeats)
        val itemEmb = model.itemTowerForward(itemFeats)
        val scores = userEmb.mul(itemEmb).sum(1).sigmoid()
        preds.appendAll(scores.toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        b.labels.foreach { y => labels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) }
      }
    }
    if (preds.isEmpty) return Map("AUC" -> 0.0f)
    val auc = new AUC()
    auc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute())
  }

  // ==================== YoutubeDNN ====================
  private def trainYoutubeDnn(model: YoutubeDNN, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
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
         val feats = b.sparseFeatures
         val seqFeats = b.tokens
         if (feats.nonEmpty && seqFeats.nonEmpty) {
           val bs = feats.values.head.size(0).toInt
           optimizer.zero_grad()
           val emb = model.forward(feats, Map("seq_feat" -> seqFeats.get))
           val itemFeats = b.itemFeatures
           val itemEmb = if (itemFeats.nonEmpty) itemFeats.values.head else torch.zeros_like(emb)
           val scores = emb.mul(itemEmb).sum(1)
           val y = b.labels.getOrElse(torch.zeros(bs.toLong)).view(bs, 1).toType(ScalarType.Float)
           val loss = lossFn.apply(scores.view(bs, 1), y)
           loss.backward()
           optimizer.step()
           totalLoss += loss.item().toDouble
           numBatches += 1
           epochPreds.appendAll(scores.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
           epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
         }
       }
      val avgLoss = if (numBatches > 0) totalLoss / numBatches else 0.0
      if (epochPreds.nonEmpty) {
        aucMetric.update(epochPreds.toArray, epochLabels.toArray)
        val trainAuc = aucMetric.compute()
        aucMetric.reset()
        println(f"[YoutubeDNN] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      } else {
        println(f"[YoutubeDNN] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f")
      }
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }

  private def evalYoutubeDnn(model: YoutubeDNN, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      val feats = b.sparseFeatures
      val seqFeats = b.tokens
      if (feats.nonEmpty && seqFeats.nonEmpty) {
        val emb = model.forward(feats, Map("seq_feat" -> seqFeats.get))
        val itemFeats = b.itemFeatures
        val itemEmb = if (itemFeats.nonEmpty) itemFeats.values.head else torch.zeros_like(emb)
        val scores = emb.mul(itemEmb).sum(1).sigmoid()
        preds.appendAll(scores.toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        b.labels.foreach { y => labels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) }
      }
    }
    if (preds.isEmpty) return Map("AUC" -> 0.0f)
    val auc = new AUC()
    auc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute())
  }

  // ==================== ComirecDR ====================
  private def trainComirecDR(model: ComirecDR, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
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
         val feats = b.sparseFeatures
         val seqTokens = b.tokens
         if (feats.nonEmpty && seqTokens.nonEmpty) {
           val bs = feats.values.head.size(0).toInt
           optimizer.zero_grad()
           val emb = model.forward(feats, seqTokens.get)
           val itemFeats = b.itemFeatures
           val itemEmb = if (itemFeats.nonEmpty) itemFeats.values.head else torch.zeros_like(emb)
           val scores = emb.mul(itemEmb).sum(1)
           val y = b.labels.getOrElse(torch.zeros(bs.toLong)).view(bs, 1).toType(ScalarType.Float)
           val loss = lossFn.apply(scores.view(bs, 1), y)
           loss.backward()
           optimizer.step()
           totalLoss += loss.item().toDouble
           numBatches += 1
           epochPreds.appendAll(scores.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
           epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
         }
       }
       val avgLoss = if (numBatches > 0) totalLoss / numBatches else 0.0
       if (epochPreds.nonEmpty) {
         aucMetric.update(epochPreds.toArray, epochLabels.toArray)
         val trainAuc = aucMetric.compute()
         aucMetric.reset()
         println(f"[ComirecDR] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      } else {
        println(f"[ComirecDR] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f")
      }
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }

  private def evalComirecDR(model: ComirecDR, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      val feats = b.sparseFeatures
      val seqTokens = b.tokens
      if (feats.nonEmpty && seqTokens.nonEmpty) {
        val emb = model.forward(feats, seqTokens.get)
        val itemFeats = b.itemFeatures
        val itemEmb = if (itemFeats.nonEmpty) itemFeats.values.head else torch.zeros_like(emb)
        val scores = emb.mul(itemEmb).sum(1).sigmoid()
        preds.appendAll(scores.toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        b.labels.foreach { y => labels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) }
      }
    }
    if (preds.isEmpty) return Map("AUC" -> 0.0f)
    val auc = new AUC()
    auc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute())
  }

  // ==================== ComirecSA ====================
   private def trainComirecSA(model: ComirecSA, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
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
         val feats = b.sparseFeatures
         val seqTokens = b.tokens
         if (feats.nonEmpty && seqTokens.nonEmpty) {
           val bs = feats.values.head.size(0).toInt
           optimizer.zero_grad()
           val emb = model.forward(feats, seqTokens.get)
           val itemFeats = b.itemFeatures
           val itemEmb = if (itemFeats.nonEmpty) itemFeats.values.head else torch.zeros_like(emb)
           val scores = emb.mul(itemEmb).sum(1)
           val y = b.labels.getOrElse(torch.zeros(bs.toLong)).view(bs, 1).toType(ScalarType.Float)
           val loss = lossFn.apply(scores.view(bs, 1), y)
           loss.backward()
           optimizer.step()
           totalLoss += loss.item().toDouble
           numBatches += 1
           epochPreds.appendAll(scores.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
           epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
         }
       }
       val avgLoss = if (numBatches > 0) totalLoss / numBatches else 0.0
       if (epochPreds.nonEmpty) {
         aucMetric.update(epochPreds.toArray, epochLabels.toArray)
         val trainAuc = aucMetric.compute()
         aucMetric.reset()
         println(f"[ComirecSA] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
       } else {
         println(f"[ComirecSA] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f")
       }
       if (device == "cuda") torch.emptyCache()
       e += 1
     }
   }

  private def evalComirecSA(model: ComirecSA, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      val feats = b.sparseFeatures
      val seqTokens = b.tokens
      if (feats.nonEmpty && seqTokens.nonEmpty) {
        val emb = model.forward(feats, seqTokens.get)
        val itemFeats = b.itemFeatures
        val itemEmb = if (itemFeats.nonEmpty) itemFeats.values.head else torch.zeros_like(emb)
        val scores = emb.mul(itemEmb).sum(1).sigmoid()
        preds.appendAll(scores.toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        b.labels.foreach { y => labels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) }
      }
    }
    if (preds.isEmpty) return Map("AUC" -> 0.0f)
    val auc = new AUC()
    auc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute())
  }

  // ==================== GRU4Rec ====================
   private def trainGru4Rec(model: GRU4Rec, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
     val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
     val lossFn = new BCEWithLogitsLoss()
     val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
     val aucMetric = new AUC()
     model.setMode("user")
     var e = 0
     while (e < epochs) {
       var totalLoss = 0.0
       var numBatches = 0
       val epochPreds = mutable.ArrayBuffer[Float]()
       val epochLabels = mutable.ArrayBuffer[Float]()
       val it = loader.iterator
       while (it.hasNext) {
         val b = it.next()
         val userFeats = b.sparseFeatures
         val seqFeats = b.tokens
         if (userFeats.nonEmpty && seqFeats.nonEmpty) {
           val bs = userFeats.values.head.size(0).toInt
           optimizer.zero_grad()
            val historyFeats = Map("seq_feat" -> seqFeats.get)
            val allFeats = userFeats ++ historyFeats
            val userEmb = model.userTower(allFeats)
            model.setMode("item")
            val itemFeats = b.sparseFeatures.view.filterKeys(_ == "item_id").toMap
            val itemEmb = if (itemFeats.nonEmpty) model.itemTower(itemFeats) else torch.zeros_like(userEmb)
            val scores = userEmb.mul(itemEmb).sum(1)
           val y = b.labels.getOrElse(torch.zeros(bs.toLong)).view(bs, 1).toType(ScalarType.Float)
           val loss = lossFn.apply(scores.view(bs, 1), y)
           loss.backward()
           optimizer.step()
           totalLoss += loss.item().toDouble
           numBatches += 1
           epochPreds.appendAll(scores.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
           epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
           model.setMode("user")
         }
       }
       val avgLoss = if (numBatches > 0) totalLoss / numBatches else 0.0
       if (epochPreds.nonEmpty) {
         aucMetric.update(epochPreds.toArray, epochLabels.toArray)
         val trainAuc = aucMetric.compute()
         aucMetric.reset()
         println(f"[GRU4Rec] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
       } else {
         println(f"[GRU4Rec] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f")
       }
       if (device == "cuda") torch.emptyCache()
       e += 1
     }
   }

  private def evalGru4Rec(model: GRU4Rec, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    model.setMode("user")
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      val userFeats = b.sparseFeatures
      val seqFeats = b.tokens
      if (userFeats.nonEmpty && seqFeats.nonEmpty) {
        val historyFeats = Map("seq_feat" -> seqFeats.get)
        val allFeats = userFeats ++ historyFeats
        val userEmb = model.userTower(allFeats)
        model.setMode("item")
        val itemFeats = b.sparseFeatures.view.filterKeys(_ == "item_id").toMap
        val itemEmb = if (itemFeats.nonEmpty) model.itemTower(itemFeats) else torch.zeros_like(userEmb)
        val scores = userEmb.mul(itemEmb).sum(1).sigmoid()
        preds.appendAll(scores.toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        b.labels.foreach { y => labels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) }
        model.setMode("user")
      }
    }
    if (preds.isEmpty) return Map("AUC" -> 0.0f)
    val auc = new AUC()
    auc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute())
  }

  // ==================== NCF ====================
   private def trainNcf(model: NCF, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
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
         val feats = b.sparseFeatures.view.filterKeys(k => k == "user_id" || k == "item_id").toMap
         if (feats.nonEmpty) {
           val bs = feats.values.head.size(0).toInt
           optimizer.zero_grad()
           val logits = model.forward(feats)
           val y = b.labels.getOrElse(torch.zeros(bs.toLong)).view(bs, 1).toType(ScalarType.Float)
           val loss = lossFn.apply(logits.view(bs, 1), y)
           loss.backward()
           optimizer.step()
           totalLoss += loss.item().toDouble
           numBatches += 1
           epochPreds.appendAll(logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
           epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
         }
       }
       val avgLoss = if (numBatches > 0) totalLoss / numBatches else 0.0
       if (epochPreds.nonEmpty) {
         aucMetric.update(epochPreds.toArray, epochLabels.toArray)
         val trainAuc = aucMetric.compute()
         aucMetric.reset()
         println(f"[NCF] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
       } else {
         println(f"[NCF] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f")
       }
       if (device == "cuda") torch.emptyCache()
       e += 1
     }
   }

  private def evalNcf(model: NCF, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      val feats = b.sparseFeatures.view.filterKeys(k => k == "user_id" || k == "item_id").toMap
      if (feats.nonEmpty) {
        val logits = model.forward(feats)
        preds.appendAll(logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        b.labels.foreach { y => labels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) }
      }
    }
    if (preds.isEmpty) return Map("AUC" -> 0.0f)
    val auc = new AUC()
    auc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute())
  }

  // ==================== NARM ====================
   private def trainNarm(model: NARM, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
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
         val seqTokens = b.tokens
         if (seqTokens.nonEmpty) {
           val bs = seqTokens.get.size(0).toInt
           optimizer.zero_grad()
           val emb = model.forward(seqTokens.get)
           val itemFeats = b.itemFeatures
           val itemEmb = if (itemFeats.nonEmpty) itemFeats.values.head else torch.zeros_like(emb)
           val scores = emb.mul(itemEmb).sum(1)
           val y = b.labels.getOrElse(torch.zeros(bs.toLong)).view(bs, 1).toType(ScalarType.Float)
           val loss = lossFn.apply(scores.view(bs, 1), y)
           loss.backward()
           optimizer.step()
           totalLoss += loss.item().toDouble
           numBatches += 1
           epochPreds.appendAll(scores.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
           epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
         }
       }
       val avgLoss = if (numBatches > 0) totalLoss / numBatches else 0.0
       if (epochPreds.nonEmpty) {
         aucMetric.update(epochPreds.toArray, epochLabels.toArray)
         val trainAuc = aucMetric.compute()
         aucMetric.reset()
         println(f"[NARM] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
       } else {
         println(f"[NARM] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f")
       }
       if (device == "cuda") torch.emptyCache()
       e += 1
     }
   }

  private def evalNarm(model: NARM, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      val seqTokens = b.tokens
      if (seqTokens.nonEmpty) {
        val emb = model.forward(seqTokens.get)
        val itemFeats = b.itemFeatures
        val itemEmb = if (itemFeats.nonEmpty) itemFeats.values.head else torch.zeros_like(emb)
        val scores = emb.mul(itemEmb).sum(1).sigmoid()
        preds.appendAll(scores.toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        b.labels.foreach { y => labels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) }
      }
    }
    if (preds.isEmpty) return Map("AUC" -> 0.0f)
    val auc = new AUC()
    auc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute())
  }

  // ==================== STAMP ====================
   private def trainStamp(model: STAMP, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
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
         val seqTokens = b.tokens
         if (seqTokens.nonEmpty) {
           val bs = seqTokens.get.size(0).toInt
           optimizer.zero_grad()
           val emb = model.forward(seqTokens.get)
           val itemFeats = b.itemFeatures
           val itemEmb = if (itemFeats.nonEmpty) itemFeats.values.head else torch.zeros_like(emb)
           val scores = emb.mul(itemEmb).sum(1)
           val y = b.labels.getOrElse(torch.zeros(bs.toLong)).view(bs, 1).toType(ScalarType.Float)
           val loss = lossFn.apply(scores.view(bs, 1), y)
           loss.backward()
           optimizer.step()
           totalLoss += loss.item().toDouble
           numBatches += 1
           epochPreds.appendAll(scores.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
           epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
         }
       }
       val avgLoss = if (numBatches > 0) totalLoss / numBatches else 0.0
       if (epochPreds.nonEmpty) {
         aucMetric.update(epochPreds.toArray, epochLabels.toArray)
         val trainAuc = aucMetric.compute()
         aucMetric.reset()
         println(f"[STAMP] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
       } else {
         println(f"[STAMP] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f")
       }
       if (device == "cuda") torch.emptyCache()
       e += 1
     }
   }

  private def evalStamp(model: STAMP, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      val seqTokens = b.tokens
      if (seqTokens.nonEmpty) {
        val emb = model.forward(seqTokens.get)
        val itemFeats = b.itemFeatures
        val itemEmb = if (itemFeats.nonEmpty) itemFeats.values.head else torch.zeros_like(emb)
        val scores = emb.mul(itemEmb).sum(1).sigmoid()
        preds.appendAll(scores.toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        b.labels.foreach { y => labels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) }
      }
    }
    if (preds.isEmpty) return Map("AUC" -> 0.0f)
    val auc = new AUC()
    auc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute())
  }

  // ==================== SINE ====================
   private def trainSine(model: SINE, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
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
         val feats = b.sparseFeatures
         val seqTokens = b.tokens
         if (feats.nonEmpty && seqTokens.nonEmpty) {
           val bs = feats.values.head.size(0).toInt
           optimizer.zero_grad()
           val emb = model.forward(feats, seqTokens.get)
           val itemFeats = b.itemFeatures
           val itemEmb = if (itemFeats.nonEmpty) itemFeats.values.head else torch.zeros_like(emb)
           val scores = emb.mul(itemEmb).sum(1)
           val y = b.labels.getOrElse(torch.zeros(bs.toLong)).view(bs, 1).toType(ScalarType.Float)
           val loss = lossFn.apply(scores.view(bs, 1), y)
           loss.backward()
           optimizer.step()
           totalLoss += loss.item().toDouble
           numBatches += 1
           epochPreds.appendAll(scores.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
           epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
         }
       }
       val avgLoss = if (numBatches > 0) totalLoss / numBatches else 0.0
       if (epochPreds.nonEmpty) {
         aucMetric.update(epochPreds.toArray, epochLabels.toArray)
         val trainAuc = aucMetric.compute()
         aucMetric.reset()
         println(f"[SINE] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
       } else {
         println(f"[SINE] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f")
       }
       if (device == "cuda") torch.emptyCache()
       e += 1
     }
   }

  private def evalSine(model: SINE, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      val feats = b.sparseFeatures
      val seqTokens = b.tokens
      if (feats.nonEmpty && seqTokens.nonEmpty) {
        val emb = model.forward(feats, seqTokens.get)
        val itemFeats = b.itemFeatures
        val itemEmb = if (itemFeats.nonEmpty) itemFeats.values.head else torch.zeros_like(emb)
        val scores = emb.mul(itemEmb).sum(1).sigmoid()
        preds.appendAll(scores.toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        b.labels.foreach { y => labels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) }
      }
    }
    if (preds.isEmpty) return Map("AUC" -> 0.0f)
    val auc = new AUC()
    auc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute())
  }
}
