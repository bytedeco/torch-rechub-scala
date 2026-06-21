package benchmarks.rec

import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import org.bytedeco.pytorch.*
import torchrec.Implicits.{RichTensor, tensor}
import torchrec.basic.features.{Feature, SequenceFeature, SparseFeature}
import torchrec.basic.losses.BCEWithLogitsLoss
import torchrec.basic.metrics.{AUC, Accuracy, LogLoss}
import torchrec.data.{DataLoader, SequenceDataset, TensorDataset}
import torchrec.models.ranking.*

import scala.collection.mutable
import scala.io.Source
import scala.util.Random

//[PASS] DIEN (ranking-seq) -> AUC=0.4730, LogLoss=0.9254, Accuracy=0.1120
//[PASS] ETA (ranking-seq) -> AUC=0.5173, LogLoss=0.7350, Accuracy=0.1100
//[PASS] SIM (ranking-seq) -> AUC=0.4773, LogLoss=0.6963, Accuracy=0.4461
//[PASS] BST (ranking-seq) -> AUC=0.4521, LogLoss=0.7130, Accuracy=0.3859
//[PASS] DeepFM (ranking) -> AUC=0.5354, LogLoss=4.2734, Accuracy=0.6680
//[PASS] xDeepFM (ranking) -> AUC=0.5453, LogLoss=5.5495, Accuracy=0.6120
//[PASS] AutoInt (ranking) -> AUC=0.8343, LogLoss=0.2746, Accuracy=0.8880
//[PASS] DeepFFM (ranking) -> AUC=0.4892, LogLoss=6.1050, Accuracy=0.5726
//[PASS] EDCN (ranking) -> AUC=0.6590, LogLoss=0.8888, Accuracy=0.1120
//[PASS] DIN (ranking-seq) -> AUC=0.5087, LogLoss=0.6523, Accuracy=0.8257
//[PASS] LiquidNetWork (ranking-seq) -> AUC=0.5134, LogLoss=0.7010, Accuracy=0.3859
//[PASS] MEMBA (ranking-seq) -> AUC=0.6164, LogLoss=0.6869, Accuracy=0.5705
object RunRankingPipeline2 {

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

    if (device == "cuda") { torch.emptyCache(); System.gc() }


    // ==================== DIEN ====================
    reports += runReport("DIEN", "ranking-seq") {
      val ds = buildSequenceCtrData(prepared, device)
      val features = List(SparseFeature("feat_0", prepared.numBins + 2, 8))
      val seqFeatures = List(SequenceFeature("seq_feat", prepared.numBins + 2, 8, maxLen = prepared.seqLen))
      // Validate indices in first batch before creating model to catch OOB early
      val firstBatch = ds._1.get(0)
      validateBatchIndices(firstBatch, features, seqFeatures)
      val model = new DIEN(features, seqFeatures, embedDim = 8, hiddenDim = 8, mlpDims = List(64L, 32L), dropout = 0.2f, device = device)
      trainDIEN(model, ds._1, epochs, batchSize, device)
      evalDIEN(model, ds._3, batchSize, device)
    }



    // ==================== ETA ====================
    reports += runReport("ETA", "ranking-seq") {
      val ds = buildSequenceCtrData(prepared, device)
      val features = List(SparseFeature("feat_0", prepared.numBins + 2, 8))
      val seqFeatures = List(SequenceFeature("seq_feat", prepared.numBins + 2, 8, maxLen = prepared.seqLen))
      val targetFeatures = List(SequenceFeature("target_feat", prepared.numBins + 2, 8, maxLen = 1))
      // Include targetFeatures in seqFeatures passed to the model so seqEmbedding contains 'target_feat'
      val seqWithTarget = seqFeatures ++ targetFeatures
      val firstBatch = ds._1.get(0)
      validateBatchIndices(firstBatch, features, seqWithTarget)
      val model = new ETA(features, seqWithTarget, embedDim = 8, hashSize = 64, attentionUnits = 36, topK = 20, mlpDims = List(256L, 128L, 64L), dropout = 0.2f, device = device)
      trainETA(model, ds._1, epochs, batchSize, device)
      evalETA(model, ds._3, batchSize, device)
    }

    // ==================== SIM ====================
    reports += runReport("SIM", "ranking-seq") {
      val ds = buildSequenceCtrData(prepared, device)
      val features = List(SparseFeature("feat_0", prepared.numBins + 2, 8))
      val seqFeatures = List(SequenceFeature("seq_feat", prepared.numBins + 2, 8, maxLen = prepared.seqLen))
      // Ensure target_feat embedding exists in seq/cate/time embedding layers
      val seqWithTarget = seqFeatures ++ List(SequenceFeature("target_feat", prepared.numBins + 2, 8, maxLen = 1))
      val firstBatch = ds._1.get(0)
      validateBatchIndices(firstBatch, features, seqWithTarget)
      val model = new SIM(features, seqWithTarget, seqWithTarget, seqWithTarget, embedDim = 8, attentionUnits = 36, mode = "hard", threshold = 0.8f, mlpDims = List(256L, 128L, 64L), dropout = 0.2f, device = device)
      trainSIM(model, ds._1, epochs, batchSize, device)
      evalSIM(model, ds._3, batchSize, device)
    }

    // ==================== BST ====================
    reports += runReport("BST", "ranking-seq") {
      val ds = buildSequenceCtrData(prepared, device)
      val features = List(SparseFeature("feat_0", prepared.numBins + 2, 8))
      val seqFeatures = List(SequenceFeature("seq_feat", prepared.numBins + 2, 8, maxLen = prepared.seqLen))
      val targetFeatures = List(SequenceFeature("target_feat", prepared.numBins + 2, 8, maxLen = 1))
      val firstBatch = ds._1.get(0)
      validateBatchIndices(firstBatch, features, seqFeatures ++ targetFeatures)
      val model = new BST(features, seqFeatures, targetFeatures, embedDim = 8, numHeads = 8, numLayers = 1, maxSeqLen = 51, mlpDims = List(256L, 128L), dropout = 0.2f, device = device)
      trainBST(model, ds._1, epochs, batchSize, device)
      evalBST(model, ds._3, batchSize, device)
    }
        // ==================== DeepFM ====================
        reports += runReport("DeepFM", "ranking") {
          val (tr, va, te, feats) = buildRankingData(prepared, embedDim = 8)
          val half = feats.size / 2
          val deepFeatures = feats.take(half)
          val fmFeatures = feats.drop(half)
          val model = new DeepFM(deepFeatures, fmFeatures, embedDim = 8, mlpDims = List(256L, 128L), dropout = 0.2f, device = device)
          trainDeepFM(model, tr, epochs, batchSize, device)
          evalDeepFM(model, te, batchSize, device)
        }

        // ==================== xDeepFM ====================
        reports += runReport("xDeepFM", "ranking") {
          val (tr, va, te, feats) = buildRankingData(prepared, embedDim = 8)
          val model = new xDeepFM(feats, 8, List(64, 32), List(64L, 32L), splitHalf = true, dropout = 0.2f, device = device)
          trainXDeepFM(model, tr, epochs, batchSize, device)
          evalXDeepFM(model, te, batchSize, device)
        }

        // ==================== AutoInt ====================
        reports += runReport("AutoInt", "ranking") {
          val (tr, va, te, feats) = buildRankingData(prepared, embedDim = 8)
          val model = new AutoInt(feats, embedDim = 8, numAttnHeads = 2, numLayers = 3, mlpDims = List(128L, 64L), dropout = 0.0f, useMlp = true, device = device)
          trainAutoInt(model, tr, epochs, batchSize, device)
          evalAutoInt(model, te, batchSize, device)
        }

        // ==================== DeepFFM ====================
        reports += runReport("DeepFFM", "ranking") {
          val (tr, va, te, feats) = buildRankingData(prepared, embedDim = 8)
          val model = new DeepFFM(feats, embedDim = 8, fieldNum = prepared.numFeatures, mlpDims = List(256L, 128L), dropout = 0.2f, device = device)
          trainDeepFFM(model, tr, epochs, batchSize, device)
          evalDeepFFM(model, te, batchSize, device)
        }

        // ==================== EDCN ====================
        reports += runReport("EDCN", "ranking") {
          val (tr, va, te, feats) = buildRankingData(prepared, embedDim = 8)
          val mlpParams = Map("dims" -> List(256L, 128L), "activation" -> "relu", "dropout" -> 0.2f)
          val model = new EDCN(feats, nCrossLayers = 3, mlpParams = mlpParams, bridgeType = "hadamard_product", useRegulationModule = true, temperature = 1.0f, device = device)
          trainEDCN(model, tr, epochs, batchSize, device)
          evalEDCN(model, te, batchSize, device)
        }

//     ==================== Sequence Models ====================

        // ==================== DIN ====================
        reports += runReport("DIN", "ranking-seq") {
          val ds = buildSequenceCtrData(prepared, device)
          val features = List(SparseFeature("feat_0", prepared.numBins + 2, 8))
          val seqFeatures = List(SequenceFeature("seq_feat", prepared.numBins + 2, 8, maxLen = prepared.seqLen))
          val model = new DIN(features, seqFeatures, embedDim = 8, mlpDims = List(256L, 128L), dropout = 0.2f, attentionUnits = 64, device = device)
          trainDIN(model, ds._1, epochs, batchSize, device)
          evalDIN(model, ds._3, batchSize, device)
        }

        // ==================== LiquidNetWork ====================
    reports += runReport("LiquidNetWork", "ranking-seq") {
      val ds = buildSequenceCtrData(prepared, device)
      val features = List(SparseFeature("feat_0", prepared.numBins + 2, 8))
      val seqFeatures = List(SequenceFeature("seq_feat", prepared.numBins + 2, 8, maxLen = prepared.seqLen))
      val model = new LiquidNetWork(features, seqFeatures, embedDim = 8, hiddenDim = 16, numOdeSteps = 3, mlpDims = List(64L, 32L), dropout = 0.2f, device = device)
      trainLiquidNetWork(model, ds._1, epochs, batchSize, device)
      evalLiquidNetWork(model, ds._3, batchSize, device)
    }

    // ==================== MEMBA ====================
    reports += runReport("MEMBA", "ranking-seq") {
      val ds = buildSequenceCtrData(prepared, device)
      val features = List(SparseFeature("feat_0", prepared.numBins + 2, 8))
      val seqFeatures = List(SequenceFeature("seq_feat", prepared.numBins + 2, 8, maxLen = prepared.seqLen))
      val model = new MEMBA(features, seqFeatures, embedDim = 8, numMemorySlots = 16, numHeads = 2, mlpDims = List(256L, 128L), dropout = 0.2f, device = device)
      trainMEMBA(model, ds._1, epochs, batchSize, device)
      evalMEMBA(model, ds._3, batchSize, device)
    }

    println("\n================ RANKING PIPELINE 2 SUMMARY ================")
    reports.foreach { r =>
      val m = r.metrics.map { case (k, v) => s"$k=${"%.4f".format(v)}" }.mkString(", ")
      println(s"[${if (r.ok) "PASS" else "FAIL"}] ${r.name} (${r.task}) -> ${if (m.nonEmpty) m else r.note}")
      if (!r.ok) println(s"  reason: ${r.note}")
    }
  }

  // Validate that indices in a batch do not exceed declared vocab sizes for given features.
  private def validateBatchIndices(batch: torchrec.data.Batch, sparseDefs: List[SparseFeature], seqDefs: List[SequenceFeature]): Unit = {
    def checkTensor(name: String, t: Tensor, vocabSize: Long): Unit = {
      try {
        val tLong = t.toType(ScalarType.Long)
        val tMin = tLong.min().item().toDouble.toLong
        val tMax = tLong.max().item().toDouble.toLong
        if (tMin < 0 || tMax >= vocabSize) {
          val msg = s"Feature '$name' indices out of range: min=$tMin max=$tMax vocabSize=$vocabSize"
          System.err.println("[ERROR] " + msg)
          throw new RuntimeException(msg)
        }
      } catch {
        case e: Throwable =>
          System.err.println(s"[WARN] Could not validate feature $name: ${e.getMessage}")
      }
    }

    sparseDefs.foreach { f =>
      batch.sparseFeatures.get(f.name).foreach(t => checkTensor(f.name, t, f.vocabSize))
    }
    seqDefs.foreach { f =>
      batch.sequenceFeatures.get(f.name).foreach(t => checkTensor(f.name, t, f.vocabSize))
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
      while (i < 29) { mins(i) = math.min(mins(i), f(i)); maxs(i) = math.max(maxs(i), f(i)); i += 1 }
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
    (Random(seed + 2).shuffle((ptr ++ ntr).toVector), Random(seed + 3).shuffle((pva ++ nva).toVector), Random(seed + 4).shuffle((pte ++ nte).toVector))
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

  private def buildSequenceCtrData(prepared: FraudPrepared, device: String): (SequenceDataset, SequenceDataset, SequenceDataset) = {
    def mk(rows: Vector[EncodedRow]): SequenceDataset = {
      val feat0 = tensor(rows.map(_.ids(0).toFloat).toArray, Array(rows.size.toLong)).toType(ScalarType.Long)
      val flatTokens = rows.flatMap(_.tokens.map(_.toFloat)).toArray
      val tokens = tensor(flatTokens, Array(rows.size.toLong, prepared.seqLen.toLong)).toType(ScalarType.Long)
      // Provide a target feature tensor (maxLen = 1) so models that expect a target_feat can find embeddings
      val targetArr = rows.map(_.ids(0).toFloat).toArray
      val targetFeat = tensor(targetArr, Array(rows.size.toLong, 1L)).toType(ScalarType.Long)
      val positions = tensor(Array.fill(rows.size * prepared.seqLen)(0f).zipWithIndex.map { case (_, i) => (i % prepared.seqLen).toFloat }, Array(rows.size.toLong, prepared.seqLen.toLong)).toType(ScalarType.Long)
      val labels = tensor(rows.map(_.label).toArray, Array(rows.size.toLong))
      new SequenceDataset(
        features = Map("feat_0" -> feat0),
        sequenceFeatures = Map("seq_feat" -> tokens, "target_feat" -> targetFeat),
        labels = Some(labels),
        tokens = Some(tokens),
        positions = Some(positions)
      )
    }
    (mk(prepared.train), mk(prepared.valid), mk(prepared.test))
  }

  // ==================== DeepFM ====================
  private def trainDeepFM(model: DeepFM, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0; var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float](); val epochLabels = mutable.ArrayBuffer[Float]()
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
          totalLoss += loss.item().toDouble; numBatches += 1
          epochPreds.appendAll(logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
          epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        }
      }
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      println(f"[DeepFM] Epoch ${e + 1}/$epochs - TrainLoss: ${totalLoss / numBatches}%.4f - TrainAUC: ${aucMetric.compute()}%.4f")
      aucMetric.reset()
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalDeepFM(model: DeepFM, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float](); val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) { val b = it.next(); b.labels.foreach { yRaw => preds.appendAll(model.forward(b.sparseFeatures).sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray); labels.appendAll(yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) } }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray); ll.update(preds.toArray, labels.toArray); acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== xDeepFM ====================
  private def trainXDeepFM(model: xDeepFM, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0; var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float](); val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        b.labels.foreach { yRaw =>
          optimizer.zero_grad()
          val logitsRaw = model.forward(b.sparseFeatures)
          val bs = yRaw.size(0).toInt
          val logits = if (logitsRaw.dim() == 2 && logitsRaw.size(1) == bs.toLong) logitsRaw.diag().view(bs, 1) else logitsRaw.view(bs, 1)
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble; numBatches += 1
          epochPreds.appendAll(logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
          epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        }
      }
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      println(f"[xDeepFM] Epoch ${e + 1}/$epochs - TrainLoss: ${totalLoss / numBatches}%.4f - TrainAUC: ${aucMetric.compute()}%.4f")
      aucMetric.reset()
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalXDeepFM(model: xDeepFM, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float](); val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) { val b = it.next(); b.labels.foreach { yRaw => val logitsRaw = model.forward(b.sparseFeatures); val bs = yRaw.size(0).toInt; val logits = if (logitsRaw.dim() == 2 && logitsRaw.size(1) == bs.toLong) logitsRaw.diag() else logitsRaw.squeeze(); preds.appendAll(logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray); labels.appendAll(yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) } }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray); ll.update(preds.toArray, labels.toArray); acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== AutoInt ====================
  private def trainAutoInt(model: AutoInt, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0; var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float](); val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) { val b = it.next(); b.labels.foreach { yRaw => optimizer.zero_grad(); val logits = model.forward(b.sparseFeatures); val bs = yRaw.size(0).toInt; val y = yRaw.view(bs, 1).toType(ScalarType.Float); val loss = lossFn.apply(logits, y); loss.backward(); optimizer.step(); totalLoss += loss.item().toDouble; numBatches += 1; epochPreds.appendAll(logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray); epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) } }
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      println(f"[AutoInt] Epoch ${e + 1}/$epochs - TrainLoss: ${totalLoss / numBatches}%.4f - TrainAUC: ${aucMetric.compute()}%.4f")
      aucMetric.reset()
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalAutoInt(model: AutoInt, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float](); val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) { val b = it.next(); b.labels.foreach { yRaw => preds.appendAll(model.forward(b.sparseFeatures).sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray); labels.appendAll(yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) } }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray); ll.update(preds.toArray, labels.toArray); acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== DeepFFM ====================
  private def trainDeepFFM(model: DeepFFM, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0; var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float](); val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) { val b = it.next(); b.labels.foreach { yRaw => optimizer.zero_grad(); val logits = model.forward(b.sparseFeatures); val bs = yRaw.size(0).toInt; val y = yRaw.view(bs, 1).toType(ScalarType.Float); val loss = lossFn.apply(logits, y); loss.backward(); optimizer.step(); totalLoss += loss.item().toDouble; numBatches += 1; epochPreds.appendAll(logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray); epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) } }
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      println(f"[DeepFFM] Epoch ${e + 1}/$epochs - TrainLoss: ${totalLoss / numBatches}%.4f - TrainAUC: ${aucMetric.compute()}%.4f")
      aucMetric.reset()
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalDeepFFM(model: DeepFFM, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float](); val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) { val b = it.next(); b.labels.foreach { yRaw => preds.appendAll(model.forward(b.sparseFeatures).sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray); labels.appendAll(yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) } }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray); ll.update(preds.toArray, labels.toArray); acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== EDCN ====================
  private def trainEDCN(model: EDCN, train: TensorDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0; var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float](); val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) { val b = it.next(); b.labels.foreach { yRaw => optimizer.zero_grad(); val logits = model.forward(b.sparseFeatures); val bs = yRaw.size(0).toInt; val y = yRaw.view(bs, 1).toType(ScalarType.Float); val loss = lossFn.apply(logits, y); loss.backward(); optimizer.step(); totalLoss += loss.item().toDouble; numBatches += 1; epochPreds.appendAll(logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray); epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) } }
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      println(f"[EDCN] Epoch ${e + 1}/$epochs - TrainLoss: ${totalLoss / numBatches}%.4f - TrainAUC: ${aucMetric.compute()}%.4f")
      aucMetric.reset()
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalEDCN(model: EDCN, test: TensorDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float](); val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) { val b = it.next(); b.labels.foreach { yRaw => preds.appendAll(model.forward(b.sparseFeatures).sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray); labels.appendAll(yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) } }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray); ll.update(preds.toArray, labels.toArray); acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== DIN ====================
  private def trainDIN(model: DIN, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0; var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float](); val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        val sparseFeats = b.sparseFeatures
        val seqFeats = b.sequenceFeatures
        val yRawOpt = b.labels
        if (sparseFeats.nonEmpty && seqFeats.nonEmpty && yRawOpt.isDefined) {
          val yRaw = yRawOpt.get
          val bs = yRaw.size(0).toInt
          optimizer.zero_grad()
          val targetTensor = sparseFeats.values.headOption.getOrElse(throw new RuntimeException("No sparse features")).view(bs, 1)
          val logits = model.forward(sparseFeats, seqFeats, targetTensor)
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble; numBatches += 1
          epochPreds.appendAll(logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
          epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        }
      }
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      println(f"[DIN] Epoch ${e + 1}/$epochs - TrainLoss: ${totalLoss / numBatches}%.4f - TrainAUC: ${aucMetric.compute()}%.4f")
      aucMetric.reset()
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalDIN(model: DIN, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float](); val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      val sparseFeats = b.sparseFeatures
      val seqFeats = b.sequenceFeatures
      val yRawOpt = b.labels
      if (sparseFeats.nonEmpty && seqFeats.nonEmpty && yRawOpt.isDefined) {
        val yRaw = yRawOpt.get
        val bs = yRaw.size(0).toInt
        val targetTensor = sparseFeats.values.headOption.getOrElse(throw new RuntimeException("No sparse features")).view(bs, 1)
        preds.appendAll(model.forward(sparseFeats, seqFeats, targetTensor).sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        labels.appendAll(yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray); ll.update(preds.toArray, labels.toArray); acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== DIEN ====================
  private def trainDIEN(model: DIEN, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0; var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float](); val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        val sparseFeats = b.sparseFeatures
        val seqFeats = b.sequenceFeatures
        val yRawOpt = b.labels
        if (sparseFeats.nonEmpty && seqFeats.nonEmpty && yRawOpt.isDefined) {
          val yRaw = yRawOpt.get
          val bs = yRaw.size(0).toInt
          optimizer.zero_grad()
          val logits = model.forward(sparseFeats, seqFeats)
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble; numBatches += 1
          epochPreds.appendAll(logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
          epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        }
      }
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      println(f"[DIEN] Epoch ${e + 1}/$epochs - TrainLoss: ${totalLoss / numBatches}%.4f - TrainAUC: ${aucMetric.compute()}%.4f")
      aucMetric.reset()
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalDIEN(model: DIEN, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float](); val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      val sparseFeats = b.sparseFeatures
      val seqFeats = b.sequenceFeatures
      val yRawOpt = b.labels
      if (sparseFeats.nonEmpty && seqFeats.nonEmpty && yRawOpt.isDefined) {
        val yRaw = yRawOpt.get
        preds.appendAll(model.forward(sparseFeats, seqFeats).sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        labels.appendAll(yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray); ll.update(preds.toArray, labels.toArray); acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== BST ====================
  private def trainBST(model: BST, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0; var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float](); val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        val sparseFeats = b.sparseFeatures
        val seqFeats = b.sequenceFeatures
        val yRawOpt = b.labels
        if (sparseFeats.nonEmpty && seqFeats.nonEmpty && yRawOpt.isDefined) {
          val yRaw = yRawOpt.get
          val bs = yRaw.size(0).toInt
          optimizer.zero_grad()
          val logits = model.forward(sparseFeats, seqFeats)
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble; numBatches += 1
          epochPreds.appendAll(logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
          epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        }
      }
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      println(f"[BST] Epoch ${e + 1}/$epochs - TrainLoss: ${totalLoss / numBatches}%.4f - TrainAUC: ${aucMetric.compute()}%.4f")
      aucMetric.reset()
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalBST(model: BST, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float](); val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      val sparseFeats = b.sparseFeatures
      val seqFeats = b.sequenceFeatures
      val yRawOpt = b.labels
      if (sparseFeats.nonEmpty && seqFeats.nonEmpty && yRawOpt.isDefined) {
        val yRaw = yRawOpt.get
        preds.appendAll(model.forward(sparseFeats, seqFeats).sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        labels.appendAll(yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray); ll.update(preds.toArray, labels.toArray); acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== ETA ====================
  private def trainETA(model: ETA, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0; var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float](); val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        val sparseFeats = b.sparseFeatures
        val seqFeats = b.sequenceFeatures
        val yRawOpt = b.labels
        if (sparseFeats.nonEmpty && seqFeats.nonEmpty && yRawOpt.isDefined) {
          val yRaw = yRawOpt.get
          val bs = yRaw.size(0).toInt
          optimizer.zero_grad()
           val targetFeats = seqFeats.get("target_feat") match {
             case Some(t) => Map("target_feat" -> t)
             case None => sparseFeats.map { case (k, v) => k.replace("feat_0", "target_feat") -> v }
           }
           val logits = model.forward(sparseFeats, seqFeats, targetFeats)
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble; numBatches += 1
          epochPreds.appendAll(logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
          epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        }
      }
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      println(f"[ETA] Epoch ${e + 1}/$epochs - TrainLoss: ${totalLoss / numBatches}%.4f - TrainAUC: ${aucMetric.compute()}%.4f")
      aucMetric.reset()
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalETA(model: ETA, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float](); val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      val sparseFeats = b.sparseFeatures
      val seqFeats = b.sequenceFeatures
      val yRawOpt = b.labels
      if (sparseFeats.nonEmpty && seqFeats.nonEmpty && yRawOpt.isDefined) {
        val yRaw = yRawOpt.get
        val targetFeats = seqFeats.get("target_feat") match {
          case Some(t) => Map("target_feat" -> t)
          case None => sparseFeats.map { case (k, v) => k.replace("feat_0", "target_feat") -> v }
        }
        preds.appendAll(model.forward(sparseFeats, seqFeats, targetFeats).sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        labels.appendAll(yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray); ll.update(preds.toArray, labels.toArray); acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== SIM ====================
  private def trainSIM(model: SIM, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0; var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float](); val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        val sparseFeats = b.sparseFeatures
        val seqFeats = b.sequenceFeatures
        val yRawOpt = b.labels
        if (sparseFeats.nonEmpty && seqFeats.nonEmpty && yRawOpt.isDefined) {
          val yRaw = yRawOpt.get
          val bs = yRaw.size(0).toInt
          optimizer.zero_grad()
           val targetFeats = seqFeats.get("target_feat") match {
             case Some(t) => Map("target_feat" -> t)
             case None => sparseFeats.map { case (k, v) => k.replace("feat_0", "target_feat") -> v }
           }
           val logits = model.forward(sparseFeats, seqFeats, seqFeats, seqFeats, targetFeats)
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble; numBatches += 1
          epochPreds.appendAll(logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
          epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        }
      }
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      println(f"[SIM] Epoch ${e + 1}/$epochs - TrainLoss: ${totalLoss / numBatches}%.4f - TrainAUC: ${aucMetric.compute()}%.4f")
      aucMetric.reset()
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalSIM(model: SIM, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float](); val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      val sparseFeats = b.sparseFeatures
      val seqFeats = b.sequenceFeatures
      val yRawOpt = b.labels
      if (sparseFeats.nonEmpty && seqFeats.nonEmpty && yRawOpt.isDefined) {
        val yRaw = yRawOpt.get
        val targetFeats = seqFeats.get("target_feat") match {
          case Some(t) => Map("target_feat" -> t)
          case None => sparseFeats.map { case (k, v) => k.replace("feat_0", "target_feat") -> v }
        }
        preds.appendAll(model.forward(sparseFeats, seqFeats, seqFeats, seqFeats, targetFeats).sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        labels.appendAll(yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray); ll.update(preds.toArray, labels.toArray); acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== LiquidNetWork ====================
  private def trainLiquidNetWork(model: LiquidNetWork, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0; var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float](); val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        val sparseFeats = b.sparseFeatures
        val seqFeats = b.sequenceFeatures
        val yRawOpt = b.labels
        if (sparseFeats.nonEmpty && seqFeats.nonEmpty && yRawOpt.isDefined) {
          val yRaw = yRawOpt.get
          val bs = yRaw.size(0).toInt
          optimizer.zero_grad()
          val logits = model.forward(sparseFeats, seqFeats)
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble; numBatches += 1
          epochPreds.appendAll(logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
          epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        }
      }
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      println(f"[LiquidNetWork] Epoch ${e + 1}/$epochs - TrainLoss: ${totalLoss / numBatches}%.4f - TrainAUC: ${aucMetric.compute()}%.4f")
      aucMetric.reset()
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalLiquidNetWork(model: LiquidNetWork, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float](); val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      val sparseFeats = b.sparseFeatures
      val seqFeats = b.sequenceFeatures
      val yRawOpt = b.labels
      if (sparseFeats.nonEmpty && seqFeats.nonEmpty && yRawOpt.isDefined) {
        val yRaw = yRawOpt.get
        preds.appendAll(model.forward(sparseFeats, seqFeats).sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        labels.appendAll(yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray); ll.update(preds.toArray, labels.toArray); acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  // ==================== MEMBA ====================
  private def trainMEMBA(model: MEMBA, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0; var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float](); val epochLabels = mutable.ArrayBuffer[Float]()
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()
        val sparseFeats = b.sparseFeatures
        val seqFeats = b.sequenceFeatures
        val yRawOpt = b.labels
        if (sparseFeats.nonEmpty && seqFeats.nonEmpty && yRawOpt.isDefined) {
          val yRaw = yRawOpt.get
          val bs = yRaw.size(0).toInt
          optimizer.zero_grad()
          val targetIdx = sparseFeats.values.headOption.getOrElse(throw new RuntimeException("No sparse features")).view(bs, 1)
          val logits = model.forward(sparseFeats, seqFeats, targetIdx)
          val y = yRaw.view(bs, 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward(); optimizer.step()
          totalLoss += loss.item().toDouble; numBatches += 1
          epochPreds.appendAll(logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
          epochLabels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        }
      }
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      println(f"[MEMBA] Epoch ${e + 1}/$epochs - TrainLoss: ${totalLoss / numBatches}%.4f - TrainAUC: ${aucMetric.compute()}%.4f")
      aucMetric.reset()
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }
  private def evalMEMBA(model: MEMBA, test: SequenceDataset, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float](); val labels = mutable.ArrayBuffer[Float]()
    val loader = new DataLoader(test, batchSize = batchSize, shuffle = false, device = device)
    val it = loader.iterator
    while (it.hasNext) {
      val b = it.next()
      val sparseFeats = b.sparseFeatures
      val seqFeats = b.sequenceFeatures
      val yRawOpt = b.labels
      if (sparseFeats.nonEmpty && seqFeats.nonEmpty && yRawOpt.isDefined) {
        val yRaw = yRawOpt.get
        val bs = yRaw.size(0).toInt
        val targetIdx = sparseFeats.values.headOption.getOrElse(throw new RuntimeException("No sparse features")).view(bs, 1)
        preds.appendAll(model.forward(sparseFeats, seqFeats, targetIdx).sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        labels.appendAll(yRaw.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
      }
    }
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray); ll.update(preds.toArray, labels.toArray); acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }
}