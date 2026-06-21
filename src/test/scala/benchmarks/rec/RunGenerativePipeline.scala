package benchmarks.rec

import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import org.bytedeco.pytorch.*
import torchrec.Implicits.{RichTensor, tensor}
import torchrec.basic.features.{SequenceFeature, SparseFeature}
import torchrec.basic.losses.BCEWithLogitsLoss
import torchrec.basic.metrics.{AUC, Accuracy, LogLoss}
import torchrec.data.{DataLoader, MultiTaskDataset, SequenceDataset, TensorDataset}
import torchrec.models.generative.*
import torchrec.models.matching.{MAMBA, MIND}
import torchrec.models.multi_task.*
import torchrec.models.ranking.{LiquidNetWork, MEMBA, xDeepFM}
import torchrec.trainers.{CTRTrainer, MTLTrainer, MatchTrainer}

import scala.collection.mutable
import scala.io.Source
import scala.util.Random

object RunGenerativePipeline {

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
    val maxRows = p.getOrElse("max_rows", "1900").toInt
    val batchSize = p.getOrElse("batch_size", "64").toInt
    val epochs = p.getOrElse("epochs", "15").toInt
    val device = p.getOrElse("device", "cuda")
    val seed = p.getOrElse("seed", "2026").toInt
    val bins = p.getOrElse("bins", "128").toInt
    val seqLen = p.getOrElse("seq_len", "20").toInt

    val prepared = prepareFraud(csvPath, maxRows, bins, seqLen, seed, itemEmbedDim = 16)

    val reports = mutable.ArrayBuffer[ModelReport]()

    // ========== HLLM ==========
    clearMemory(device)
    reports += runReport("HLLM", "generative-ctr") {
      val ds = buildSequenceCtrData(prepared, device)
      val embData = Array.fill((prepared.numBins + 2) * 16)(0.01f)
      val itemEmb = tensor(embData, Array((prepared.numBins + 2).toLong, 16L))
      val model = new HLLM(itemEmbeddings = itemEmb, vocabSize = prepared.numBins + 2, dModel = 16, nHeads = 2, nLayers = 2, dropout = 0.2f, device = device)
      val trainer = new CTRTrainer(model, learningRate = 1e-3f, device = device, numEpochs = epochs, verbose = true)
      trainer.fit(new DataLoader(ds._1, batchSize = batchSize, shuffle = true, device = device), Some(new DataLoader(ds._2, batchSize = batchSize, shuffle = false, device = device)))
      trainer.evaluate(new DataLoader(ds._3, batchSize = batchSize, shuffle = false, device = device))
    }
    clearMemory(device)

    // ========== HSTU ==========
    clearMemory(device)
    reports += runReport("HSTU", "generative-custom") {
      val hstuDevice = torchrec.utils.DeviceSupport.backend
      val (train0, valid0, test0, trainY0, validY0, testY0) = buildHstuTensors(prepared, hstuDevice)
      val dev = new Device(hstuDevice)
      val train = (train0._1.to(dev, ScalarType.Long), train0._2.to(dev, ScalarType.Float), train0._3.to(dev, ScalarType.Long))
      val valid = (valid0._1.to(dev, ScalarType.Long), valid0._2.to(dev, ScalarType.Float), valid0._3.to(dev, ScalarType.Long))
      val test = (test0._1.to(dev, ScalarType.Long), test0._2.to(dev, ScalarType.Float), test0._3.to(dev, ScalarType.Long))
      val trainY = trainY0.to(dev, ScalarType.Float)
      val validY = validY0.to(dev, ScalarType.Float)
      val testY = testY0.to(dev, ScalarType.Float)
      val model = new HSTU(vocabSize = (prepared.numBins + 2).toLong, dModel = 32, nHeads = 2, nLayers = 2, maxSeqLen = prepared.seqLen, dropout = 0.2f, device = hstuDevice)
      trainHstu(model, train, trainY, valid, validY, epochs, batchSize, device)
      evalHstu(model, test._1, test._2, test._3, Some(testY))
    }
    clearMemory(device)

    // ========== LLM4Rec ==========
    clearMemory(device)
    reports += runReport("LLM4Rec", "generative-ctr") {
      val ds = buildSequenceCtrData(prepared, device)
      val model = new LLM4Rec(vocabSize = prepared.numBins + 2, embedDim = 16, numHeads = 2, numLayers = 2, maxSeqLen = prepared.seqLen + 1, mlpDims = List(64L, 32L), dropout = 0.1f, device = device)
      val trainer = new CTRTrainer(model, learningRate = 1e-3f, device = device, numEpochs = epochs, verbose = true)
      trainer.fit(new DataLoader(ds._1, batchSize = batchSize, shuffle = true, device = device), Some(new DataLoader(ds._2, batchSize = batchSize, shuffle = false, device = device)))
      trainer.evaluate(new DataLoader(ds._3, batchSize = batchSize, shuffle = false, device = device))
    }
    clearMemory(device)

    // ========== RQVAE ==========
    clearMemory(device)
    reports += runReport("RQVAE", "generative-anomaly") {
      val model = new RQVAE(in_dim = prepared.numFeatures, num_emb_list = List(256, 256, 256, 256), e_dim = 64, device = device)
      trainRQVAE(model, prepared.train, epochs = epochs, batchSize = batchSize, device = device)
      evalRQVAE(model, prepared.test, device)
    }
    clearMemory(device)

    // ========== TIGER ==========
    clearMemory(device)
    reports += runReport("TIGER", "generative-retrieval") {
      val model = buildTigerModel(prepared, device)
      trainTiger(model, prepared, epochs, batchSize, device)
      evalTiger(model, prepared, batchSize, device)
    }
    clearMemory(device)

    println("\n============ GENERATIVE PIPELINE SUMMARY ============")
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

  private def clearMemory(device: String): Unit = {
    if (device == "cuda") {
      torch.emptyCache()
      System.gc()
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

  private def buildSequenceCtrData(prepared: FraudPrepared, device: String): (SequenceDataset, SequenceDataset, SequenceDataset) = {
    def mk(rows: Vector[EncodedRow]): SequenceDataset = {
      val feat0 = tensor(rows.map(_.ids(0).toFloat).toArray, Array(rows.size.toLong)).toType(ScalarType.Long)
      val flatTokens = rows.flatMap(_.tokens.map(_.toFloat)).toArray
      val tokens = tensor(flatTokens, Array(rows.size.toLong, prepared.seqLen.toLong)).toType(ScalarType.Long)
      val positions = tensor(Array.fill(rows.size * prepared.seqLen)(0f).zipWithIndex.map { case (_, i) => (i % prepared.seqLen).toFloat }, Array(rows.size.toLong, prepared.seqLen.toLong)).toType(ScalarType.Long)
      val labels = tensor(rows.map(_.label).toArray, Array(rows.size.toLong))
      new SequenceDataset(
        features = Map("feat_0" -> feat0),
        sequenceFeatures = Map("seq_feat" -> tokens),
        labels = Some(labels),
        tokens = Some(tokens),
        positions = Some(positions)
      )
    }
    (mk(prepared.train), mk(prepared.valid), mk(prepared.test))
  }

  private def buildHstuTensors(prepared: FraudPrepared, device: String): ((Tensor, Tensor, Tensor), (Tensor, Tensor, Tensor), (Tensor, Tensor, Tensor), Tensor, Tensor, Tensor) = {
    def pack(rows: Vector[EncodedRow]): (Tensor, Tensor, Tensor, Tensor) = {
      val flatTokens = rows.flatMap(_.tokens.map(_.toFloat)).toArray
      val tokens = tensor(flatTokens, Array(rows.size.toLong, prepared.seqLen.toLong)).toType(ScalarType.Long)
      val zerosTime = torch.zeros(Array(rows.size.toLong, prepared.seqLen.toLong), new TensorOptions().dtype(new org.bytedeco.pytorch.ScalarTypeOptional(ScalarType.Float)))
      val targetIds = tensor(rows.map(_.ids(0).toFloat).toArray, Array(rows.size.toLong)).toType(ScalarType.Long)
      val labels = tensor(rows.map(_.label).toArray, Array(rows.size.toLong)).toType(ScalarType.Float)
      (tokens, zerosTime, targetIds, labels)
    }
    val (trT, trTime, trTarget, trY) = pack(prepared.train)
    val (vaT, vaTime, vaTarget, vaY) = pack(prepared.valid)
    val (teT, teTime, teTarget, teY) = pack(prepared.test)
    ((trT, trTime, trTarget), (vaT, vaTime, vaTarget), (teT, teTime, teTarget), trY, vaY, teY)
  }

  // Build TIGER model with item embeddings from the dataset
  private def buildTigerModel(prepared: FraudPrepared, device: String): TIGER = {
    val numItems = (prepared.numBins + 2).toLong
    val embedDim = 16
    // Create item embeddings from item vectors in the dataset
    val itemEmbData = prepared.train.flatMap(_.itemVec).toArray
    val itemEmb = tensor(itemEmbData, Array(numItems, embedDim.toLong))
    new TIGER(itemEmbeddings = itemEmb, embedDim = embedDim, hiddenDim = 128, numLayers = 2, dropout = 0.2f, device = device)
  }

  private def trainTiger(model: TIGER, prepared: FraudPrepared, epochs: Int, batchSize: Int, device: String): Unit = {
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val n = prepared.train.size
    val aucMetric = new AUC()

    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()

      var i = 0
      while (i < n) {
        val bs = math.min(batchSize, n - i)
        // Build sequence tensor for batch
        val batchRows = prepared.train.slice(i, i + bs)
        val flatTokens = batchRows.flatMap(_.tokens.map(_.toFloat)).toArray
        val tokens = tensor(flatTokens, Array(bs.toLong, prepared.seqLen.toLong)).toType(ScalarType.Long)
        val labels = tensor(batchRows.map(_.label).toArray, Array(bs.toLong))

        optimizer.zero_grad()
        val encoded = model.forward(tokens)
        // Use mean pooling over hidden dim for prediction
        val logits = encoded.mean(1).view(bs, 1)
        val y = labels.view(bs, 1).toType(ScalarType.Float)
        val loss = lossFn.apply(logits, y)
        loss.backward(); optimizer.step()

        totalLoss += loss.item().toDouble
        numBatches += 1

        val sigmoidPred = logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val labelArr = y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        epochPreds.appendAll(sigmoidPred)
        epochLabels.appendAll(labelArr)

        i += bs
      }

      val avgLoss = totalLoss / numBatches
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute()
      aucMetric.reset()
      println(f"[TIGER] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")

      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }

  private def evalTiger(model: TIGER, prepared: FraudPrepared, batchSize: Int, device: String): Map[String, Float] = {
    val preds = mutable.ArrayBuffer[Float]()
    val labels = mutable.ArrayBuffer[Float]()

    var i = 0
    val n = prepared.test.size
    while (i < n) {
      val bs = math.min(batchSize, n - i)
      val batchRows = prepared.test.slice(i, i + bs)
      val flatTokens = batchRows.flatMap(_.tokens.map(_.toFloat)).toArray
      val tokens = tensor(flatTokens, Array(bs.toLong, prepared.seqLen.toLong)).toType(ScalarType.Long)
      val labelArr = batchRows.map(_.label).toArray

      val encoded = model.forward(tokens)
      val logits = encoded.mean(1)
      preds.appendAll(logits.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray)
      labels.appendAll(labelArr)

      i += bs
    }

    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray)
    ll.update(preds.toArray, labels.toArray)
    acc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  private def trainHstu(model: HSTU, train: (Tensor, Tensor, Tensor), trainY: Tensor, valid: (Tensor, Tensor, Tensor), validY: Tensor, epochs: Int, batchSize: Int, device: String): Unit = {
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val n = trainY.size(0).toInt
    val aucMetric = new AUC()
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()
      var i = 0
      while (i < n) {
        val bs = math.min(batchSize, n - i)
        val tok = train._1.narrow(0, i, bs)
        val tim = train._2.narrow(0, i, bs)
        val tgt = train._3.narrow(0, i, bs).view(bs, 1)
        val y = trainY.narrow(0, i, bs).view(bs, 1)
        optimizer.zero_grad()
        val logits = model.forward(tok, tim)
        val last = logits.select(1, logits.size(1) - 1)
        val pred = last.gather(1, tgt)
        val loss = lossFn.apply(pred, y)
        loss.backward(); optimizer.step()
        totalLoss += loss.item().toDouble
        numBatches += 1
        val sigmoidPred = pred.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val labels = y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        epochPreds.appendAll(sigmoidPred)
        epochLabels.appendAll(labels)
        i += bs
      }
      val avgLoss = totalLoss / numBatches
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute()
      aucMetric.reset()
      println(f"[HSTU] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f - TrainAUC: ${trainAuc}%.4f")
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }

  private def evalHstu(model: HSTU, test: Tensor, time: Tensor, target: Tensor, labels: Option[Tensor]): Map[String, Float] = {
    val ys = labels.getOrElse(torch.zeros(Array(test.size(0)), new TensorOptions().dtype(new org.bytedeco.pytorch.ScalarTypeOptional(ScalarType.Float))))
    val logits = model.forward(test, time)
    val last = logits.select(1, logits.size(1) - 1)
    val pred = last.gather(1, target.view(target.size(0), 1)).sigmoid().squeeze()
    val predArr = pred.toType(ScalarType.Float).contiguous().cpu().toFloatArray
    val yArr = ys.toType(ScalarType.Float).contiguous().cpu().toFloatArray
    val auc = new AUC(); val ll = new LogLoss(); val acc = new Accuracy()
    auc.update(predArr, yArr); ll.update(predArr, yArr); acc.update(predArr, yArr)
    Map("AUC" -> auc.compute(), "LogLoss" -> ll.compute(), "Accuracy" -> acc.compute())
  }

  private def trainRQVAE(model: RQVAE, train: Vector[EncodedRow], epochs: Int, batchSize: Int, device: String): Unit = {
    val xAll = tensor(train.flatMap(_.dense).toArray, Array(train.size.toLong, train.headOption.map(_.dense.length).getOrElse(1).toLong)).toType(ScalarType.Float)
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      var i = 0
      while (i < train.size) {
        val bs = math.min(batchSize, train.size - i)
        val x = xAll.narrow(0, i, bs)
        optimizer.zero_grad()
        val (decoded, qloss, _) = model.forward(x, use_sk = false)
        val (totalLossBatch, _) = model.computeLoss(decoded, qloss, x, "mse")
        totalLossBatch.backward(); optimizer.step()
        totalLoss += totalLossBatch.item().toDouble
        numBatches += 1
        i += bs
      }
      val avgLoss = totalLoss / numBatches
      println(f"[RQVAE] Epoch ${e + 1}/${epochs} - TrainLoss: ${avgLoss}%.4f")
      if (device == "cuda") torch.emptyCache()
      e += 1
    }
  }

  private def evalRQVAE(model: RQVAE, test: Vector[EncodedRow], device: String): Map[String, Float] = {
    val x = tensor(test.flatMap(_.dense).toArray, Array(test.size.toLong, test.headOption.map(_.dense.length).getOrElse(1).toLong)).toType(ScalarType.Float)
    val (decoded, _, _) = model.forward(x, use_sk = false)
    val err = torch.pow(decoded.sub(x), new Scalar(2.0)).mean(1)
    val predArr = err.toType(ScalarType.Float).contiguous().cpu().toFloatArray
    val yArr = test.map(_.label).toArray
    val auc = new AUC(); auc.update(predArr, yArr)
    Map("AUC(recon_error)" -> auc.compute())
  }
}
