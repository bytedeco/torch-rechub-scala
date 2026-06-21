package benchmarks.rec

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.basic.losses.BCEWithLogitsLoss
import torchrec.basic.metrics.{AUC, Accuracy, LogLoss}
import torchrec.data.{DataLoader, SequenceDataset, TensorDataset}
import torchrec.models.knowledge_tracing.*
import torchrec.models.knowledge_tracing.datasets.KTDataset

import scala.collection.mutable
import scala.io.Source
import scala.util.Random

/**
 * Benchmark pipeline for knowledge tracing models on fraud data.
 *
 * Converts fraud_data.csv into KT format:
 * - Each feature column becomes a "concept"
 * - Historical feature values form the response sequence (correct/incorrect based on label)
 * - Treats each row as a student interaction sequence
 *
 * Trains 22 KT models: AKT, ATDKT, ATKT, CSKT, DIMKT, DKT, DKTForget, DKVMN,
 * GKT, IEKT, LPKT, MTKT, PromptKT, QDKT, RKT, RobustKT, SAINT, SAINTPlusPlus,
 * SAKT, SKVMN, StableKT, UKT
 */
object RunKnowledgeTracingPipeline {

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

  case class ModelReport(name: String, ok: Boolean, trainLoss: Float, trainAuc: Float, note: String)

  def main(args: Array[String]): Unit = {
    val csvPath = "/home/muller/IdeaProjects/torch-rechub-scala/src/main/resources/fraud_data.csv"
    val maxRows = 21900
    val batchSize = 256
    val epochs = 150
    val device = "cuda"
    val seed = 2026
    val bins = 128
    val seqLen = 20

    val prepared = prepareFraud(csvPath, maxRows, bins, seqLen, seed, itemEmbedDim = 16)

    val reports = mutable.ArrayBuffer[ModelReport]()

    // Convert fraud data to KT format
    val ktData = buildKTSequenceData(prepared, device)
    val numConcepts = (prepared.numBins + 2).toLong
    val numQuestions = numConcepts

    println(s"\n========== Knowledge Tracing Benchmark ==========")
    println(s"numConcepts=$numConcepts, numQuestions=$numQuestions, seqLen=$seqLen")
    println(s"trainSamples=${ktData._1.size}, validSamples=${ktData._2.size}, testSamples=${ktData._3.size}")

    // ---- DKT ----
    reports += runKTReport("DKT", device, epochs, batchSize) {
      val model = new DKT(numConcepts, embedDim = 64, numLayers = 1, dropout = 0.2f, device = device)
      trainKTModel(model, "DKT", ktData, epochs, batchSize, device, numConcepts) {
        case (conceptIds, responses) => model.predict(conceptIds, responses)
      }
    }

    // ---- DKTForget ----
    reports += runKTReport("DKTForget", device, epochs, batchSize) {
      val model = new DKTForget(numConcepts, embedDim = 64, numLayers = 1, dropout = 0.2f, device = device)
      trainKTModel(model, "DKTForget", ktData, epochs, batchSize, device, numConcepts) {
        case (conceptIds, responses) => model.predict(conceptIds, responses)
      }
    }

    // ---- ATDKT ----
    reports += runKTReport("ATDKT", device, epochs, batchSize) {
      val model = new ATDKT(numConcepts, embedDim = 64, numLayers = 1, numHeads = 4, dropout = 0.2f, device = device)
      trainKTModel(model, "DKT", ktData, epochs, batchSize, device, numConcepts) {
        case (conceptIds, responses) => model.predict(conceptIds, responses)
      }
    }

    // ---- ATKT ----
    reports += runKTReport("ATKT", device, epochs, batchSize) {
      val model = new ATKT(numConcepts, skillDim = 64, answerDim = 64, hiddenDim = 64, attentionDim = 80, dropout = 0.2f, fix = true, device = device)
      trainKTAtkt(model, ktData, epochs, batchSize, device, numConcepts)
    }

    // ---- AKT ----
    reports += runKTReport("AKT", device, epochs, batchSize) {
      val model = new AKT(numConcepts, embedDim = 64, numHeads = 8, numBlocks = 2, ffnDim = 128, dropout = 0.1f, device = device)
      trainKTModel(model, "AKT", ktData, epochs, batchSize, device, numConcepts) {
        case (conceptIds, responses) => model.predict(conceptIds, responses)
      }
    }

    // ---- CSKT ----
    reports += runKTReport("CSKT", device, epochs, batchSize) {
      val model = new CSKT(numConcepts, embedDim = 64, numHeads = 8, numBlocks = 2, dropout = 0.1f, device = device)
      trainKTModel(model, "CSKT", ktData, epochs, batchSize, device, numConcepts) {
        case (conceptIds, responses) => model.predict(conceptIds, responses)
      }
    }

    // ---- DIMKT ----
    reports += runKTReport("DIMKT", device, epochs, batchSize) {
      val model = new DIMKT(numConcepts, embedDim = 64, numHeads = 4, numBlocks = 2, hiddenDim = 64, dropout = 0.1f, device = device)
      trainKTModel(model, "DIMKT", ktData, epochs, batchSize, device, numConcepts) {
        case (conceptIds, responses) => model.predict(conceptIds, conceptIds, conceptIds, responses)
      }
    }

    // ---- DKVMN ----
    reports += runKTReport("DKVMN", device, epochs, batchSize) {
      val model = new DKVMN(numConcepts, numQuestions, memDim = 64, memSize = 20, dropout = 0.2f, device = device)
      trainKTDkvmn(model, ktData, epochs, batchSize, device, numConcepts, numQuestions)
    }

    // ---- GKT ----
    reports += runKTReport("GKT", device, epochs, batchSize) {
      val model = new GKT(numConcepts, embedDim = 64, hiddenDim = 64, dropout = 0.5f, graphType = "dense", device = device)
      trainKTModel(model, "GKT", ktData, epochs, batchSize, device, numConcepts) {
        case (conceptIds, responses) => model.predict(conceptIds, responses)
      }
    }

    // ---- IEKT ----
    reports += runKTReport("IEKT", device, epochs, batchSize) {
      val model = new IEKT(numConcepts, embedDim = 64, numCogLevels = 10, numAcqLevels = 10, numLayers = 1, dropout = 0.2f, gamma = 0.93f, device = device)
      trainKTModel(model, "IEKT", ktData, epochs, batchSize, device, numConcepts) {
        case (conceptIds, responses) => model.predict(conceptIds, responses)
      }
    }

    // ---- LPKT ----
    reports += runKTReport("LPKT", device, epochs, batchSize) {
      val model = new LPKT(numExercises = numConcepts, numConcepts = numConcepts, numActionTypes = 1, embedDim = 64, exerciseDim = 64, dropout = 0.2f, device = device)
      trainKTLpkt(model, ktData, epochs, batchSize, device, numConcepts)
    }

    // ---- MTKT ----
    reports += runKTReport("MTKT", device, epochs, batchSize) {
      val model = new MTKT(numConcepts, embedDim = 64, numHeads = 8, numBlocks = 2, dropout = 0.2f, device = device)
      trainKTModel(model, "MTKT", ktData, epochs, batchSize, device, numConcepts) {
        case (conceptIds, responses) => model.predict(conceptIds, responses)
      }
    }

    // ---- PromptKT ----
    reports += runKTReport("PromptKT", device, epochs, batchSize) {
      val model = new PromptKT(numConcepts, embedDim = 64, numHeads = 8, numBlocks = 2, dropout = 0.2f, device = device)
      trainKTModel(model, "PromptKT", ktData, epochs, batchSize, device, numConcepts) {
        case (conceptIds, responses) => model.predict(conceptIds, responses)
      }
    }

    // ---- QDKT ----
    reports += runKTReport("QDKT", device, epochs, batchSize) {
      val model = new QDKT(numQuestions, numConcepts, embedDim = 64, numLayers = 1, dropout = 0.2f, device = device)
      trainKTQdkt(model, ktData, epochs, batchSize, device, numConcepts, numQuestions)
    }

    // ---- RKT ----
    reports += runKTReport("RKT", device, epochs, batchSize) {
      val model = new RKT(numConcepts, embedDim = 64, numHeads = 4, numLayers = 1, dropout = 0.2f, device = device)
      trainKTModel(model, "RKT", ktData, epochs, batchSize, device, numConcepts) {
        case (conceptIds, responses) => model.predict(conceptIds, responses)
      }
    }

    // ---- RobustKT ----
    reports += runKTReport("RobustKT", device, epochs, batchSize) {
      val model = new RobustKT(numConcepts, embedDim = 64, numHeads = 8, numBlocks = 2, kernelSize = 5, dropout = 0.2f, device = device)
      trainKTModel(model, "RobustKT", ktData, epochs, batchSize, device, numConcepts) {
        case (conceptIds, responses) => model.predict(conceptIds, responses)
      }
    }

    // ---- SAINT ----
    reports += runKTReport("SAINT", device, epochs, batchSize) {
      val model = new SAINT(numExercises = numConcepts, numCategories = numConcepts, numResponses = 2, embedDim = 64, numHeads = 8, numEncoderBlocks = 2, numDecoderBlocks = 2, ffnDim = 128, dropout = 0.2f, device = device)
      trainKTSaint(model, ktData, epochs, batchSize, device, numConcepts)
    }

    // ---- SAINTPlusPlus ----
    reports += runKTReport("SAINTPlusPlus", device, epochs, batchSize) {
      val model = new SAINTPlusPlus(numExercises = numConcepts, numCategories = numConcepts, numResponses = 2, embedDim = 64, numHeads = 8, numEncoderBlocks = 2, numDecoderBlocks = 2, ffnDim = 128, dropout = 0.2f, device = device)
      trainKTSaint(model, ktData, epochs, batchSize, device, numConcepts)
    }

    // ---- SAKT ----
    reports += runKTReport("SAKT", device, epochs, batchSize) {
      val model = new SAKT(numConcepts, embedDim = 64, numHeads = 8, numBlocks = 2, dropout = 0.2f, device = device)
      trainKTModel(model, "SAKT", ktData, epochs, batchSize, device, numConcepts) {
        case (conceptIds, responses) => model.predict(conceptIds, responses, conceptIds)
      }
    }

    // ---- SKVMN ----
    reports += runKTReport("SKVMN", device, epochs, batchSize) {
      val model = new SKVMN(numConcepts, embedDim = 64, memSize = 20, dropout = 0.2f, device = device)
      trainKTSkvmn(model, ktData, epochs, batchSize, device, numConcepts)
    }

    // ---- StableKT ----
    reports += runKTReport("StableKT", device, epochs, batchSize) {
      val model = new StableKT(numConcepts, embedDim = 64, numHeads = 8, numBlocks = 2, dropout = 0.2f, device = device)
      trainKTModel(model, "StableKT", ktData, epochs, batchSize, device, numConcepts) {
        case (conceptIds, responses) => model.predict(conceptIds, responses)
      }
    }

    // ---- UKT ----
    reports += runKTReport("UKT", device, epochs, batchSize) {
      val model = new UKT(numConcepts, embedDim = 64, numHeads = 8, numBlocks = 2, dropout = 0.2f, device = device)
      trainKTModel(model, "UKT", ktData, epochs, batchSize, device, numConcepts) {
        case (conceptIds, responses) => model.predict(conceptIds, responses)
      }
    }

    println("\n========== Knowledge Tracing Summary ==========")
    reports.foreach { r =>
      val status = if (r.ok) "PASS" else "FAIL"
      if (r.ok) {
        println(f"[$status] ${r.name}%20s -> TrainLoss: ${r.trainLoss}%.4f, TrainAUC: ${r.trainAuc}%.4f")
      } else {
        println(f"[$status] ${r.name}%20s -> ${r.note}")
      }
    }
  }

  // -------------------------------------------------------------------------
  // Data preparation
  // -------------------------------------------------------------------------

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

  /**
   * Builds KT-style sequence data from fraud data.
   * Treats each row as a concept interaction, with response=1 if label matches prediction (label==1),
   * and uses the sequence of feature bins as concept sequence.
   */
   private def buildKTSequenceData(prepared: FraudPrepared, device: String): (KTDataset, KTDataset, KTDataset) = {
     def buildKTDataset(rows: Vector[EncodedRow]): KTDataset = {
       val n = rows.size
       val seqLen = prepared.seqLen

       // concept_seqs: bin IDs as concept IDs (batch, seq_len)
       val conceptSeqData = rows.flatMap(_.tokens.map(_.toFloat)).toArray
       val conceptSeqs = tensor(conceptSeqData, Array(n.toLong, seqLen.toLong)).toType(ScalarType.Long)

       // question_seqs: same as concept_seqs (each feature is both question and concept)
       val questionSeqs = conceptSeqs.clone()

       // response_seqs: response per interaction (simulate response based on label)
       // For each position t: response is based on whether this would be "correct" given label
       // We treat label=1 as "positive interaction", so all responses in sequence reflect the label
       val responseSeqData = rows.flatMap { row =>
         Array.fill(seqLen)(if (row.label > 0.5f) 1.0f else 0.0f)
       }.toArray
       val responseSeqs = tensor(responseSeqData, Array(n.toLong, seqLen.toLong)).toType(ScalarType.Long)

       // mask_seqs: all valid (batch, seq_len)
       val maskData = Array.fill(n * seqLen)(1.0f)
       val maskSeqs = tensor(maskData, Array(n.toLong, seqLen.toLong)).toType(ScalarType.Long)

       // label_seqs: shifted responses as targets (same as response_seqs for simplicity)
       val labelSeqs = responseSeqs.clone().toType(ScalarType.Float)

       new KTDataset(
         conceptSeqs = conceptSeqs,
         questionSeqs = questionSeqs,
         responseSeqs = responseSeqs,
         maskSeqs = maskSeqs,
         labelSeqs = labelSeqs
       )
     }

     (buildKTDataset(prepared.train), buildKTDataset(prepared.valid), buildKTDataset(prepared.test))
   }

  // -------------------------------------------------------------------------
  // Training helpers
  // -------------------------------------------------------------------------

  private def runKTReport(name: String, device: String, epochs: Int, batchSize: Int)(
    fn: => (Float, Float)
  ): ModelReport = {
    try {
      if (device == "cuda") {
        torch.emptyCache()
        System.gc()
      }
      val (trainLoss, trainAuc) = fn
      if (device == "cuda") {
        torch.emptyCache()
        System.gc()
      }
      ModelReport(name, ok = true, trainLoss, trainAuc, "")
    } catch {
      case e: Throwable =>
        if (device == "cuda") {
          torch.emptyCache()
          System.gc()
        }
        ModelReport(name, ok = false, 0f, 0f, Option(e.getMessage).getOrElse(e.getClass.getSimpleName))
    }
  }

  /**
   * Generic KT trainer for models with (conceptIds, responses) => predictions signature.
   * DKT, ATDKT, AKT, CSKT, DIMKT, GKT, IEKT, MTKT, PromptKT, RKT, RobustKT, SAKT, StableKT, UKT
   */
  private def trainKTModel(
    model: Module,
    name: String,
    ktData: (KTDataset, KTDataset, KTDataset),
    epochs: Int,
    batchSize: Int,
    device: String,
    numConcepts: Long
  )(
    predictFn: (Tensor, Tensor) => Tensor
  ): (Float, Float) = {
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()

    val trainLoader = new DataLoader(ktData._1, batchSize = batchSize, shuffle = true, device = device)
    val numTrain = ktData._1.size.toInt

    var e = 0
    var finalLoss = 0f
    var finalAuc = 0f

    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()

      val it = trainLoader.iterator
      while (it.hasNext) {
        val b = it.next()
        val conceptSeq = b.sequenceFeatures.getOrElse("concept_seq",
          b.sparseFeatures.getOrElse("concepts", throw new RuntimeException("Missing concept seq")))
        val responseSeq = b.sequenceFeatures.getOrElse("response_seq",
          b.sparseFeatures.getOrElse("responses", throw new RuntimeException("Missing response seq")))
        val labels = b.labels.getOrElse(throw new RuntimeException("Missing labels"))

        val bs = conceptSeq.size(0).toInt
        optimizer.zero_grad()

        val preds = predictFn(conceptSeq, responseSeq)  // (batch, seqLen)

        // Use last timestep prediction
        val lastPred = preds.select(1, preds.size(1) - 1).view(bs, 1)
        val y = labels.view(bs, 1).toType(ScalarType.Float)

        val loss = lossFn.apply(lastPred, y)
        loss.backward()
        optimizer.step()

        totalLoss += loss.item().toDouble
        numBatches += 1

        val predArr = lastPred.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val labelArr = y.contiguous().cpu().toFloatArray
        epochPreds.appendAll(predArr)
        epochLabels.appendAll(labelArr)
      }

      val avgLoss = (totalLoss / numBatches).toFloat
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute().toFloat
      aucMetric.reset()

      println(f"[$name] Epoch ${e + 1}/$epochs - TrainLoss: $avgLoss%.4f - TrainAUC: $trainAuc%.4f")

      if (e == epochs - 1) {
        finalLoss = avgLoss
        finalAuc = trainAuc
      }

      if (device == "cuda") torch.emptyCache()
      e += 1
    }

    (finalLoss, finalAuc)
  }

  /**
   * Trainer for ATKT (uses answerIds 0/1/2 where 2=padding).
   */
  private def trainKTAtkt(
    model: ATKT,
    ktData: (KTDataset, KTDataset, KTDataset),
    epochs: Int,
    batchSize: Int,
    device: String,
    numConcepts: Long
  ): (Float, Float) = {
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()

    val trainLoader = new DataLoader(ktData._1, batchSize = batchSize, shuffle = true, device = device)

    var e = 0
    var finalLoss = 0f
    var finalAuc = 0f

    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()

      val it = trainLoader.iterator
      while (it.hasNext) {
        val b = it.next()
        val conceptSeq = b.sequenceFeatures.getOrElse("concept_seq",
          b.sparseFeatures.getOrElse("concepts", throw new RuntimeException("Missing concept seq")))
        val responseSeq = b.sequenceFeatures.getOrElse("response_seq",
          b.sparseFeatures.getOrElse("responses", throw new RuntimeException("Missing response seq")))
        val labels = b.labels.getOrElse(throw new RuntimeException("Missing labels"))

        val bs = conceptSeq.size(0).toInt
        optimizer.zero_grad()

        val preds = model.predict(conceptSeq, responseSeq)
        val lastPred = preds.select(1, preds.size(1) - 1).view(bs, 1)
        val y = labels.view(bs, 1).toType(ScalarType.Float)

        val loss = lossFn.apply(lastPred, y)
        loss.backward()
        optimizer.step()

        totalLoss += loss.item().toDouble
        numBatches += 1

        val predArr = lastPred.toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val labelArr = y.contiguous().cpu().toFloatArray
        epochPreds.appendAll(predArr)
        epochLabels.appendAll(labelArr)
      }

      val avgLoss = (totalLoss / numBatches).toFloat
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute().toFloat
      aucMetric.reset()

      println(f"[ATKT] Epoch ${e + 1}/$epochs - TrainLoss: $avgLoss%.4f - TrainAUC: $trainAuc%.4f")

      if (e == epochs - 1) {
        finalLoss = avgLoss
        finalAuc = trainAuc
      }

      if (device == "cuda") torch.emptyCache()
      e += 1
    }

    (finalLoss, finalAuc)
  }

  /**
   * Trainer for DKVMN (requires question IDs and answer interactions).
   */
  private def trainKTDkvmn(
    model: DKVMN,
    ktData: (KTDataset, KTDataset, KTDataset),
    epochs: Int,
    batchSize: Int,
    device: String,
    numConcepts: Long,
    numQuestions: Long
  ): (Float, Float) = {
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()

    val trainLoader = new DataLoader(ktData._1, batchSize = batchSize, shuffle = true, device = device)

    var e = 0
    var finalLoss = 0f
    var finalAuc = 0f

    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()

      val it = trainLoader.iterator
      while (it.hasNext) {
        val b = it.next()
        val questionSeq = b.sequenceFeatures.getOrElse("question_seq",
          b.sparseFeatures.getOrElse("questions", throw new RuntimeException("Missing question seq")))
        val conceptSeq = b.sequenceFeatures.getOrElse("concept_seq",
          b.sparseFeatures.getOrElse("concepts", throw new RuntimeException("Missing concept seq")))
        val responseSeq = b.sequenceFeatures.getOrElse("response_seq",
          b.sparseFeatures.getOrElse("responses", throw new RuntimeException("Missing response seq")))
        val labels = b.labels.getOrElse(throw new RuntimeException("Missing labels"))

        val bs = conceptSeq.size(0).toInt
        optimizer.zero_grad()

        val preds = model.forward(conceptSeq, responseSeq)
        val lastPred = preds.select(1, preds.size(1) - 1).view(bs, 1)
        val y = labels.view(bs, 1).toType(ScalarType.Float)

        val loss = lossFn.apply(lastPred, y)
        loss.backward()
        optimizer.step()

        totalLoss += loss.item().toDouble
        numBatches += 1

        val predArr = lastPred.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val labelArr = y.contiguous().cpu().toFloatArray
        epochPreds.appendAll(predArr)
        epochLabels.appendAll(labelArr)
      }

      val avgLoss = (totalLoss / numBatches).toFloat
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute().toFloat
      aucMetric.reset()

      println(f"[DKVMN] Epoch ${e + 1}/$epochs - TrainLoss: $avgLoss%.4f - TrainAUC: $trainAuc%.4f")

      if (e == epochs - 1) {
        finalLoss = avgLoss
        finalAuc = trainAuc
      }

      if (device == "cuda") torch.emptyCache()
      e += 1
    }

    (finalLoss, finalAuc)
  }

  /**
   * Trainer for LPKT (requires exercise IDs and concept IDs with time seqs).
   */
  private def trainKTLpkt(
    model: LPKT,
    ktData: (KTDataset, KTDataset, KTDataset),
    epochs: Int,
    batchSize: Int,
    device: String,
    numConcepts: Long
  ): (Float, Float) = {
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()

    val trainLoader = new DataLoader(ktData._1, batchSize = batchSize, shuffle = true, device = device)

    var e = 0
    var finalLoss = 0f
    var finalAuc = 0f

    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()

      val it = trainLoader.iterator
      while (it.hasNext) {
        val b = it.next()
        val conceptSeq = b.sequenceFeatures.getOrElse("concept_seq",
          b.sparseFeatures.getOrElse("concepts", throw new RuntimeException("Missing concept seq")))
        val responseSeq = b.sequenceFeatures.getOrElse("response_seq",
          b.sparseFeatures.getOrElse("responses", throw new RuntimeException("Missing response seq")))
        val labels = b.labels.getOrElse(throw new RuntimeException("Missing labels"))

        val bs = conceptSeq.size(0).toInt
        optimizer.zero_grad()

        val timeSeq = b.timeDiffs.getOrElse {
          // Create dummy time seq (zeros)
          torch.zeros(Array(bs.toLong, conceptSeq.size(1).toLong),
            new TensorOptions().dtype(new org.bytedeco.pytorch.ScalarTypeOptional(ScalarType.Float)))
        }

        val preds = model.forward(conceptSeq, conceptSeq, conceptSeq)
        val lastPred = preds.select(1, preds.size(1) - 1).view(bs, 1)
        val y = labels.view(bs, 1).toType(ScalarType.Float)

        val loss = lossFn.apply(lastPred, y)
        loss.backward()
        optimizer.step()

        totalLoss += loss.item().toDouble
        numBatches += 1

        val predArr = lastPred.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val labelArr = y.contiguous().cpu().toFloatArray
        epochPreds.appendAll(predArr)
        epochLabels.appendAll(labelArr)
      }

      val avgLoss = (totalLoss / numBatches).toFloat
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute().toFloat
      aucMetric.reset()

      println(f"[LPKT] Epoch ${e + 1}/$epochs - TrainLoss: $avgLoss%.4f - TrainAUC: $trainAuc%.4f")

      if (e == epochs - 1) {
        finalLoss = avgLoss
        finalAuc = trainAuc
      }

      if (device == "cuda") torch.emptyCache()
      e += 1
    }

    (finalLoss, finalAuc)
  }

  /**
   * Trainer for QDKT (requires question IDs and concept IDs).
   */
  private def trainKTQdkt(
    model: QDKT,
    ktData: (KTDataset, KTDataset, KTDataset),
    epochs: Int,
    batchSize: Int,
    device: String,
    numConcepts: Long,
    numQuestions: Long
  ): (Float, Float) = {
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()

    val trainLoader = new DataLoader(ktData._1, batchSize = batchSize, shuffle = true, device = device)

    var e = 0
    var finalLoss = 0f
    var finalAuc = 0f

    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()

      val it = trainLoader.iterator
      while (it.hasNext) {
        val b = it.next()
        val questionSeq = b.sequenceFeatures.getOrElse("question_seq",
          b.sparseFeatures.getOrElse("questions", throw new RuntimeException("Missing question seq")))
        val conceptSeq = b.sequenceFeatures.getOrElse("concept_seq",
          b.sparseFeatures.getOrElse("concepts", throw new RuntimeException("Missing concept seq")))
        val responseSeq = b.sequenceFeatures.getOrElse("response_seq",
          b.sparseFeatures.getOrElse("responses", throw new RuntimeException("Missing response seq")))
        val labels = b.labels.getOrElse(throw new RuntimeException("Missing labels"))

        val bs = conceptSeq.size(0).toInt
        optimizer.zero_grad()

        val preds = model.forward(questionSeq, conceptSeq, responseSeq)
        val lastPred = preds.select(1, preds.size(1) - 1).view(bs, 1)
        val y = labels.view(bs, 1).toType(ScalarType.Float)

        val loss = lossFn.apply(lastPred, y)
        loss.backward()
        optimizer.step()

        totalLoss += loss.item().toDouble
        numBatches += 1

        val predArr = lastPred.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val labelArr = y.contiguous().cpu().toFloatArray
        epochPreds.appendAll(predArr)
        epochLabels.appendAll(labelArr)
      }

      val avgLoss = (totalLoss / numBatches).toFloat
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute().toFloat
      aucMetric.reset()

      println(f"[QDKT] Epoch ${e + 1}/$epochs - TrainLoss: $avgLoss%.4f - TrainAUC: $trainAuc%.4f")

      if (e == epochs - 1) {
        finalLoss = avgLoss
        finalAuc = trainAuc
      }

      if (device == "cuda") torch.emptyCache()
      e += 1
    }

    (finalLoss, finalAuc)
  }

  /**
   * Trainer for SAINT and SAINTPlusPlus (exercise IDs, category IDs, response IDs).
   */
  private def trainKTSaint(
    model: Module,
    ktData: (KTDataset, KTDataset, KTDataset),
    epochs: Int,
    batchSize: Int,
    device: String,
    numConcepts: Long
  ): (Float, Float) = {
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()

    val saint = model.asInstanceOf[SAINT]
    val trainLoader = new DataLoader(ktData._1, batchSize = batchSize, shuffle = true, device = device)

    var e = 0
    var finalLoss = 0f
    var finalAuc = 0f

    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()

      val it = trainLoader.iterator
      while (it.hasNext) {
        val b = it.next()
        val exerciseSeq = b.sequenceFeatures.getOrElse("question_seq",
          b.sparseFeatures.getOrElse("questions", throw new RuntimeException("Missing exercise seq")))
        val conceptSeq = b.sequenceFeatures.getOrElse("concept_seq",
          b.sparseFeatures.getOrElse("concepts", throw new RuntimeException("Missing concept seq")))
        val responseSeq = b.sequenceFeatures.getOrElse("response_seq",
          b.sparseFeatures.getOrElse("responses", throw new RuntimeException("Missing response seq")))
        val labels = b.labels.getOrElse(throw new RuntimeException("Missing labels"))

        val bs = conceptSeq.size(0).toInt
        optimizer.zero_grad()

        val preds = saint.forward(exerciseSeq, conceptSeq, responseSeq)
        val lastPred = preds.select(1, preds.size(1) - 1).view(bs, 1)
        val y = labels.view(bs, 1).toType(ScalarType.Float)

        val loss = lossFn.apply(lastPred, y)
        loss.backward()
        optimizer.step()

        totalLoss += loss.item().toDouble
        numBatches += 1

        val predArr = lastPred.toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val labelArr = y.contiguous().cpu().toFloatArray
        epochPreds.appendAll(predArr)
        epochLabels.appendAll(labelArr)
      }

      val avgLoss = (totalLoss / numBatches).toFloat
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute().toFloat
      aucMetric.reset()

      println(f"[SAINT] Epoch ${e + 1}/$epochs - TrainLoss: $avgLoss%.4f - TrainAUC: $trainAuc%.4f")

      if (e == epochs - 1) {
        finalLoss = avgLoss
        finalAuc = trainAuc
      }

      if (device == "cuda") torch.emptyCache()
      e += 1
    }

    (finalLoss, finalAuc)
  }

  /**
   * Trainer for SKVMN.
   */
  private def trainKTSkvmn(
    model: SKVMN,
    ktData: (KTDataset, KTDataset, KTDataset),
    epochs: Int,
    batchSize: Int,
    device: String,
    numConcepts: Long
  ): (Float, Float) = {
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    val aucMetric = new AUC()

    val trainLoader = new DataLoader(ktData._1, batchSize = batchSize, shuffle = true, device = device)

    var e = 0
    var finalLoss = 0f
    var finalAuc = 0f

    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val epochPreds = mutable.ArrayBuffer[Float]()
      val epochLabels = mutable.ArrayBuffer[Float]()

      val it = trainLoader.iterator
      while (it.hasNext) {
        val b = it.next()
        val conceptSeq = b.sequenceFeatures.getOrElse("concept_seq",
          b.sparseFeatures.getOrElse("concepts", throw new RuntimeException("Missing concept seq")))
        val responseSeq = b.sequenceFeatures.getOrElse("response_seq",
          b.sparseFeatures.getOrElse("responses", throw new RuntimeException("Missing response seq")))
        val labels = b.labels.getOrElse(throw new RuntimeException("Missing labels"))

        val bs = conceptSeq.size(0).toInt
        optimizer.zero_grad()

        val preds = model.forward(conceptSeq, responseSeq)
        val lastPred = preds.select(1, preds.size(1) - 1).view(bs, 1)
        val y = labels.view(bs, 1).toType(ScalarType.Float)

        val loss = lossFn.apply(lastPred, y)
        loss.backward()
        optimizer.step()

        totalLoss += loss.item().toDouble
        numBatches += 1

        val predArr = lastPred.sigmoid().toType(ScalarType.Float).contiguous().cpu().toFloatArray
        val labelArr = y.contiguous().cpu().toFloatArray
        epochPreds.appendAll(predArr)
        epochLabels.appendAll(labelArr)
      }

      val avgLoss = (totalLoss / numBatches).toFloat
      aucMetric.update(epochPreds.toArray, epochLabels.toArray)
      val trainAuc = aucMetric.compute().toFloat
      aucMetric.reset()

      println(f"[SKVMN] Epoch ${e + 1}/$epochs - TrainLoss: $avgLoss%.4f - TrainAUC: $trainAuc%.4f")

      if (e == epochs - 1) {
        finalLoss = avgLoss
        finalAuc = trainAuc
      }

      if (device == "cuda") torch.emptyCache()
      e += 1
    }

    (finalLoss, finalAuc)
  }
}
