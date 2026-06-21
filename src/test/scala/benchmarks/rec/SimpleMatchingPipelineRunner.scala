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

import scala.collection.mutable
import scala.io.Source
import scala.util.Random

/**
 * Simplified pipeline runner for generating model training metrics
 */
object SimpleMatchingPipelineRunner {

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

  case class ModelMetrics(name: String, auc: Float, loss: Float, time_ms: Long)

  def main(args: Array[String]): Unit = {
    println("\n" + "=" * 70)
    println("TORCH-RECHUB SCALA: MODEL TRAINING METRICS GENERATOR")
    println("=" * 70)

    val csvPath = "/home/muller/IdeaProjects/torch-rechub-scala/src/main/resources/fraud_data.csv"
    val maxRows = 2000  // Even smaller for quick testing
    val batchSize = 128
    val epochs = 2
    val device = "cpu"  // Use CPU for faster local testing
    val seed = 2026
    val bins = 128
    val seqLen = 20

    try {
      val prepared = prepareFraud(csvPath, maxRows, bins, seqLen, seed, itemEmbedDim = 8)
      val (trainDs, validDs, testDs) = buildMatchingDatasets(prepared, device)
      val metricsBuffer = mutable.ArrayBuffer[ModelMetrics]()

      println(s"\nDataset prepared:")
      println(s"  Train size: ${prepared.train.size}")
      println(s"  Valid size: ${prepared.valid.size}")
      println(s"  Test size: ${prepared.test.size}")
      println(s"  Device: $device")
      println(s"  Batch Size: $batchSize")
      println(s"  Epochs: $epochs")
      println("\n" + "=" * 70)
      println("STARTING MODEL TRAINING")
      println("=" * 70 + "\n")

      // Train DSSM as example
      val startTime = System.currentTimeMillis()
      try {
        if (device == "cuda") { torch.emptyCache(); System.gc() }
        val userFeatures = List(SparseFeature("user_id", prepared.userVocab + 2, 8))
        val itemFeatures = List(SparseFeature("item_id", prepared.itemVocab + 2, 8))
        val model = new DSSM(userFeatures, itemFeatures, embedDim = 8, towerDims = List(256L, 128L), dropout = 0.2f, device)

        println("[DSSM] Starting training...")
        trainDssm(model, trainDs, epochs, batchSize, device)
        val metrics = evalDssm(model, testDs, batchSize, device)
        val elapsed = System.currentTimeMillis() - startTime

        val auc = metrics.getOrElse("AUC", 0.5f)
        metricsBuffer += ModelMetrics("DSSM", auc, 0.0f, elapsed)
        println(s"[DSSM] Completed in ${elapsed}ms with AUC=${auc}")
      } catch {
        case e: Exception =>
          println(s"[DSSM] FAILED: ${e.getMessage}")
          e.printStackTrace()
      }

      // Print final summary
      println("\n" + "=" * 70)
      println("TRAINING SUMMARY")
      println("=" * 70)
      metricsBuffer.foreach { m =>
        println(f"${m.name}%-15s | AUC: ${m.auc}%.4f | Time: ${m.time_ms}ms")
      }
      println("=" * 70)

    } catch {
      case e: Exception =>
        println(s"ERROR: ${e.getMessage}")
        e.printStackTrace()
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
      val labels = tensor(rows.map(_.label).toArray, Array(rows.size.toLong))

      // Create SequenceDataset with features map including both user and item ids
      new SequenceDataset(
        features = Map("user_id" -> userIds, "item_id" -> itemIds),
        labels = Some(labels),
        itemFeatures = Some(Map("item_id" -> itemIds)) // Explicitly set itemFeatures
      )
    }
    (mk(prepared.train), mk(prepared.valid), mk(prepared.test))
  }

  private def trainDssm(model: DSSM, train: SequenceDataset, epochs: Int, batchSize: Int, device: String): Unit = {
    val loader = new DataLoader(train, batchSize = batchSize, shuffle = true, device = device)
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(1e-3))
    var e = 0
    while (e < epochs) {
      var totalLoss = 0.0
      var numBatches = 0
      val it = loader.iterator
      while (it.hasNext) {
        val b = it.next()

        // Debug: print available features
        if (numBatches == 0) {
          println(s"[DEBUG trainDssm] Batch sparseFeatures keys: ${b.sparseFeatures.keys.mkString(", ")}")
          println(s"[DEBUG trainDssm] Batch itemFeatures is Map: ${b.itemFeatures.keys.mkString(", ")}")
        }

        val userFeats = b.sparseFeatures.filter(kv => kv._1.contains("user"))
        val itemFeats = if (b.itemFeatures.nonEmpty) b.itemFeatures else Map.empty[String, Tensor]

        if (userFeats.nonEmpty && itemFeats.nonEmpty) {
          try {
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
            }
          } catch {
            case e: Exception =>
              System.err.println(s"[ERROR] Error processing batch: ${e.getMessage}")
              e.printStackTrace()
          }
        } else {
          System.err.println(s"[WARNING] Batch has empty features - userFeats: ${userFeats.nonEmpty}, itemFeats: ${itemFeats.nonEmpty}")
        }
      }
      val avgLoss = if (numBatches > 0) totalLoss / numBatches else 0.0
      println(f"[DSSM] Epoch ${e + 1}/$epochs - TrainLoss: $avgLoss%.4f")
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
      val userFeats = b.sparseFeatures
      val itemFeats = b.itemFeatures
      if (userFeats.nonEmpty && itemFeats.nonEmpty) {
        val bs = userFeats.values.head.size(0).toInt
        val userEmb = model.userTowerForward(userFeats)
        val itemEmb = model.itemTowerForward(itemFeats)
        val scores = userEmb.mul(itemEmb).sum(1).sigmoid()
        preds.appendAll(scores.toType(ScalarType.Float).contiguous().cpu().toFloatArray)
        b.labels.foreach { y => labels.appendAll(y.squeeze().toType(ScalarType.Float).contiguous().cpu().toFloatArray) }
      }
    }
    if (preds.isEmpty) return Map("AUC" -> 0.5f)
    val auc = new AUC()
    auc.update(preds.toArray, labels.toArray)
    Map("AUC" -> auc.compute())
  }
}

