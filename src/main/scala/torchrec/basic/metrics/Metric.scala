package torchrec.basic.metrics

import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.math._
import scala.collection.mutable
import scala.util.Sorting

/**
 * Base trait for evaluation metrics
 */
trait Metric {
  def name: String
  def update(predictions: Array[Float], labels: Array[Float]): Unit
  def compute(): Float
  def reset(): Unit
}

/**
 * Area Under the ROC Curve
 */
class AUC(posLabel: Int = 1) extends Metric {
  private var pairs = 0.0
  private var posCount = 0
  private var negCount = 0

  def name: String = "AUC"

  def update(predictions: Array[Float], labels: Array[Float]): Unit = {
    val n = predictions.length
    val sortedIndices = predictions.indices.sortBy(i => -predictions(i))

    // Scan from end: for each positive sample, count how many negatives rank after it
    var negAfter = 0
    var batchPos = 0
    var batchNeg = 0
    var i = n - 1
    while (i >= 0) {
      val idx = sortedIndices(i)
      if (labels(idx) <= 0.5f) {
        negAfter += 1
        batchNeg += 1
      } else {
        pairs += negAfter
        batchPos += 1
      }
      i -= 1
    }
    posCount += batchPos
    negCount += batchNeg
  }

  def compute(): Float = {
    if (posCount == 0 || negCount == 0) return 0.5f
    (pairs / (posCount * negCount)).toFloat
  }

  def reset(): Unit = {
    pairs = 0.0
    posCount = 0
    negCount = 0
  }
}

object AUC {
  def calculate(predictions: Array[Float], labels: Array[Float]): Float = {
    val metric = new AUC()
    metric.update(predictions, labels)
    metric.compute()
  }
}

/**
 * Log Loss (Binary Cross Entropy)
 */
class LogLoss extends Metric {
  private var totalLoss = 0.0
  private var count = 0

  def name: String = "LogLoss"

  def update(predictions: Array[Float], labels: Array[Float]): Unit = {
    require(predictions.length == labels.length)

    var i = 0
    while (i < predictions.length) {
      // Guard against NaN / Infinite predictions which would make log/pow fail
      val raw = predictions(i)
      val p = if (java.lang.Float.isNaN(raw) || java.lang.Float.isInfinite(raw)) {
        // Replace bad prediction with neutral probability 0.5
        0.5f
      } else {
        math.max(1e-7, math.min(1 - 1e-7, raw)).toFloat
      }
      val y = labels(i)
      totalLoss += -(y * log(p) + (1 - y) * log(1 - p))
      count += 1
      i += 1
    }
  }

  def compute(): Float = if (count == 0) 0.0f else (totalLoss / count).toFloat

  def reset(): Unit = {
    totalLoss = 0.0
    count = 0
  }
}

object LogLoss {
  def calculate(predictions: Array[Float], labels: Array[Float]): Float = {
    val metric = new LogLoss()
    metric.update(predictions, labels)
    metric.compute()
  }
}

/**
 * Accuracy metric
 */
class Accuracy(threshold: Float = 0.5f) extends Metric {
  private var correct = 0
  private var total = 0

  def name: String = "Accuracy"

  def update(predictions: Array[Float], labels: Array[Float]): Unit = {
    require(predictions.length == labels.length)

    var i = 0
    while (i < predictions.length) {
      val pred = if (predictions(i) > threshold) 1.0f else 0.0f
      if (pred == labels(i)) correct += 1
      total += 1
      i += 1
    }
  }

  def compute(): Float = if (total == 0) 0.0f else correct.toFloat / total

  def reset(): Unit = {
    correct = 0
    total = 0
  }
}

/**
 * Top-K Hit Rate
 * For CTR: rank all predictions, check if any positive label is in top-k
 */
class HitRate[K: Numeric](k: Int) extends Metric {
  private var hits = 0
  private var total = 0

  def name: String = s"Hit@$k"

  def update(predictions: Array[Float], labels: Array[Float]): Unit = {
    val n = predictions.length
    if (n == 0) return
    // Sort indices by prediction score (descending), take top-k
    val sortedIndices = predictions.indices.sortBy(i => -predictions(i))
    val topKIndices = sortedIndices.take(math.min(k, n))
    var found = false
    var i = 0
    while (i < topKIndices.length && !found) {
      if (labels(topKIndices(i)) > 0.5f) found = true
      i += 1
    }
    total += 1
    if (found) hits += 1
  }

  def compute(): Float = if (total == 0) 0.0f else hits.toFloat / total

  def reset(): Unit = {
    hits = 0
    total = 0
  }
}

/**
 * NDCG@K (Normalized Discounted Cumulative Gain)
 */
class NDCG[K: Numeric](k: Int) extends Metric {
  private var totalNDCG = 0.0
  private var count = 0

  def name: String = s"NDCG@$k"

  def update(predictions: Array[Float], labels: Array[Float]): Unit = {
    count += 1
    val rankedLabels = labels.zip(predictions).sortBy(-_._2).take(k)
    var dcg = 0.0
    var idx = 0
    while (idx < rankedLabels.length) {
      val rel = rankedLabels(idx)._1
      if (rel > 0) {
        dcg += 1.0 / log(idx + 2) / log(2)
      }
      idx += 1
    }

    val idealRanked = labels.sorted.reverse.take(k)
    var idcg = 0.0
    idx = 0
    while (idx < idealRanked.length) {
      val rel = idealRanked(idx)
      if (rel > 0) {
        idcg += 1.0 / log(idx + 2) / log(2)
      }
      idx += 1
    }

    totalNDCG += (if (idcg > 0) dcg / idcg else 0.0)
  }

  def compute(): Float = (totalNDCG / count).toFloat

  def reset(): Unit = {
    totalNDCG = 0.0
    count = 0
  }
}

/**
 * MRR (Mean Reciprocal Rank)
 */
class MRR extends Metric {
  private var totalRR = 0.0
  private var count = 0

  def name: String = "MRR"

  def update(predictions: Array[Float], labels: Array[Float]): Unit = {
    count += 1
    val ranked = labels.zip(predictions).sortBy(-_._2)
    var found = false
    var i = 0
    while (i < ranked.length && !found) {
      if (ranked(i)._1 > 0) {
        totalRR += 1.0 / (i + 1)
        found = true
      }
      i += 1
    }
  }

  def compute(): Float = (totalRR / count).toFloat

  def reset(): Unit = {
    totalRR = 0.0
    count = 0
  }
}

/**
 * MSE (Mean Squared Error)
 */
class MSE extends Metric {
  private var totalSqError = 0.0
  private var count = 0

  def name: String = "MSE"

  def update(predictions: Array[Float], labels: Array[Float]): Unit = {
    require(predictions.length == labels.length)
    var i = 0
    while (i < predictions.length) {
      val diff = predictions(i) - labels(i)
      totalSqError += diff * diff
      count += 1
      i += 1
    }
  }

  def compute(): Float = (totalSqError / count).toFloat

  def reset(): Unit = {
    totalSqError = 0.0
    count = 0
  }
}

/**
 * RMSE (Root Mean Squared Error)
 */
class RMSE extends MSE {
  override def name: String = "RMSE"
  override def compute(): Float = sqrt(super.compute().toDouble).toFloat
}

/**
 * MAE (Mean Absolute Error)
 */
class MAE extends Metric {
  private var totalAbsError = 0.0
  private var count = 0

  def name: String = "MAE"

  def update(predictions: Array[Float], labels: Array[Float]): Unit = {
    require(predictions.length == labels.length)
    var i = 0
    while (i < predictions.length) {
      totalAbsError += abs(predictions(i) - labels(i))
      count += 1
      i += 1
    }
  }

  def compute(): Float = (totalAbsError / count).toFloat

  def reset(): Unit = {
    totalAbsError = 0.0
    count = 0
  }
}

/**
 * Metric registry for managing multiple metrics
 */
class MetricRegistry {
  private val metrics = mutable.Map[String, Metric]()

  def register(name: String, metric: Metric): this.type = {
    metrics(name) = metric
    this
  }

  def update(predictions: Array[Float], labels: Array[Float]): Unit = {
    metrics.values.foreach(_.update(predictions, labels))
  }

  def compute(): Map[String, Float] = metrics.view.mapValues(_.compute()).toMap

  def reset(): Unit = metrics.values.foreach(_.reset())

  def get(name: String): Option[Metric] = metrics.get(name)

  def names: Set[String] = metrics.keySet.toSet
}
