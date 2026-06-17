package torchrec.basic

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.math._

/**
 * Evaluation metrics for recommender systems.
 *
 * Accuracy metrics: AUC, GAUC, LogLoss
 * Top-K metrics: NDCG, MRR, Recall, Hit, Precision
 * Beyond-accuracy metrics: Diversity, Coverage, Novelty
 */
object Metrics {

  /**
   * Compute AUC score.
   */
  def aucScore(yTrue: Array[Float], yPred: Array[Float]): Float = {
    require(yTrue.length == yPred.length, "y_true and y_pred must have same length")

    val n = yTrue.length
    if (n == 0) return 0.0f

    // Count pairs where yTrue[i] > yTrue[j]
    var posCount = 0L
    var negCount = 0L
    var tieCount = 0L

    for (i <- 0 until n; j <- i + 1 until n) {
      val ti = yTrue(i)
      val tj = yTrue(j)
      val pi = yPred(i)
      val pj = yPred(j)

      if (ti > tj) {
        if (pi > pj) posCount += 1
        else if (pi < pj) negCount += 1
        else tieCount += 1
      } else if (ti < tj) {
        if (pi < pj) posCount += 1
        else if (pi > pj) negCount += 1
        else tieCount += 1
      }
    }

    if (posCount + negCount == 0) return 0.5f
    (posCount + 0.5f * tieCount) / (posCount + negCount).toFloat
  }

  /**
   * Divide predictions by user id.
   */
  private def getUserPred(
    yTrue: Array[Float],
    yPred: Array[Float],
    users: Array[Int]
  ): mutable.Map[Int, UserPred] = {
    val userPred = mutable.Map[Int, UserPred]()
    for (i <- yTrue.indices) {
      val u = users(i)
      userPred.get(u) match {
        case Some(pred) =>
          pred.yTrue += yTrue(i)
          pred.yPred += yPred(i)
        case None =>
          userPred(u) = UserPred(ListBuffer(yTrue(i)), ListBuffer(yPred(i)))
      }
    }
    userPred
  }

  /**
   * Compute Group-AUC (GAUC).
   */
  def gaucScore(
    yTrue: Array[Float],
    yPred: Array[Float],
    users: Array[Int],
    weights: Option[Map[Int, Float]] = None
  ): Float = {
    require(yTrue.length == yPred.length && yTrue.length == users.length,
      "y_true, y_pred, and users must have same length")

    val userPred = getUserPred(yTrue, yPred, users)
    var score = 0.0f
    var num = 0.0f

    userPred.foreach { case (u, pred) =>
      val auc = aucScore(pred.yTrue.toArray, pred.yPred.toArray)
      val userWeight = weights match {
        case Some(w) => w.getOrElse(u, pred.yTrue.length.toFloat)
        case None => pred.yTrue.length.toFloat
      }
      score += auc * userWeight
      num += userWeight
    }

    if (num == 0) 0.0f else score / num
  }

  /**
   * Compute NDCG score.
   */
  def ndcgScore(yTrue: Map[String, List[Int]], yPred: Map[String, List[Int]], topKs: Seq[Int] = Seq(5)): Map[String, Float] = {
    val result = topkMetrics(yTrue, yPred, topKs)
    topKs.map(k => s"NDCG@$k" -> result(s"NDCG@$k").toFloat).toMap
  }

  def hitScore(yTrue: Map[String, List[Int]], yPred: Map[String, List[Int]], topKs: Seq[Int] = Seq(5)): Map[String, Float] = {
    val result = topkMetrics(yTrue, yPred, topKs)
    topKs.map(k => s"Hits@$k" -> result(s"Hits@$k").toFloat).toMap
  }

  def mrrScore(yTrue: Map[String, List[Int]], yPred: Map[String, List[Int]], topKs: Seq[Int] = Seq(5)): Map[String, Float] = {
    val result = topkMetrics(yTrue, yPred, topKs)
    topKs.map(k => s"MRR@$k" -> result(s"MRR@$k").toFloat).toMap
  }

  def recallScore(yTrue: Map[String, List[Int]], yPred: Map[String, List[Int]], topKs: Seq[Int] = Seq(5)): Map[String, Float] = {
    val result = topkMetrics(yTrue, yPred, topKs)
    topKs.map(k => s"Recall@$k" -> result(s"Recall@$k").toFloat).toMap
  }

  def precisionScore(yTrue: Map[String, List[Int]], yPred: Map[String, List[Int]], topKs: Seq[Int] = Seq(5)): Map[String, Float] = {
    val result = topkMetrics(yTrue, yPred, topKs)
    topKs.map(k => s"Precision@$k" -> result(s"Precision@$k").toFloat).toMap
  }

  /**
   * Compute top-K metrics: NDCG, MRR, Recall, Hit, Precision.
   */
  def topkMetrics(
    yTrue: Map[String, List[Int]],
    yPred: Map[String, List[Int]],
    topKs: Seq[Int] = Seq(5)
  ): Map[String, String] = {
    require(yTrue.size == yPred.size, "y_true and y_pred must have same size")
    require(topKs.nonEmpty, "topKs must not be empty")

    val results = mutable.Map[String, String]()

    val userIds = yTrue.keys.toSeq
    val numUsers = userIds.length

    for (k <- topKs) {
      var ndcgs = 0.0
      var mrrs = 0.0
      var hits = 0.0
      var precisions = 0.0
      var recalls = 0.0
      var gts = 0L

      for (u <- userIds) {
        val trueItems = yTrue(u)
        val predItems = yPred(u)

        if (trueItems.nonEmpty) {
          var mrrTmp = 0.0
          var mrrFlag = true
          var hitTmp = 0.0
          var dcgTmp = 0.0
          var idcgTmp = 0.0

          for (j <- 0 until k.min(predItems.length)) {
            if (trueItems.contains(predItems(j))) {
              hitTmp += 1.0
              if (mrrFlag) {
                mrrFlag = false
                mrrTmp = 1.0 / (1 + j)
              }
              dcgTmp += 1.0 / (math.log(j + 2) / math.log(2))
            }
            if (j < trueItems.length) {
              idcgTmp += 1.0 / (math.log(j + 2) / math.log(2))
            }
          }

          gts += trueItems.length
          hits += hitTmp
          mrrs += mrrTmp
          recalls += hitTmp / trueItems.length
          precisions += hitTmp / k
          if (idcgTmp != 0) {
            ndcgs += dcgTmp / idcgTmp
          }
        }
      }

      val ndcgVal = roundDecimals(ndcgs / numUsers, 4)
      val mrrVal = roundDecimals(mrrs / numUsers, 4)
      val recallVal = roundDecimals(recalls / numUsers, 4)
      val hitVal = roundDecimals(hits / gts, 4)
      val precisionVal = roundDecimals(precisions / numUsers, 4)

      results(s"NDCG@$k") = f"$ndcgVal%.4f"
      results(s"MRR@$k") = f"$mrrVal%.4f"
      results(s"Recall@$k") = f"$recallVal%.4f"
      results(s"Hits@$k") = f"$hitVal%.4f"
      results(s"Precision@$k") = f"$precisionVal%.4f"
    }

    results.toMap
  }

  /**
   * Compute log loss.
   */
//  def logLoss(yTrue: Array[Float], yPred: Array[Float]): Float = {
//    require(yTrue.length == yPred.length, "y_true and y_pred must have same length")
//    var score = 0.0f
//    for (i <- yTrue.indices) {
//      val yt = yTrue(i)
//      val yp = yPred(i).max(1e-10f).min(1.0f - 1e-10f)
//      score += (yt * log(yp) + (1.0f - yt) * log(1.0f - yp)).toFloat
//    }
//    -score / yTrue.length
//  }

  import math.log

  import scala.math.log

  def logLoss(yTrue: Array[Float], yPred: Array[Float]): Float = {
    require(yTrue.length == yPred.length, "y_true and y_pred must have same length")
    val eps = 1e-10f
    var total = 0.0
    val count = yTrue.length
    for (i <- yTrue.indices) {
      val yt = yTrue(i)
      val yp = yPred(i)
      // 裁剪防止log(0)
      val p = math.max(eps, math.min(1.0f - eps, yp))
      val lossItem = yt * log(p) + (1.0 - yt) * log(1.0 - p)
      total += lossItem
    }
    // 标准公式：-1/N * sum
    (-total / count / 2).toFloat
  }

  /**
   * Compute Intra-List Diversity (ILD): average pairwise cosine distance within each user's recommendation list.
   */
  def diversityScore(
    yPred: Map[String, List[Int]],
    itemEmbeddings: Map[Int, Array[Float]],
    topKs: Seq[Int] = Seq(5)
  ): Map[String, String] = {
    require(topKs.nonEmpty, "topKs must not be empty")

    val results = mutable.Map[String, String]()

    for (k <- topKs) {
      val userDiversities = ListBuffer[Double]()

      yPred.foreach { case (_, items) =>
        val selectedItems = items.take(k)
        if (selectedItems.length >= 2) {
          // Collect embeddings
          val embs = ListBuffer[Array[Float]]()
          for (item <- selectedItems) {
            itemEmbeddings.get(item).foreach { emb =>
              embs += emb
            }
          }

          if (embs.length >= 2) {
            val embArray = embs.toArray
            val n = embArray.length

            // Compute norms
            val norms = embArray.map { emb =>
              math.sqrt(emb.foldLeft(0.0)((sum, v) => sum + v.toDouble * v.toDouble)).max(1e-10)
            }

            // Normalize
            val normed = embArray.zip(norms).map { case (emb, norm) =>
              emb.map(v => (v / norm).toFloat)
            }

            // Compute cosine similarity matrix
            var distSum = 0.0
            var pairCount = 0
            for (i <- 0 until n; j <- i + 1 until n) {
              var sim = 0.0
              for (d <- 0 until normed(0).length) {
                sim += normed(i)(d) * normed(j)(d)
              }
              distSum += 1.0 - sim
              pairCount += 1
            }

            if (pairCount > 0) {
              userDiversities += distSum / pairCount
            }
          }
        }
      }

      val score = if (userDiversities.nonEmpty) {
        roundDecimals(userDiversities.sum / userDiversities.length, 4)
      } else {
        0.0
      }
      results(s"Diversity@$k") = f"$score%.4f"
    }

    results.toMap
  }

  /**
   * Compute Catalog Coverage: fraction of all items that appear in at least one user's recommendation list.
   */
  def coverageScore(
    yPred: Map[String, List[Int]],
    allItems: Set[Int],
    topKs: Seq[Int] = Seq(5)
  ): Map[String, String] = {
    require(topKs.nonEmpty, "topKs must not be empty")
    require(allItems.nonEmpty, "allItems must not be empty")

    val results = mutable.Map[String, String]()

    for (k <- topKs) {
      val recItems = mutable.Set[Int]()
      yPred.foreach { case (_, items) =>
        recItems ++= items.take(k)
      }
      val score = roundDecimals(recItems.size.toDouble / allItems.size, 4)
      results(s"Coverage@$k") = f"$score%.4f"
    }

    results.toMap
  }

  /**
   * Compute Mean Self-Information: measures how "surprising" or niche the recommendations are.
   */
  def noveltyScore(
    yPred: Map[String, List[Int]],
    itemPopularity: Map[Int, Float],
    topKs: Seq[Int] = Seq(5)
  ): Map[String, String] = {
    require(topKs.nonEmpty, "topKs must not be empty")

    val results = mutable.Map[String, String]()

    for (k <- topKs) {
      val userNovelties = ListBuffer[Double]()

      yPred.foreach { case (_, items) =>
        val selectedItems = items.take(k)
        if (selectedItems.nonEmpty) {
          var selfInfoSum = 0.0
          for (item <- selectedItems) {
            val pop = itemPopularity.getOrElse(item, 1e-10f).max(1e-10f)
            selfInfoSum += -(math.log(pop.toDouble) / math.log(2))
          }
          userNovelties += selfInfoSum / selectedItems.length
        }
      }

      val score = if (userNovelties.nonEmpty) {
        roundDecimals(userNovelties.sum / userNovelties.length, 2)
      } else {
        0.0
      }
      results(s"Novelty@$k") = f"$score%.2f"
    }

    results.toMap
  }

  private case class UserPred(yTrue: ListBuffer[Float], yPred: ListBuffer[Float])

  private def roundDecimals(x: Double, decimals: Int): Double = {
    val factor = math.pow(10, decimals)
    math.round(x * factor) / factor
  }
}