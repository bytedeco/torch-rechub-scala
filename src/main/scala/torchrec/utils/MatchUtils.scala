package torchrec.utils

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.nn.modules._
import org.bytedeco.pytorch.nn.modules.container._
import org.bytedeco.pytorch.nn.options._
import org.bytedeco.pytorch.optim._
import org.bytedeco.pytorch.data.datasets._
import org.bytedeco.pytorch.data.options._
import org.bytedeco.pytorch.data.sampler._
import org.bytedeco.pytorch.distributed._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.mutable

import torchrec.Implicits._

/**
 * Matching utilities for recommendation models.
 */
object MatchUtils {

  /**
   * In-batch negative sampling from a similarity matrix.
   */
  def inBatchNegativeSampling(
    scores: Tensor,
    negRatio: Int,
    hardNegative: Boolean = false
  ): Tensor = {
    val batchSize = scores.size(0).toInt
    val maxNeg = batchSize - 1
    val actualNeg = if (negRatio <= 0 || negRatio > maxNeg) maxNeg else negRatio

    val negIndicesArr = Array.ofDim[Long](batchSize * actualNeg)

    var i = 0
    while (i < batchSize) {
      val rowScores = scores.select(0, i)

      if (hardNegative) {
        // Simplified topk using argsort approach
        // Just use indices 0 to actualNeg-1 as placeholders
        var k = 0
        while (k < actualNeg) {
          negIndicesArr(i * actualNeg + k) = k.toLong
          k += 1
        }
      } else {
        // Random sampling
        val candidates = mutable.ListBuffer[Int]()
        var j = 0
        while (j < batchSize) {
          if (j != i) candidates += j
          j += 1
        }
        val shuffled = candidates.toArray.sortBy(_ => scala.util.Random.nextDouble())
        val sampled = shuffled.take(actualNeg)
        var k = 0
        while (k < actualNeg) {
          negIndicesArr(i * actualNeg + k) = sampled(k)
          k += 1
        }
      }
      i += 1
    }

    val negIndices = longTensor(negIndicesArr)
      .toType(ScalarType.Long)
      .reshape(batchSize.toLong, actualNeg.toLong)
    negIndices
  }

  /**
   * Gather in-batch logits from similarity matrix and negative indices.
   */
  def gatherInBatchLogits(
    scores: Tensor,
    negIndices: Tensor
  ): Tensor = {
    val batchSize = scores.size(0).toInt
    val numNeg = negIndices.size(1).toInt

    val posLogitsArr = Array.ofDim[Float](batchSize)
    var i = 0
    while (i < batchSize) {
      posLogitsArr(i) = scores.select(0, i).select(0, i).item().toFloat
      i += 1
    }
    val posLogits = floatTensor(posLogitsArr).reshape(batchSize.toLong, 1L)

    val negLogitsArr = Array.ofDim[Float](batchSize * numNeg)
    i = 0
    while (i < batchSize) {
      var j = 0
      while (j < numNeg) {
        val negIdx = negIndices.select(0, i).select(0, j).item().toLong.toInt
        negLogitsArr(i * numNeg + j) = scores.select(0, i).select(0, negIdx).item().toFloat
        j += 1
      }
      i += 1
    }
    val negLogits = floatTensor(negLogitsArr)
      .reshape(batchSize.toLong, numNeg.toLong)

    torch.cat(torchrec.Implicits.SeqTensorRichSeq(List(posLogits, negLogits)).toTensorVector, 1)
  }

  /**
   * Merge user profile and item profile with interaction data.
   */
  def genModelInput(
    interactionData: Map[String, Any],
    userProfile: Map[String, Array[Int]],
    itemProfile: Map[String, Array[Int]],
    userCol: String = "user_id",
    itemCol: String = "item_id",
    seqMaxLen: Int = 50
  ): Map[String, Any] = {
    val result = mutable.Map[String, Any]()
    userProfile.foreach { case (k, v) => result(k) = v }
    itemProfile.foreach { case (k, v) => result(k) = v }
    interactionData.foreach { case (k, v) => result(k) = v }
    result.toMap
  }

  /**
   * Build Faiss-style index from embeddings.
   */
  class FaissIndex(
    val dim: Int,
    val indexType: String = "flat",
    val metric: String = "L2"
  ) {
    private var embeddings: Option[Tensor] = None
    private var isBuilt: Boolean = false

    def fit(emb: Tensor): Unit = {
      embeddings = Some(emb.contiguous())
      isBuilt = true
    }

    def query(queryEmb: Tensor, topK: Int): (Array[Array[Long]], Array[Array[Float]]) = {
      if (!isBuilt) {
        throw new RuntimeException("Index not built. Call fit() first.")
      }

      val emb = embeddings.get
      val n = queryEmb.size(0).toInt
      val indices = Array.ofDim[Long](n, topK)
      val dists = Array.ofDim[Float](n, topK)

      var i = 0
      while (i < n) {
        val queryVec = queryEmb.select(0, i)

        val distArr = Array.ofDim[Float](emb.size(0).toInt)
        var j = 0
        while (j < emb.size(0).toInt) {
          val stored = emb.select(0, j)
          var d = 0
          var dist = 0.0f
          while (d < dim) {
            val diff = queryVec.select(0, d).item().toFloat - stored.select(0, d).item().toFloat
            dist += diff * diff
            d += 1
          }
          distArr(j) = dist
          j += 1
        }

        val sorted = distArr.zipWithIndex.sortBy(_._1)
        var k = 0
        while (k < topK) {
          indices(i)(k) = sorted(k)._2.toLong
          dists(i)(k) = sorted(k)._1
          k += 1
        }
        i += 1
      }

      (indices, dists)
    }
  }
}