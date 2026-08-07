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
import scala.util.Random

/**
 * Data processing utilities for recommendation models.
 */
object DataUtils {

  /**
   * Pad sequences to equal length.
   */
  def padSequences(
    sequences: List[List[Int]],
    maxLen: Option[Int] = None,
    padding: String = "pre",
    truncating: String = "pre",
    value: Int = 0
  ): Array[Array[Int]] = {
    val actualMaxLen = maxLen.getOrElse(sequences.map(_.length).max)
    val result = Array.ofDim[Int](sequences.length, actualMaxLen)

    sequences.zipWithIndex.foreach { case (seq, idx) =>
      val processed = if (seq.length > actualMaxLen) {
        if (truncating == "pre") seq.takeRight(actualMaxLen) else seq.take(actualMaxLen)
      } else {
        seq
      }

      val prePad = if (padding == "pre") actualMaxLen - processed.length else 0

      var j = 0
      while (j < prePad) { result(idx)(j) = value; j += 1 }
      j = 0
      while (j < processed.length) { result(idx)(prePad + j) = processed(j); j += 1 }
      j = prePad + processed.length
      while (j < actualMaxLen) { result(idx)(j) = value; j += 1 }
    }

    result
  }

  /**
   * Generate a negative sample that is not in the history.
   */
  def negSample(clickHist: List[Int], itemSize: Int, rng: Random = new Random()): Int = {
    var neg = rng.nextInt(itemSize) + 1
    while (clickHist.contains(neg)) {
      neg = rng.nextInt(itemSize) + 1
    }
    neg
  }

  /**
   * Random negative sampling.
   */
  def randomNegSample(itemSize: Int, numSamples: Int, rng: Random = new Random()): List[Int] = {
    List.fill(numSamples)(rng.nextInt(itemSize) + 1)
  }

  /**
   * Popularity-based negative sampling.
   */
  def popularityNegSample(
    itemCounts: Map[Int, Int],
    numSamples: Int,
    rng: Random = new Random()
  ): List[Int] = {
    val items = itemCounts.keys.toList
    val counts = items.map(itemCounts)
    val total = counts.map(c => math.log(c + 1) + 1e-6).sum
    val probs = counts.map(c => (math.log(c + 1) + 1e-6) / total)

    List.fill(numSamples) {
      val r = rng.nextDouble()
      var cumsum = 0.0
      var selected = items.head
      var i = 0
      while (i < items.size) {
        cumsum += probs(i)
        if (r <= cumsum) { selected = items(i); i = items.size }
        i += 1
      }
      selected
    }
  }

  /**
   * Calculate auto embedding dimension based on vocabulary size.
   */
  def getAutoEmbeddingDim(numClasses: Int): Int = {
    math.floor(6 * math.pow(numClasses, 0.25)).toInt
  }

  /**
   * Convert label-encoded data to 1-indexed.
   */
  def relabelFeatures(data: mutable.Map[String, Array[Int]]): mutable.Map[String, Array[Int]] = {
    data.map { case (k, arr) => k -> arr.map(_ + 1) }
  }

  /**
   * Build item to attribute mapping.
   */
  def buildItemAttributeMap(
    itemIds: Array[Int],
    attributeIds: Array[Int]
  ): Map[Int, Int] = {
    itemIds.zip(attributeIds).toMap
  }

  /**
   * Tokenize text into character-level tokens.
   */
  def tokenize(text: String): List[String] = {
    text.map(_.toString).toList
  }

  /**
   * Build vocabulary from token list.
   */
  def buildVocabulary(tokens: List[List[String]]): Map[String, Int] = {
    val vocab = mutable.Map[String, Int]()
    var idx = 0
    tokens.flatten.foreach { token =>
      if (!vocab.contains(token)) { vocab(token) = idx; idx += 1 }
    }
    vocab.toMap
  }

  /**
   * Encode tokens using vocabulary.
   */
  def encodeTokens(tokens: List[String], vocab: Map[String, Int]): List[Int] = {
    tokens.map(vocab.getOrElse(_, 0))
  }

  /**
   * Split sequences into train/val/test using sliding window.
   */
  def generateSequenceSplits(
    sequences: List[(Int, List[Int])],
    maxLen: Int = 50,
    minLen: Int = 1
  ): (List[List[Int]], List[List[Int]], List[List[Int]]) = {
    val train = mutable.ListBuffer[List[Int]]()
    val valid = mutable.ListBuffer[List[Int]]()
    val test = mutable.ListBuffer[List[Int]]()

    sequences.foreach { case (_, items) =>
      if (items.length >= minLen) {
        for (i <- 1 until items.length) {
          val hist = items.take(i)
          val paddedHist = hist ++ List.fill(maxLen - hist.length)(0)
          val target = items(i)
          if (i == items.length - 1) test += (0 :: target :: hist)
          else if (i == items.length - 2) valid += (0 :: target :: hist)
          else train += (0 :: target :: hist)
        }
      }
    }

    (train.toList, valid.toList, test.toList)
  }

  /**
   * Collate batch of sequence data.
   */
  def collateSequenceBatch(
    batch: List[(Array[Int], Array[Int])],
    maxLen: Int
  ): (Tensor, Tensor, Tensor) = {
    val batchSize = batch.length
    // Simplified placeholder - in real impl would create tensors
    val emptyInput = torch.zeros(batchSize.toLong, maxLen.toLong)
    val emptyTarget = torch.zeros(batchSize.toLong)
    val emptyMask = torch.ones(batchSize.toLong, maxLen.toLong)
    (emptyInput, emptyMask, emptyTarget)
  }
}

/**
 * Sequence dataset wrapper.
 */
class SequenceDataset(
  val seqTokens: Array[Array[Int]],
  val seqPositions: Array[Array[Int]],
  val targets: Array[Int],
  val seqTimeDiffs: Array[Array[Int]] = Array.empty
) {
  def length: Int = targets.length

  def apply(index: Int): (Tensor, Tensor, Tensor, Tensor) = {
    // Simplified - return empty tensors
    val emptySeq = torch.zeros(1L)
    val emptyPos = torch.zeros(1L)
    val emptyTarget = torch.zeros(1L)
    val emptyTime = torch.zeros(1L)
    (emptySeq, emptyPos, emptyTime, emptyTarget)
  }
}