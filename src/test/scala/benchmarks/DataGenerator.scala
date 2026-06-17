package benchmarks

import scala.util.Random
import scala.collection.mutable
import scala.jdk.CollectionConverters._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import torchrec.Implicits._
import torchrec.data.{Dataset, MatchingDataset, TensorDataset}

/**
 * Synthetic data generator for benchmarks
 */
object DataGenerator {

  /**
   * Generate ranking dataset (DeepFM, WideDeep, DCN).
   * Returns (trainDataset, valDataset, testDataset) with the given feature names.
   */
  def generateRankingData(
                           numSamples: Int,
                           numSparseFeatures: Int,
                           numDenseFeatures: Int,
                           vocabSize: Int,
                           trainRatio: Float = 0.7f,
                           valRatio: Float = 0.1f,
                           seed: Int = 42,
                           featureNames: Seq[String] = Nil
                         ): (Dataset, Dataset, Dataset) = {
    val rng = new Random(seed)

    // Build feature names: use provided names or generate sparse_N/dense_N
    val sparseNames = (0 until numSparseFeatures).map { i =>
      Option(featureNames).flatMap { names =>
        if (i < names.length) Some(names(i)) else None
      }.getOrElse(s"sparse_$i")
    }
    val denseNames = (0 until numDenseFeatures).map(i => s"dense_$i")

    // Generate sparse features
    val sparseCols = sparseNames.map { name =>
      val data = Array.ofDim[Float](numSamples)
      for (j <- 0 until numSamples) {
        data(j) = rng.nextInt(vocabSize).toFloat
      }
      name -> tensor(data, Array(numSamples.toLong)).toType(ScalarType.Long)
    }.toMap

    // Generate dense features
    val denseCols = denseNames.map { name =>
      val data = Array.ofDim[Float](numSamples)
      for (j <- 0 until numSamples) {
        data(j) = rng.nextFloat() * 100
      }
      name -> tensor(data, Array(numSamples.toLong))
    }.toMap

    // Generate labels based on features (CTR: ~30% positive)
    val labelData = Array.ofDim[Float](numSamples)
    for (j <- 0 until numSamples) {
      var score = 0.0
      for (k <- 0 until math.min(3, numSparseFeatures)) {
        val idx = sparseCols(sparseNames(k)).toFloatArray(j).toInt % vocabSize
        score += (idx.toFloat / vocabSize) * (k + 1)
      }
      val prob = 1.0f / (1.0f + math.exp(-score).toFloat)
      labelData(j) = if (rng.nextFloat() < prob) 1.0f else 0.0f
    }
    val labels = tensor(labelData, Array(numSamples.toLong))

    // Split into train/val/test
    val trainSize = (numSamples * trainRatio).toInt
    val valSize = (numSamples * valRatio).toInt

    val trainSparse = sparseCols.map { case (k, v) => k -> v.narrow(0, 0, trainSize) }
    val trainDense = denseCols.map { case (k, v) => k -> v.narrow(0, 0, trainSize) }
    val trainLabels = labels.narrow(0, 0, trainSize)

    val valSparse = sparseCols.map { case (k, v) => k -> v.narrow(0, trainSize, valSize) }
    val valDense = denseCols.map { case (k, v) => k -> v.narrow(0, trainSize, valSize) }
    val valLabels = labels.narrow(0, trainSize, valSize)

    val testSparse = sparseCols.map { case (k, v) => k -> v.narrow(0, trainSize + valSize, numSamples - trainSize - valSize) }
    val testDense = denseCols.map { case (k, v) => k -> v.narrow(0, trainSize + valSize, numSamples - trainSize - valSize) }
    val testLabels = labels.narrow(0, trainSize + valSize, numSamples - trainSize - valSize)

    (
      new TensorDataset(trainSparse, trainDense, Some(trainLabels)),
      new TensorDataset(valSparse, valDense, Some(valLabels)),
      new TensorDataset(testSparse, testDense, Some(testLabels))
    )
  }

  /**
   * Generate matching dataset (DSSM) with user and item features.
   * Returns (trainDataset, valDataset, testDataset).
   */
  def generateMatchingData(
                            numUsers: Int,
                            numItems: Int,
                            avgSequenceLength: Int,
                            numUserFeatures: Int,
                            numItemFeatures: Int,
                            vocabSize: Int,
                            seed: Int = 42
                          ): (Dataset, Dataset, Dataset) = {
    val rng = new Random(seed)

    // User features
    val userCols = (0 until numUserFeatures).map { i =>
      val data = Array.ofDim[Float](numUsers)
      for (j <- 0 until numUsers) {
        data(j) = rng.nextInt(vocabSize).toFloat
      }
      s"user_$i" -> tensor(data, Array(numUsers.toLong)).toType(ScalarType.Long)
    }.toMap

    // Item features
    val itemCols = (0 until numItemFeatures).map { i =>
      val data = Array.ofDim[Float](numItems)
      for (j <- 0 until numItems) {
        data(j) = rng.nextInt(vocabSize).toFloat
      }
      s"item_$i" -> tensor(data, Array(numItems.toLong)).toType(ScalarType.Long)
    }.toMap

    // Pair users and items: total samples = min(numUsers, numItems)
    val totalSamples = math.min(numUsers, numItems)
    val userFeatCols = userCols.map { case (k, v) => k -> v.narrow(0, 0, totalSamples) }
    val itemFeatCols = itemCols.map { case (k, v) => k -> v.narrow(0, 0, totalSamples) }

    // Split
    val trainSize = (totalSamples * 0.8).toInt
    val valSize = (totalSamples * 0.1).toInt

    def makeMatchingDS(userF: Map[String, Tensor], itemF: Map[String, Tensor]) =
      new MatchingDataset(userF, itemF, None)

    (
      makeMatchingDS(
        userFeatCols.map { case (k, v) => k -> v.narrow(0, 0, trainSize) },
        itemFeatCols.map { case (k, v) => k -> v.narrow(0, 0, trainSize) }
      ),
      makeMatchingDS(
        userFeatCols.map { case (k, v) => k -> v.narrow(0, trainSize, valSize) },
        itemFeatCols.map { case (k, v) => k -> v.narrow(0, trainSize, valSize) }
      ),
      makeMatchingDS(
        userFeatCols.map { case (k, v) => k -> v.narrow(0, trainSize + valSize, totalSamples - trainSize - valSize) },
        itemFeatCols.map { case (k, v) => k -> v.narrow(0, trainSize + valSize, totalSamples - trainSize - valSize) }
      )
    )
  }

  /**
   * Generate multi-task dataset (MMOE, ShareBottom, etc.)
   */
  def generateMultiTaskData(
                             numSamples: Int,
                             numFeatures: Int,
                             taskNames: List[String],
                             vocabSize: Int,
                             seed: Int = 42,
                             featureNames: Seq[String] = Nil
                           ): (Dataset, Dataset, Dataset) = {
    generateRankingData(numSamples, numFeatures, 0, vocabSize, 0.8f, 0.1f, seed, featureNames)
  }
}
