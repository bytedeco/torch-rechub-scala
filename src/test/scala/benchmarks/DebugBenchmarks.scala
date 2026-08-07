package benchmarks

import torchrec.basic.features._
import torchrec.data._
import torchrec.models.ranking._
import torchrec.models.matching._
import torchrec.trainers._
import torchrec.utils.DeviceSupport
import torchrec.Implicits.tensor
import torchrec.Implicits._

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
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.util.Random

/** 精简调试脚本：先单独验证 XGBoost 和 MAMBA 再合并 */
object DebugBenchmarks {

  def main(args: Array[String]): Unit = {
    // 强制使用 CPU，避免 CUDA 设备不一致问题
    println("=== Debug: XGBoost (CPU) ===")
    testXGBoostAliExpressCPU()

    println("\n=== Debug: MAMBA (CPU) ===")
    testMAMBAAiExpressCPU()
  }

  def testXGBoostAliExpressCPU(): Unit = {
    val device = "cpu"
    // 设置线程级 device
    torchrec.utils.DataLoaderDevice.set(device)
    try {
      val taskNames = List("click", "conversion")
      val (trainDS, _) = AliExpressDataset.load(
        datasetPath = "./data/AliExpress_NL",
        taskNames = taskNames
      )
      println(s"  Data loaded: train=${trainDS.size}")

      val allFeatureNames = trainDS.features.keys.toList
      val sparseNames = allFeatureNames.filter(_.startsWith("cat_")).take(16)

      val features = sparseNames.map { name =>
        SparseFeature(name, 1000, 8)
      }.toList

      val clickLabelKey = taskNames.head
      val trainLabels = trainDS.taskLabels.get(clickLabelKey)

      val trainSingleDS = new torchrec.data.TensorDataset(
        trainDS.features,
        Map.empty,
        trainLabels
      )

      val trainLoader = new DataLoader(trainSingleDS, 256, shuffle = true)

      val numSparse = sparseNames.size
      val linkFeatDim = (numSparse * 8).toLong
      val model = new XGBoostModel(
        features = features,
        numTrees = 4,
        treeDepth = 3,
        embedDim = 8,
        linkFeatDim = linkFeatDim,
        device = device
      )

      val trainer = new CTRTrainer(
        model,
        learningRate = 0.001f,
        device = device,
        numEpochs = 1,
        verbose = true
      )

      trainer.fit(trainLoader, None)
      println("  [PASS] XGBoost training done")
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] XGBoost: ${e.getMessage}")
        e.printStackTrace()
    } finally {
      torchrec.utils.DataLoaderDevice.clear()
    }
  }

  def testMAMBAAiExpressCPU(): Unit = {
    val device = "cpu"
    // 设置线程级 device
    torchrec.utils.DataLoaderDevice.set(device)
    try {
      val taskNames = List("click", "conversion")
      val (trainDS, _) = AliExpressDataset.load(
        datasetPath = "./data/AliExpress_NL",
        taskNames = taskNames
      )
      println(s"  Data loaded: ${trainDS.size}")

      val numSamples = math.min(trainDS.size.toInt, 1000)
      val seqLen = 10
      val vocabSize = 1000L
      val rng = new Random(42)

      val catFeatName = "cat_0"
      val catTensor = trainDS.features(catFeatName)

      val tokensArr = Array.ofDim[Float](numSamples * seqLen)
      for (i <- 0 until numSamples) {
        val baseIdx = catTensor.select(0, i).itemSafe().toInt
        for (j <- 0 until seqLen) {
          val offset = rng.nextInt(100) - 50
          tokensArr(i * seqLen + j) = math.max(0, math.min(vocabSize - 1, baseIdx + offset)).toFloat
        }
      }
      val tokensTensor = tensor(tokensArr, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

      val positionsArr = Array.range(0, seqLen).map(_.toFloat)
      val positionsFlat = Array.fill(numSamples)(positionsArr).flatten
      val positionsTensor = tensor(positionsFlat, Array(numSamples.toLong, seqLen.toLong)).toType(ScalarType.Long)

      val seqDataset = new torchrec.data.SequenceDataset(
        features = Map.empty,
        sequenceFeatures = Map.empty,
        labels = None,
        tokens = Some(tokensTensor),
        positions = Some(positionsTensor)
      )

      val trainLoader = new DataLoader(seqDataset, 64, shuffle = true)

      val model = new MAMBA(
        vocabSize = vocabSize,
        embedDim = 32,
        dState = 4,
        numLayers = 1,
        maxSeqLen = seqLen,
        mlpDims = List(32L),
        dropout = 0.1f,
        device = device
      )

      val trainer = new MatchTrainer(
        model,
        learningRate = 0.001f,
        device = device,
        numEpochs = 1,
        verbose = true
      )

      trainer.fit(trainLoader)
      println("  [PASS] MAMBA training done")
    } catch {
      case e: Throwable =>
        println(s"  [FAIL] MAMBA: ${e.getMessage}")
        e.printStackTrace()
    } finally {
      torchrec.utils.DataLoaderDevice.clear()
    }
  }
}
