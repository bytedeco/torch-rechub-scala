package torchrec.smoke

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
import torchrec.basic.features.{DenseFeature, SequenceFeature, SparseFeature}
import torchrec.models.ranking.{AutoInt, DIN, DIEN, EDCN}

object RankingSmokeTest {
  def main(args: Array[String]): Unit = {
    torch.manual_seed(42L)
    val embedDim = 4
    val batchSize = 3
    val numSparse = 4
    val seqLen = 6

    val sparseFeats = (0 until numSparse).map { i =>
      SparseFeature(name = s"sparse_$i", vocabSize = 10, embedDim = embedDim)
    }.toList

    val denseFeats = List(DenseFeature("dense_0", embedDim = 1))

    val seqFeats = List(SequenceFeature("hist", vocabSize = 10, embedDim = embedDim,
      pooling = "mean", maxLen = seqLen, paddingIdx = 0L))

    val sparseMap = sparseFeats.map { f =>
      f.name -> torch.randint(0L, 10L, Array(batchSize.toLong, 1L),
        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
    }.toMap
    val denseMap = Map("dense_0" -> torch.randn(Array(batchSize.toLong, 1L),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))))
    val seqMap = Map("hist" -> torch.randint(0L, 10L, Array(batchSize.toLong, seqLen.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long))))
    val targetMap = Map("hist" -> torch.randint(0L, 10L, Array(batchSize.toLong, 1L),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long))))

    println("--- AutoInt ---")
    val autoint = new AutoInt(sparseFeatures = sparseFeats, denseFeatures = denseFeats,
      numAttnHeads = 2, numLayers = 2, device = "cpu")
    val y1 = autoint.forward(sparseMap, denseMap)
    println(s"shape: ${y1.size(0)}, ${y1.size(1)}")
    require(y1.size(0) == batchSize.toLong && y1.size(1) == 1L)

    println("--- DIN ---")
    val din = new DIN(features = sparseFeats, sequenceFeatures = seqFeats,
      mlpDims = List(16L, 8L), device = "cpu")
    val y2 = din.forward(sparseMap, seqMap, targetMap)
    println(s"shape: ${y2.size(0)}")
    require(y2.size(0) == batchSize.toLong)

    println("--- DIEN ---")
    val dien = new DIEN(features = sparseFeats, sequenceFeatures = seqFeats,
      mlpDims = List(16L, 8L), device = "cpu")
    val y3 = dien.forward(sparseMap, seqMap)
    println(s"shape: ${y3.size(0)}")
    require(y3.size(0) == batchSize.toLong)

    println("--- EDCN ---")
    val edcn = new EDCN(features = sparseFeats, nCrossLayers = 2,
      mlpParams = Map("dims" -> List(8L, 4L), "activation" -> "relu", "dropout" -> 0.0f),
      bridgeType = "hadamard_product", device = "cpu")
    val y4 = edcn.forward(sparseMap)
    println(s"shape: ${y4.size(0)}")
    require(y4.size(0) == batchSize.toLong)

    println("OK")
  }
}
