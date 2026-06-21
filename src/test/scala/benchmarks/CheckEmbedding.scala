package benchmarks

import torchrec.basic.features.*
import torchrec.data.*
import torchrec.basic.layers.*
import torchrec.utils.DeviceSupport
import torchrec.Implicits.tensor
import torchrec.Implicits.*
import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.util.Random

/**
 * 检查 EmbeddingLayer 输出
 */
object CheckEmbedding {
  def main(args: Array[String]): Unit = {
    println("=" * 60)
    println("Checking EmbeddingLayer output dimensions")
    println("=" * 60)

    val numSparse = 10
    val vocabSize = 100
    val embedDim = 8
    val batchSize = 32
    val device = "cpu"

    // Create features
    val features = (0 until numSparse).map { i =>
      SparseFeature(s"feat_$i", vocabSize, embedDim)
    }.toList

    // Create embedding layer
    val embeddingLayer = new EmbeddingLayer(features, embedDim, device)

    // Create fake input data
    val sparseFeats = features.map { f =>
      val indices = tensor(Array.fill(batchSize)(scala.util.Random.nextInt(vocabSize).toFloat), Array(batchSize.toLong, 1L)).toType(ScalarType.Long)
      f.name -> indices
    }.toMap

    // Test forward (2D output)
    println("\n--- Testing forward(sparseFeats, squeeze=true) ---")
    try {
      val out2D = embeddingLayer.forward(sparseFeats, sequenceFeats = Map.empty, squeeze = true)
      val sizes = out2D.sizes()
      println(s"forward output shape: [${sizes.get(0)}, ${sizes.get(1)}]")
      println(s"Expected: ($batchSize, ${numSparse * embedDim}) = ($batchSize, 80)")
    } catch {
      case e: Throwable => println(s"Error: ${e.getMessage}")
    }

    // Test forward3D (3D output)
    println("\n--- Testing forward3D ---")
    try {
      val out3D = embeddingLayer.forward3D(sparseFeats, sequenceFeats = Map.empty)
      val sizes3D = out3D.sizes()
      println(s"forward3D output shape: [${sizes3D.get(0)}, ${sizes3D.get(1)}, ${sizes3D.get(2)}]")
      println(s"Expected: ($batchSize, $numSparse, $embedDim) = ($batchSize, 10, 8)")

      // Manually flatten to check
      val flattened = out3D.view(batchSize, -1)
      val flatSizes = flattened.sizes()
      println(s"After view($batchSize, -1): [${flatSizes.get(0)}, ${flatSizes.get(1)}]")
    } catch {
      case e: Throwable => println(s"Error: ${e.getMessage}")
    }

    // Test sparseDim calculation
    val sparseDim = Features.calcSparseDim(features)
    println(s"\nFeatures.calcSparseDim(features) = $sparseDim")
    val expectedSparseDim = features.map(_.embedDim).sum
    println(s"Expected: $expectedSparseDim")

    // Check individual feature embed dims
    features.foreach { f =>
      println(s"  ${f.name}: embedDim=${f.embedDim}")
    }

    println("\n" + "=" * 60)
  }
}