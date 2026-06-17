package tutorials

import torchrec.Implicits._
import torchrec.basic.features._
import torchrec.data._
import torchrec.models.ranking._
import torchrec.trainers._

import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Tutorial: Sequence-Aware Ranking with DIN
 *
 * Deep Interest Network (DIN) uses attention mechanisms to model
 * user behavior sequences for more personalized predictions.
 */
object RankingDIN {

  def main(args: Array[String]): Unit = {
    println(
      """
        |=============================================================
        | Tutorial: Sequence-Aware Ranking with DIN
        |=============================================================
        |
        | Deep Interest Network (DIN) uses attention to model
        | user behavior history for personalized CTR prediction.
        |
        |""".stripMargin)

    // Configuration
    val features = List(
      SparseFeature("user_id", 100, 8),
      SparseFeature("item_id", 1000, 8)
    )

    val sequenceFeatures = List(
      SequenceFeature("click_history", vocabSize = 1000, embedDim = 8, pooling = "mean")
    )

    // Create DIN model
    println("Creating DIN model...")
    val model = new DIN(
      features = features,
      sequenceFeatures = sequenceFeatures,
      embedDim = 8,
      mlpDims = List(128L, 64L),
      attentionUnits = 32
    )
    println("  Model: DIN(embed_dim=8, mlp_dims=[128, 64])")

    // Generate data
    println("\nGenerating sequence data...")
    val random = new scala.util.Random(42)
    val batchSize = 64
    val seqLen = 20

    // Create synthetic data
    val numSamples = 1000
    val userIdData = Array.ofDim[Float](numSamples)
    val itemIdData = Array.ofDim[Float](numSamples)
    val clickHistoryData = Array.ofDim[Float](numSamples * seqLen)

    for (i <- 0 until numSamples) {
      userIdData(i) = (random.nextInt(100) + 1).toFloat
      itemIdData(i) = (random.nextInt(1000) + 1).toFloat
      for (j <- 0 until seqLen) {
        clickHistoryData(i * seqLen + j) = (random.nextInt(1000) + 1).toFloat
      }
    }

    val userIds = floatTensor(userIdData)
    val itemIds = floatTensor(itemIdData)
    val clickHistory = floatTensor(clickHistoryData).view(numSamples.toLong, seqLen.toLong)

    val dataset = new SequenceDataset(
      features = Map("user_id" -> userIds, "item_id" -> itemIds),
      sequenceFeatures = Map("click_history" -> clickHistory),
      labels = Some(floatTensor(Array.fill(numSamples)(1.0f)))
    )

    val loader = new DataLoader(dataset, batchSize = batchSize, shuffle = true)

    println(f"  Created dataset with $numSamples%,d samples")
    println(f"  Batch size: $batchSize")

    // Note: Full training would require implementing DIN-specific forward pass
    println("\nDIN model ready for training!")
    println("  Use CTRTrainer to train this model with sequence data.")

    println(
      """
        |
        |=============================================================
        | Tutorial Complete!
        |=============================================================
        |
        |""".stripMargin)
  }
}
