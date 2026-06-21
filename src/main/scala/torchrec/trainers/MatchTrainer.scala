package torchrec.trainers

import torchrec.Implicits._
import torchrec.data.DataLoader
import torchrec.models.matching._
import torchrec.models.ranking.DIEN
import torchrec.basic.metrics._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.mutable

import torchrec.basic.losses.BPRLoss
import torchrec.utils.DeviceSupport
import torchrec.TorchRec

/**
 * Trainer for matching/retrieval models
 */
class MatchTrainer(
  model: Module,
  learningRate: Float = 1e-3f,
  weightDecay: Float = 1e-6f,
  device: String = DeviceSupport.backend,
  mode: Int = 0,
  temperature: Float = 0.07f,
  numEpochs: Int = 10,
  earlyStopPatience: Int = 500,
  verbose: Boolean = true
) {
  private val optimizer = new Adam(model.parameters(), new AdamOptions(learningRate.toDouble))
  private val bprLoss = new BPRLoss()
  private var bestRecall = 0.0f
  private var patienceCounter = 0

  def fit(
    trainLoader: DataLoader,
    valLoader: Option[DataLoader] = None
  ): Unit = {
    model.train(true)

    for (epoch <- 0 until numEpochs) {
      var totalLoss = 0.0f
      var numBatches = 0

      println(s"MatchTrainer epoch ${epoch + 1}/$numEpochs")
      // Ensure DataLoader device matches model device to avoid cross-device tensor issues
      torchrec.utils.DataLoaderDevice.set(device)

      val iter = trainLoader.iterator
      while (iter.hasNext) {
        val batch = iter.next()
        val userFeats = batch.sparseFeatures
        val itemFeats = batch.itemFeatures

        if (userFeats.isEmpty || itemFeats.isEmpty) {
          // No features, skip
        } else {
          optimizer.zero_grad()

          model match {
            case dssm: DSSM =>
              val batchSize = userFeats.values.head.size(0).toInt

              if (batchSize > 1) {
                // In-batch negative sampling
                val userEmb = dssm.userTowerForward(userFeats)
                val itemEmb = dssm.itemTowerForward(itemFeats)

                // Positive scores: diagonal (user_i matched with item_i)
                val posScores = userEmb.mul(itemEmb).sum(1)

                // Accumulate loss using BPR triplet margin
                var lossTensor: Tensor = null.asInstanceOf[Tensor]
                var pairCount = 0
                for (i <- 0 until batchSize) {
                  val userVec = userEmb.select(0, i)
                  val posScoreI = posScores.select(0, i)
                  for (j <- 0 until batchSize if j != i) {
                    val negItemVec = itemEmb.select(0, j)
                    val negScoreIJ = userVec.mul(negItemVec).sum(0).reshape(1)
                    val pairLoss = bprLoss.apply(posScoreI.reshape(1), negScoreIJ)
                    if (lossTensor == null) {
                      lossTensor = pairLoss
                    } else {
                      lossTensor = lossTensor.add(pairLoss)
                    }
                    pairCount += 1
                  }
                }
                if (lossTensor != null && pairCount > 0) {
                  val avgLossTensor = lossTensor.div(new Scalar(pairCount.toDouble))
                  avgLossTensor.backward()
                  optimizer.step()
                  totalLoss += avgLossTensor.item().toFloat
                }
                numBatches += 1
                lossTensor.close() // = Nil  // close if not consumed
              } else {
                numBatches += 1
              }
            case mamba: MAMBA =>
              val batchSize = userFeats.values.head.size(0).toInt
              if (batchSize > 1) {
                // MAMBA takes tokens and positions as input
                val tokensOpt = batch.tokens
                val positionsOpt = batch.positions
                if (tokensOpt.nonEmpty) {
                  val tokens = tokensOpt.get
                  val positions = positionsOpt.getOrElse {
                    TorchRec.arange(0, tokens.size(1).toInt).toType(ScalarType.Long)
                      .unsqueeze(0)
                      .repeat(tokens.size(0), 1)
                  }
                  val userEmb = mamba.forward(tokens, positions)
                  // For matching, use in-batch item features as item embeddings
                  val itemEmb = if (itemFeats.nonEmpty) {
                    itemFeats.values.head
                  } else {
                    torch.zeros_like(userEmb)
                  }
                  // Vectorized in-batch negative sampling
                  // All pairwise scores: [batch, batch]
                  val itemEmbFloat = itemEmb.toType(ScalarType.Float)
                  val allScores = userEmb.matmul(itemEmbFloat.t())  // (batch, batch)
                  val batchIdx = torch.arange(new Scalar(0), new Scalar(batchSize.toDouble), new Scalar(1),
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
                    .to(new org.bytedeco.pytorch.Device(device), ScalarType.Long)
                  val posScores = allScores.gather(1, batchIdx.view(batchSize.toLong, 1)).squeeze()  // (batch,)
                  // Mask for diagonal (positive pairs) - use ones minus identity
                  val diagOnes = torch.ones(Array(batchSize.toLong, batchSize.toLong),
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                    .to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
                  val eyeMask = torch.eye(batchSize.toLong,
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                    .to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
                  val mask = diagOnes.sub(eyeMask)
                  val negScores = allScores.add(mask.mul(new Scalar(-1e9f)))  // mask out diagonal
                  val negScoresMax = torch.max(negScores, 1).get0  // (batch,)
                  // Simplified BPR loss: -log(sigmoid(pos - neg_max))
                  val diff = posScores.sub(negScoresMax)
                  val lossTensor = torch.log(torch.sigmoid(diff).add(new Scalar(1e-8f))).neg()
                  val avgLossTensor = lossTensor.mean()
                  avgLossTensor.backward()
                  optimizer.step()
                  totalLoss += avgLossTensor.item().toFloat
                  numBatches += 1
                  // Close tensors to free memory
                  lossTensor.close()
                  avgLossTensor.close()
                  allScores.close()
                  posScores.close()
                  negScores.close()
                  negScoresMax.close()
                  diff.close()
                  mask.close()
                  diagOnes.close()
                  eyeMask.close()
                } else {
                  numBatches += 1
                }
              } else {
                numBatches += 1
              }

            case mind: MIND =>
              val batchSize = userFeats.values.head.size(0).toInt
              if (batchSize > 1) {
                val tokensOpt = batch.tokens
                if (tokensOpt.nonEmpty) {
                  val tokens = tokensOpt.get
                  val userEmb = mind.forward(userFeats, tokens)
                  val itemEmb = if (itemFeats.nonEmpty) {
                    itemFeats.values.head
                  } else {
                    torch.zeros_like(userEmb)
                  }
                  val itemEmbFloat = itemEmb.toType(ScalarType.Float)
                  val allScores = userEmb.matmul(itemEmbFloat.t())
                  val batchIdx = torch.arange(new Scalar(0), new Scalar(batchSize.toDouble), new Scalar(1),
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
                    .to(new org.bytedeco.pytorch.Device(device), ScalarType.Long)
                  val posScores = allScores.gather(1, batchIdx.view(batchSize.toLong, 1)).squeeze()
                  val diagOnes = torch.ones(Array(batchSize.toLong, batchSize.toLong),
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                    .to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
                  val eyeMask = torch.eye(batchSize.toLong,
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                    .to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
                  val mask = diagOnes.sub(eyeMask)
                  val negScores = allScores.add(mask.mul(new Scalar(-1e9f)))
                  val negScoresMax = torch.max(negScores, 1).get0
                  val diff = posScores.sub(negScoresMax)
                  val lossTensor = torch.log(torch.sigmoid(diff).add(new Scalar(1e-8f))).neg()
                  val avgLossTensor = lossTensor.mean()
                  avgLossTensor.backward()
                  optimizer.step()
                  totalLoss += avgLossTensor.item().toFloat
                  numBatches += 1
                  lossTensor.close()
                  avgLossTensor.close()
                  allScores.close()
                  posScores.close()
                  negScores.close()
                  negScoresMax.close()
                  diff.close()
                  mask.close()
                  diagOnes.close()
                  eyeMask.close()
                } else {
                  numBatches += 1
                }
              } else {
                numBatches += 1
              }

            case dien: DIEN =>
              val batchSize = userFeats.values.head.size(0).toInt
              if (batchSize > 1) {
                val tokensOpt = batch.tokens
                if (tokensOpt.nonEmpty) {
                  val tokens = tokensOpt.get
                  // Prefer explicit sequenceFeatures map from the batch if available,
                  // otherwise fall back to a default key used by pipelines ("seq_feat").
                  val seqFeatsMap = if (batch.sequenceFeatures.nonEmpty) batch.sequenceFeatures else Map("seq_feat" -> tokens)
                  val userEmb = dien.forward(userFeats, seqFeatsMap)
                  val itemEmb = if (itemFeats.nonEmpty) {
                    itemFeats.values.head
                  } else {
                    torch.zeros_like(userEmb)
                  }
                  val itemEmbFloat = itemEmb.toType(ScalarType.Float)
                  val allScores = userEmb.matmul(itemEmbFloat.t())
                  val batchIdx = torch.arange(new Scalar(0), new Scalar(batchSize.toDouble), new Scalar(1),
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
                    .to(new org.bytedeco.pytorch.Device(device), ScalarType.Long)
                  val posScores = allScores.gather(1, batchIdx.view(batchSize.toLong, 1)).squeeze()
                  val diagOnes = torch.ones(Array(batchSize.toLong, batchSize.toLong),
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                    .to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
                  val eyeMask = torch.eye(batchSize.toLong,
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                    .to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
                  val mask = diagOnes.sub(eyeMask)
                  val negScores = allScores.add(mask.mul(new Scalar(-1e9f)))
                  val negScoresMax = torch.max(negScores, 1).get0
                  val diff = posScores.sub(negScoresMax)
                  val lossTensor = torch.log(torch.sigmoid(diff).add(new Scalar(1e-8f))).neg()
                  val avgLossTensor = lossTensor.mean()
                  avgLossTensor.backward()
                  optimizer.step()
                  totalLoss += avgLossTensor.item().toFloat
                  numBatches += 1
                  lossTensor.close()
                  avgLossTensor.close()
                  allScores.close()
                  posScores.close()
                  negScores.close()
                  negScoresMax.close()
                  diff.close()
                  mask.close()
                  diagOnes.close()
                  eyeMask.close()
                } else {
                  numBatches += 1
                }
              } else {
                numBatches += 1
              }

            case _ =>
            // Skip unknown model types
          }
        }
      }

      val avgLoss = if (numBatches > 0) totalLoss / numBatches else 0.0f

      if (verbose) print(s"Epoch $epoch: loss=${f"$avgLoss%.4f"}")

      // Validate if provided
      valLoader.foreach { vl =>
        model.eval()
        val recall = evaluate(vl, topk = 10)
        if (verbose) print(f", Recall@10=${recall%.4f}")

        if (recall > bestRecall) {
          bestRecall = recall
          patienceCounter = 0
        } else {
          patienceCounter += 1
        }
        model.train(true)
      }

      if (verbose) println()

      if (patienceCounter >= earlyStopPatience) {
        if (verbose) println(s"Early stopping at epoch $epoch")
        return
      }
    }
  }

  def inferenceEmbedding(
    dataLoader: DataLoader,
    mode: String
  ): Array[Tensor] = {
    model.eval()
    val embeddings = mutable.ArrayBuffer[Tensor]()

    val iter = dataLoader.iterator
    while (iter.hasNext) {
      val batch = iter.next()
      mode match {
        case "user" =>
          val userFeats = batch.sparseFeatures
          val tokensOpt = batch.tokens
          val positionsOpt = batch.positions
          (model, userFeats.get("history")) match {
            case (dssm: DSSM, _) =>
              embeddings += dssm.userTowerForward(userFeats)
            case (yt: YoutubeDNN, Some(history)) =>
              val sparseOnly = userFeats - "history"
              embeddings += yt.userTowerForward(sparseOnly, Map("history" -> history))
            case (yt: YoutubeDNN, None) =>
              embeddings += yt.userTowerForward(userFeats)
            case (mamba: MAMBA, _) if tokensOpt.nonEmpty =>
              val tokens = tokensOpt.get
              val positions = positionsOpt.getOrElse {
                TorchRec.arange(0, tokens.size(1).toInt).toType(ScalarType.Long)
                  .unsqueeze(0)
                  .repeat(tokens.size(0), 1)
              }
              embeddings += mamba.forward(tokens, positions)
            case _ =>
          }
        case "item" =>
          val itemFeats = batch.itemFeatures
          if (itemFeats.nonEmpty) {
            model match {
              case dssm: DSSM =>
                embeddings += dssm.itemTowerForward(itemFeats)
              case _ =>
            }
          }
        case _ =>
      }
    }

    embeddings.toArray
  }

  /**
   * Evaluate using in-batch evaluation.
   * For each user i in the batch, rank all items by score(user_i, item_j).
   * Positive item for user i is item i (diagonal).
   * HitRate@K = fraction of users whose positive item is in top-K.
   * NDCG@K = average NDCG@K across users.
   * MRR = average MRR across users.
   */
  def evaluate(dataLoader: DataLoader, topk: Int = 10): Float = {
    model.eval()

    val hitRateMetric = new HitRate[Float](topk)
    val ndcgMetric = new NDCG[Float](topk)
    val mrrMetric = new MRR()

    val iter = dataLoader.iterator
    while (iter.hasNext) {
      val batch = iter.next()
      val userFeats = batch.sparseFeatures
      val itemFeats = batch.itemFeatures
      val labelsOpt = batch.labels

      if (userFeats.isEmpty || itemFeats.isEmpty) { /* skip */ }
      else {
        val batchSize = userFeats.values.head.size(0).toInt
        if (batchSize < 2) { /* skip single-item batches */ }
        else {
          model match {
            case dssm: DSSM =>
              val userEmb = dssm.userTowerForward(userFeats)
              val itemEmb = dssm.itemTowerForward(itemFeats)

              // Compute all user-item scores: (batch, embed) @ (embed, batch) -> (batch, batch)
              val allScores = torch.matmul(userEmb, itemEmb.t())
              val scoresHost = allScores.to(ScalarType.Float).contiguous().cpu()
              val scoreArr = scoresHost.toFloatArray
              allScores.close()
              scoresHost.close()

              // For each user i, build label array (1 at positive item j=i, 0 elsewhere)
              for (i <- 0 until batchSize) {
                val rowOffset = i * batchSize
                val labelArr = Array.tabulate[Float](batchSize)(j => if (j == i) 1.0f else 0.0f)
                val scoreSlice = scoreArr.slice(rowOffset, rowOffset + batchSize)
                hitRateMetric.update(scoreSlice, labelArr)
                ndcgMetric.update(scoreSlice, labelArr)
                mrrMetric.update(scoreSlice, labelArr)
              }
            case mamba: MAMBA =>
              val tokensOpt = batch.tokens
              val positionsOpt = batch.positions
              if (tokensOpt.nonEmpty) {
                val tokens = tokensOpt.get
                val positions = positionsOpt.getOrElse {
                  TorchRec.arange(0, tokens.size(1).toInt).toType(ScalarType.Long)
                    .unsqueeze(0)
                    .repeat(tokens.size(0), 1)
                }
                val userEmb = mamba.forward(tokens, positions)
                val itemEmb = torch.zeros_like(userEmb)
                val allScores = torch.matmul(userEmb, itemEmb.t())
                val scoresHost = allScores.to(ScalarType.Float).contiguous().cpu()
                val scoreArr = scoresHost.toFloatArray
                allScores.close()
                scoresHost.close()

                for (i <- 0 until batchSize) {
                  val rowOffset = i * batchSize
                  val labelArr = Array.tabulate[Float](batchSize)(j => if (j == i) 1.0f else 0.0f)
                  val scoreSlice = scoreArr.slice(rowOffset, rowOffset + batchSize)
                  hitRateMetric.update(scoreSlice, labelArr)
                  ndcgMetric.update(scoreSlice, labelArr)
                  mrrMetric.update(scoreSlice, labelArr)
                }
              }
            case _ =>
          }
        }
      }
    }

    model.train(true)
    // Return HitRate@K as the primary metric
    hitRateMetric.compute()
  }

  def evaluateFull(dataLoader: DataLoader, topk: Int = 10): Map[String, Float] = {
    model.eval()

    val hitRateMetric = new HitRate[Float](topk)
    val ndcgMetric = new NDCG[Float](topk)
    val mrrMetric = new MRR()

    val iter = dataLoader.iterator
    while (iter.hasNext) {
      val batch = iter.next()
      val userFeats = batch.sparseFeatures
      val itemFeats = batch.itemFeatures

      if (userFeats.isEmpty || itemFeats.isEmpty) { /* skip */ }
      else {
        val batchSize = userFeats.values.head.size(0).toInt
        if (batchSize < 2) { /* skip */ }
        else {
          model match {
            case dssm: DSSM =>
              val userEmb = dssm.userTowerForward(userFeats)
              val itemEmb = dssm.itemTowerForward(itemFeats)
              val allScores = torch.matmul(userEmb, itemEmb.t())
              val scoresHost = allScores.to(ScalarType.Float).contiguous().cpu()
              val scoreArr = scoresHost.toFloatArray
              allScores.close()
              scoresHost.close()

              for (i <- 0 until batchSize) {
                val rowOffset = i * batchSize
                val labelArr = Array.tabulate[Float](batchSize)(j => if (j == i) 1.0f else 0.0f)
                val scoreSlice = scoreArr.slice(rowOffset, rowOffset + batchSize)
                hitRateMetric.update(scoreSlice, labelArr)
                ndcgMetric.update(scoreSlice, labelArr)
                mrrMetric.update(scoreSlice, labelArr)
              }
            case _ =>
          }
        }
      }
    }

    model.train(true)
    Map(
      s"Hit@$topk" -> hitRateMetric.compute(),
      s"NDCG@$topk" -> ndcgMetric.compute(),
      "MRR" -> mrrMetric.compute()
    )
  }
}
