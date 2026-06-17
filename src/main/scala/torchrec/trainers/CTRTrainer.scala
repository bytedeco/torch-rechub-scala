package torchrec.trainers

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.mutable
import scala.util.Random
import torchrec.Implicits.*
import torchrec.TorchRec
import torchrec.utils.DeviceSupport
import torchrec.data.*
import torchrec.basic.metrics.*
import torchrec.models.ranking.DeepFM
import torchrec.models.ranking.DCN
import torchrec.models.ranking.DCNv2
import torchrec.models.ranking.AFM
import torchrec.models.ranking.WideDeep
import torchrec.models.ranking.DIN
import torchrec.models.ranking.BST
import torchrec.models.ranking.SIM
import torchrec.models.ranking.ETA
import torchrec.models.ranking.MEMBA
import torchrec.models.ranking.XGBoostModel
import torchrec.models.ranking.LiquidNetWork
import torchrec.models.generative.{LLM4Rec, HLLM}
import torchrec.models.matching.MAMBA
import torchrec.basic.losses.{BCELoss, BCEWithLogitsLoss}

/**
 * Trainer for CTR (Click-Through Rate) models
 */
class CTRTrainer(
  model: Module,
  learningRate: Float = 1e-3f,
  weightDecay: Float = 1e-6f,
  device: String = DeviceSupport.backend,
  numEpochs: Int = 10,
  earlyStopPatience: Int = 5,
  verbose: Boolean = true
) {
  private var bestAUC = 0.0f
  private var patienceCounter = 0
  private val rng = new Random(42)
  private val optimizer = new Adam(model.parameters(), new AdamOptions(learningRate.toDouble))
  private val bceLoss = new BCEWithLogitsLoss()

  def fit(
    trainLoader: DataLoader,
    valLoader: Option[DataLoader] = None
  ): Unit = {
    model.train(true)

    for (epoch <- 0 until numEpochs) {
      var totalLoss = 0.0f
      var numBatches = 0

      // Ensure DataLoader device matches model device to avoid cross-device tensor issues
      torchrec.utils.DataLoaderDevice.set(device)

      val iter = trainLoader.iterator
      while (iter.hasNext) {
        val batch = iter.next()
        val features = batch.sparseFeatures
        val labelsOpt = batch.labels

        if (labelsOpt.isEmpty) {
          // No labels, skip
        } else {
          val labels = labelsOpt.get

          // Zero gradients before backward
          optimizer.zero_grad()

          model match {
             case deepFM: DeepFM =>
               val pred = deepFM.forward(features)
               val batchSize = pred.size(0).toInt
               val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
               val loss = bceLoss.apply(pred, target2D)
               loss.backward(); optimizer.step()
               totalLoss += loss.itemSafe().toFloat; numBatches += 1

             case dcn: DCN =>
               val pred = dcn.forward(features)
               val batchSize = pred.size(0).toInt
               val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
               val loss = bceLoss.apply(pred, target2D)
               loss.backward(); optimizer.step()
               totalLoss += loss.itemSafe().toFloat; numBatches += 1

             case dcnv2: DCNv2 =>
               val pred = dcnv2.forward(features)
               val batchSize = pred.size(0).toInt
               val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
               val loss = bceLoss.apply(pred, target2D)
               loss.backward(); optimizer.step()
               totalLoss += loss.itemSafe().toFloat; numBatches += 1

             case afm: AFM =>
               val pred = afm.forward(features)
               val batchSize = pred.size(0).toInt
               val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
               val loss = bceLoss.apply(pred, target2D)
               loss.backward(); optimizer.step()
               totalLoss += loss.itemSafe().toFloat; numBatches += 1

             case wd: WideDeep =>
               val pred = wd.forward(features)
               val batchSize = pred.size(0).toInt
               val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
               val loss = bceLoss.apply(pred, target2D)
               loss.backward(); optimizer.step()
               totalLoss += loss.itemSafe().toFloat; numBatches += 1

             case din: DIN =>
               val seqFeats = batch.sequenceFeatures
               if (seqFeats.nonEmpty) {
                 val targetIdx = labels.view(labels.size(0), 1).toType(ScalarType.Long)
                 val pred = din.forward(features, seqFeats, targetIdx)
                 val batchSize = pred.size(0).toInt
                 val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
                 val loss = bceLoss.apply(pred, target2D)
                 loss.backward(); optimizer.step()
                 totalLoss += loss.itemSafe().toFloat; numBatches += 1
                 targetIdx.close()
               }

             case bst: BST =>
               val seqFeats = batch.sequenceFeatures
               if (seqFeats.nonEmpty) {
                 val pred = bst.forward(features, seqFeats)
                 val batchSize = pred.size(0).toInt
                 val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
                 val loss = bceLoss.apply(pred, target2D)
                 loss.backward(); optimizer.step()
                 totalLoss += loss.itemSafe().toFloat; numBatches += 1
               }

             case sim: SIM =>
               val seqFeats = batch.sequenceFeatures
               if (seqFeats.nonEmpty) {
                 // Use same sequence features for item, category, and time
                 val pred = sim.forward(features, seqFeats, seqFeats, seqFeats, seqFeats)
                 val batchSize = pred.size(0).toInt
                 val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
                 val loss = bceLoss.apply(pred, target2D)
                 loss.backward(); optimizer.step()
                 totalLoss += loss.itemSafe().toFloat; numBatches += 1
               }

             case eta: ETA =>
               val seqFeats = batch.sequenceFeatures
               if (seqFeats.nonEmpty) {
                 val pred = eta.forward(features, seqFeats, seqFeats)
                 val batchSize = pred.size(0).toInt
                 val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
                 val loss = bceLoss.apply(pred, target2D)
                 loss.backward(); optimizer.step()
                 totalLoss += loss.itemSafe().toFloat; numBatches += 1
               }

             case membrana: MEMBA =>
               val seqFeats = batch.sequenceFeatures
               if (seqFeats.nonEmpty) {
                 val targetIdx = labels.view(labels.size(0), 1).toType(ScalarType.Long)
                 val pred = membrana.forward(features, seqFeats, targetIdx)
                 val batchSize = pred.size(0).toInt
                 val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
                 val loss = bceLoss.apply(pred, target2D)
                 loss.backward(); optimizer.step()
                 totalLoss += loss.itemSafe().toFloat; numBatches += 1
                 targetIdx.close()
               }

             case xgb: XGBoostModel =>
               val pred = xgb.forward(features)
               val batchSize = pred.size(0).toInt
               val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
               val loss = bceLoss.apply(pred, target2D)
               loss.backward(); optimizer.step()
               totalLoss += loss.itemSafe().toFloat; numBatches += 1

             case lnw: LiquidNetWork =>
               val seqFeats = batch.sequenceFeatures
               if (seqFeats.nonEmpty) {
                 val pred = lnw.forward(features, seqFeats)
                 val batchSize = pred.size(0).toInt
                 val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
                 val loss = bceLoss.apply(pred, target2D)
                 loss.backward(); optimizer.step()
                 totalLoss += loss.itemSafe().toFloat; numBatches += 1
               }

             case llm4rec: LLM4Rec =>
               val tokensOpt = batch.tokens
               val positionsOpt = batch.positions
               if (tokensOpt.nonEmpty) {
                 val tokens = tokensOpt.get
                 val positions = positionsOpt.getOrElse {
                   val seqLen = tokens.size(1).toInt
                   val posArr = Array.range(0, seqLen).map(_.toFloat)
                   val posFlat = Array.fill(tokens.size(0).toInt)(posArr).flatten
                   TorchRec.arange(0, seqLen).toType(ScalarType.Long)
                     .unsqueeze(0)
                     .repeat(tokens.size(0), 1)
                 }
                 val pred = llm4rec.forward(tokens, positions)
                 val batchSize = pred.size(0).toInt
                 val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
                 val loss = bceLoss.apply(pred, target2D)
                 loss.backward(); optimizer.step()
                 totalLoss += loss.itemSafe().toFloat; numBatches += 1
               }

//             case hllm: HLLM =>
//               val tokensOpt = batch.tokens
//               val timeDiffsOpt = batch.timeDiffs
//               if (tokensOpt.nonEmpty) {
//                 val tokens = tokensOpt.get
//                 val timeDiffs = if (timeDiffsOpt.nonEmpty) Some(timeDiffsOpt.get) else None
//                 val pred = hllm.forward(tokens, timeDiffs)
//                 val batchSize = pred.size(0).toInt
//                 val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
//                 val loss = bceLoss.apply(pred, target2D)
//                 loss.backward(); optimizer.step()
//                 totalLoss += loss.itemSafe().toFloat; numBatches += 1
//               }

             case hllm: HLLM =>
               val tokensOpt = batch.tokens
               val timeDiffsOpt = batch.timeDiffs
               // 🔥 确保我们有目标 Item 作为评估对象
               if (tokensOpt.nonEmpty && features.nonEmpty) {
                 val tokens = tokensOpt.get
                 val timeDiffs = if (timeDiffsOpt.nonEmpty) Some(timeDiffsOpt.get) else None

                 // 1. 获取全序列预测 [batchSize, seqLen, vocabSize]
                 val allLogits = hllm.forward(tokens, timeDiffs)
                 val batchSize = allLogits.size(0).toInt
                 val seqLen = allLogits.size(1).toInt

                 // 2. 截取最后一步的时间步 [batchSize, vocabSize]
                 val lastStepLogits = allLogits.select(1, seqLen - 1)

                 // 3. 提取目标 Item (通常在 sparseFeatures 中)
                 val targetItem = features.values.head
                 val targetItem2D = targetItem.view(batchSize, 1).toType(ScalarType.Long)

                 // 4. 使用 gather 提取该 Target Item 的单一 Logit -> [batchSize, 1]
                 val pred = lastStepLogits.gather(1, targetItem2D)

                 val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
                 val loss = bceLoss.apply(pred, target2D)
                 loss.backward(); optimizer.step()
                 totalLoss += loss.itemSafe().toFloat; numBatches += 1
               }
            case _ =>
            // Skip unknown model types
          }
        }
      }

      val avgLoss = if (numBatches > 0) totalLoss / numBatches else 0.0f

      if (verbose) {
        print(s"Epoch $epoch: train_loss=${f"$avgLoss%.4f"}")
      }

      // Validate if provided
      valLoader.foreach { vl =>
        try {
          if (model != null) model.eval()
        } catch {
          case e: Throwable => // Skip eval on error
        }
        val metrics = evaluate(vl)
        val auc = metrics.getOrElse("AUC", 0.0f)
        if (verbose) print(s", val_auc=${f"$auc%.4f"}")

        if (auc > bestAUC) {
          bestAUC = auc
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

  def evaluate(dataLoader: DataLoader): Map[String, Float] = {
    try {
    val predList = mutable.ListBuffer[Float]()
    val labelList = mutable.ListBuffer[Float]()

    val iter = dataLoader.iterator
    while (iter.hasNext) {
      val batch = iter.next()
      val features = batch.sparseFeatures
      val seqFeats = batch.sequenceFeatures
      val labelsOpt = batch.labels

      if (labelsOpt.nonEmpty) {
        val label = labelsOpt.get
        try {
          model match {
          case deepFM: DeepFM =>
            val pred = deepFM.forward(features).sigmoid()
            val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
            val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
            predList.appendAll(predHost.toFloatArray)
            labelList.appendAll(labelHost.toFloatArray)
            predHost.close(); labelHost.close()

          case dcn: DCN =>
            val pred = dcn.forward(features).sigmoid()
            val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
            val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
            predList.appendAll(predHost.toFloatArray)
            labelList.appendAll(labelHost.toFloatArray)
            predHost.close(); labelHost.close()

          case dcnv2: DCNv2 =>
            val pred = dcnv2.forward(features).sigmoid()
            val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
            val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
            predList.appendAll(predHost.toFloatArray)
            labelList.appendAll(labelHost.toFloatArray)
            predHost.close(); labelHost.close()

          case afm: AFM =>
            val pred = afm.forward(features).sigmoid()
            val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
            val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
            predList.appendAll(predHost.toFloatArray)
            labelList.appendAll(labelHost.toFloatArray)
            predHost.close(); labelHost.close()

          case wd: WideDeep =>
            val pred = wd.forward(features).sigmoid()
            val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
            val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
            predList.appendAll(predHost.toFloatArray)
            labelList.appendAll(labelHost.toFloatArray)
            predHost.close(); labelHost.close()

          case din: DIN =>
            if (seqFeats.nonEmpty) {
              val targetIdx = label.view(label.size(0), 1).toType(ScalarType.Long)
              val pred = din.forward(features, seqFeats, targetIdx).sigmoid()
              val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
              val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
              predList.appendAll(predHost.toFloatArray)
              labelList.appendAll(labelHost.toFloatArray)
              predHost.close(); labelHost.close()
              targetIdx.close()
            }

          case bst: BST =>
            if (seqFeats.nonEmpty) {
              val pred = bst.forward(features, seqFeats).sigmoid()
              val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
              val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
              predList.appendAll(predHost.toFloatArray)
              labelList.appendAll(labelHost.toFloatArray)
              predHost.close(); labelHost.close()
            }

          case sim: SIM =>
            if (seqFeats.nonEmpty) {
              val pred = sim.forward(features, seqFeats, seqFeats, seqFeats, seqFeats).sigmoid()
              val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
              val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
              predList.appendAll(predHost.toFloatArray)
              labelList.appendAll(labelHost.toFloatArray)
              predHost.close(); labelHost.close()
            }

          case eta: ETA =>
            if (seqFeats.nonEmpty) {
              val pred = eta.forward(features, seqFeats, seqFeats).sigmoid()
              val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
              val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
              predList.appendAll(predHost.toFloatArray)
              labelList.appendAll(labelHost.toFloatArray)
              predHost.close(); labelHost.close()
            }

          case membrana: MEMBA =>
            if (seqFeats.nonEmpty) {
              val targetIdx = label.view(label.size(0), 1).toType(ScalarType.Long)
              val pred = membrana.forward(features, seqFeats, targetIdx).sigmoid()
              val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
              val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
              predList.appendAll(predHost.toFloatArray)
              labelList.appendAll(labelHost.toFloatArray)
              predHost.close(); labelHost.close()
              targetIdx.close()
            }

          case xgb: XGBoostModel =>
            val pred = xgb.forward(features).sigmoid()
            val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
            val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
            predList.appendAll(predHost.toFloatArray)
            labelList.appendAll(labelHost.toFloatArray)
            predHost.close(); labelHost.close()

          case lnw: LiquidNetWork =>
            if (seqFeats.nonEmpty) {
              val pred = lnw.forward(features, seqFeats).sigmoid()
              val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
              val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
              predList.appendAll(predHost.toFloatArray)
              labelList.appendAll(labelHost.toFloatArray)
              predHost.close(); labelHost.close()
            }

          case llm4rec: LLM4Rec =>
            val tokensOpt = batch.tokens
            val positionsOpt = batch.positions
            if (tokensOpt.nonEmpty) {
              val tokens = tokensOpt.get
              val positions = positionsOpt.getOrElse {
                val seqLen = tokens.size(1).toInt
                TorchRec.arange(0, seqLen).toType(ScalarType.Long)
                  .unsqueeze(0)
                  .repeat(tokens.size(0), 1)
              }
              val pred = llm4rec.forward(tokens, positions).sigmoid()
              val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
              val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
              predList.appendAll(predHost.toFloatArray)
              labelList.appendAll(labelHost.toFloatArray)
              predHost.close(); labelHost.close()
            }
          case hllm: HLLM =>
            if (seqFeats.nonEmpty && features.nonEmpty) {
              val tokensOpt = batch.tokens
              val timeDiffsOpt = batch.timeDiffs
              if (tokensOpt.nonEmpty) {
                val tokens = tokensOpt.get
                val timeDiffs = if (timeDiffsOpt.nonEmpty) Some(timeDiffsOpt.get) else None

                // 1. 获取全序列预测
                val allLogits = hllm.forward(tokens, timeDiffs)
                val batchSize = allLogits.size(0).toInt
                val seqLen = allLogits.size(1).toInt

                // 2. 截取最后一步并查表 Target Item
                val lastStepLogits = allLogits.select(1, seqLen - 1)
                val targetItem = features.values.head
                val targetItem2D = targetItem.view(batchSize, 1).toType(ScalarType.Long)

                // 3. 获取 Target Logit 并通过 Sigmoid 转为概率
                val pred = lastStepLogits.gather(1, targetItem2D).sigmoid()

                val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
                val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
                predList.appendAll(predHost.toFloatArray)
                labelList.appendAll(labelHost.toFloatArray)
                predHost.close(); labelHost.close()
              }
            }
          case _ =>
            // Unknown model type — skip evaluation
        }
        } catch {
          case e: Throwable => // Skip on error
        }
      }
    }

    if (predList.isEmpty || labelList.isEmpty) {
      return Map("AUC" -> 0.0f, "LogLoss" -> 0.0f, "Accuracy" -> 0.0f)
    }

    val minLen = math.min(predList.length, labelList.length)
    val predArray = predList.take(minLen).toArray
    val labelArray = labelList.take(minLen).toArray

    val aucMetric = new AUC()
    val loglossMetric = new LogLoss()
    val accMetric = new Accuracy()
    val hitRateMetric = new HitRate[Float](10)
    val ndcgMetric = new NDCG[Float](10)
    val mrrMetric = new MRR()

    aucMetric.update(predArray, labelArray)
    loglossMetric.update(predArray, labelArray)
    accMetric.update(predArray, labelArray)
    hitRateMetric.update(predArray, labelArray)
    ndcgMetric.update(predArray, labelArray)
    mrrMetric.update(predArray, labelArray)

    Map(
      "AUC" -> aucMetric.compute(),
      "LogLoss" -> loglossMetric.compute(),
      "Accuracy" -> accMetric.compute(),
      "Hit@10" -> hitRateMetric.compute(),
      "NDCG@10" -> ndcgMetric.compute(),
      "MRR" -> mrrMetric.compute()
    )
    } catch {
      case e: Throwable =>
        Map("AUC" -> 0.0f, "LogLoss" -> 0.0f, "Accuracy" -> 0.0f, "Hit@10" -> 0.0f, "NDCG@10" -> 0.0f, "MRR" -> 0.0f)
    }
  }

  def predict(dataLoader: DataLoader): Array[Float] = {
    val predictions = mutable.ListBuffer[Float]()

    val iter = dataLoader.iterator
    while (iter.hasNext) {
      val batch = iter.next()
      val features = batch.sparseFeatures

      model match {
        case deepFM: DeepFM =>
          val pred = deepFM.forward(features).sigmoid()
          val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
          predictions.appendAll(predHost.toFloatArray)
          predHost.close()
        case _ =>
      }
    }

    predictions.toArray
  }

  def saveCheckpoint(path: String): Unit = {
    throw new UnsupportedOperationException("torch.save not available in JavaCPP")
  }

  def loadCheckpoint(path: String): Unit = {
    val loaded = torch.load(path)
  }
}