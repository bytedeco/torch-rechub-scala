package torchrec.trainers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.mutable

import torchrec.Implicits._
import torchrec.data.DataLoader
import torchrec.models.multi_task._
import torchrec.basic.metrics._
import torchrec.basic.losses.{BCELoss, BCEWithLogitsLoss}
import torchrec.utils.DeviceSupport

/**
 * Trainer for multi-task learning models (MMOE, SharedBottom, PLE, AITM, ESMM, OMoE, SingleTaskModel, MetaHeac)
 *
 * Models return either:
 * - Tensor: concatenated along dim=1 (ESMM, MMOE, SharedBottom, PLE, AITM, OMoE, SingleTaskModel)
 * - Map[String, Tensor]: per task name (MetaHeac)
 *
 * The trainer extracts individual task predictions appropriately.
 */
class MTLTrainer(
  model: Module,
  taskNames: List[String],
  learningRate: Float = 1e-3f,
  weightDecay: Float = 1e-6f,
  device: String = DeviceSupport.backend,
  numEpochs: Int = 10,
  earlyStopPatience: Int = 500,
  taskWeights: Option[Map[String, Float]] = None,
  verbose: Boolean = true
) {
  private val optimizer = new Adam(model.parameters(), new AdamOptions(learningRate.toDouble))
  private val bceLoss = new BCEWithLogitsLoss()
  private val defaultWeights = taskWeights.getOrElse(taskNames.map(_ -> 1.0f).toMap)

  def fit(
    trainLoader: DataLoader,
    valLoader: Option[DataLoader] = None
  ): Unit = {
    model.train(true)
    var bestMetric = 0.0f
    var patienceCounter = 0
    val nTask = taskNames.size

    for (epoch <- 0 until numEpochs) {
      var totalLoss = 0.0f
      var numBatches = 0

      val iter = trainLoader.iterator
      while (iter.hasNext) {
        val batch = iter.next()
        val sparseFeats = batch.sparseFeatures
        val taskLabelsMap = batch.taskLabels
        // Debug: print the runtime class of taskLabelsMap and a sample
        if (epoch == 0 && numBatches == 0) {
          try {
            System.err.println(s"[MTLTrainer DEBUG] taskLabelsMap runtime class: ${taskLabelsMap.getClass.getName}")
            if (taskLabelsMap != null) System.err.println(s"[MTLTrainer DEBUG] taskLabelsMap.isEmpty=${taskLabelsMap.isEmpty}")
          } catch { case _: Throwable => () }
        }
        if (sparseFeats.isEmpty || taskLabelsMap.isEmpty) { /* skip */ }
        else {
          optimizer.zero_grad()

          // Get predictions - handle both Tensor and Map[String, Tensor] returns
          val outputsAny = model match {
            case m: MMOE => m.forward(sparseFeats)
            case m: SharedBottom => m.forward(sparseFeats)
            case m: PLE => m.forward(sparseFeats)
            case m: ESMM => m.forward(sparseFeats)
            case m: AITM => m.forward(sparseFeats)
            case m: OMoE => m.forward(sparseFeats)
            case m: SingleTaskModel => m.forward(sparseFeats)
            case m: MetaHeac => m.forwardByName(sparseFeats)
            case _ => throw new IllegalArgumentException(s"Unknown model type: ${model.getClass.getName}")
          }

          var totalWeightedLoss: Tensor = null.asInstanceOf[Tensor]
          var totalWeight = 0.0f

          for (i <- 0 until nTask) {
            val taskName = taskNames(i)
            // Extract prediction for this task - handle Tensor and Map types
            val pred = outputsAny match {
              case tensor: Tensor => tensor.select(1, i)
              case map: java.util.Map[String, Tensor] @unchecked => map.get(taskName)
              case map: Map[String, Tensor] @unchecked => map(taskName)
            }
            val label = taskLabelsMap.get(taskName)
            if (label != null) {
              val actualBatchSize = label.size(0).toInt
              val target2D = if (pred.dim() == 1) pred.reshape(actualBatchSize, 1) else pred
              val label2D = label.view(actualBatchSize, 1).toType(ScalarType.Float)
               val taskLoss = bceLoss.apply(target2D, label2D)
               if (epoch == 0 && numBatches == 0) {
                 try {
                   System.err.println(s"[MTLTrainer DEBUG] epoch=0 first-batch task=$taskName taskLoss=${taskLoss.item().toDouble}")
                 } catch { case _: Throwable => () }
               }
              // Debugging: if loss is zero, print sample preds/labels and dtypes/shapes
              try {
                val lossVal = taskLoss.detach().toType(ScalarType.Float).item().toFloat
                if (lossVal == 0.0f) {
                  System.err.println(s"[MTLTrainer DEBUG] task=$taskName loss=0 -> pred dtype=${target2D.dtype()} shape=${target2D.sizes().toString} label dtype=${label2D.dtype()} shape=${label2D.sizes().toString}")
                  try {
                    val pArr = target2D.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray.take(5)
                    val lArr = label2D.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray.take(5)
                    System.err.println(s"[MTLTrainer DEBUG] sample preds=${pArr.mkString(",")} sample labels=${lArr.mkString(",")}")
                  } catch { case _: Throwable => () }
                }
              } catch { case _: Throwable => () }
              val weight = defaultWeights.getOrElse(taskName, 1.0f).toFloat
              if (totalWeightedLoss == null) {
                totalWeightedLoss = taskLoss.mul(new Scalar(weight.toDouble))
              } else {
                totalWeightedLoss = totalWeightedLoss.add(taskLoss.mul(new Scalar(weight.toDouble)))
              }
              totalWeight += weight
            }
          }

          if (totalWeightedLoss != null && totalWeight > 0) {
            val avgLoss = totalWeightedLoss.div(new Scalar(totalWeight.toDouble))
            // Debug: print first-batch sample values for epoch 0
            if (epoch == 0 && numBatches == 0) {
              try {
                System.err.println(s"[MTLTrainer DEBUG] first-batch avgLoss=${avgLoss.item().toDouble}")
              } catch { case _: Throwable => () }
            }
            avgLoss.backward()
            optimizer.step()
            totalLoss += avgLoss.item().toFloat
          }
          numBatches += 1
        }
      }

      val avgLoss = if (numBatches > 0) totalLoss / numBatches else 0.0f

      if (verbose) print(s"Epoch $epoch: loss=${avgLoss.toString}")

      valLoader.foreach { vl =>
        model.eval()
        val metrics = evaluate(vl)
        val auc = metrics.values.headOption.getOrElse(0.0f)
        if (verbose) print(s", ${taskNames.head}_AUC=${f"$auc%.4f"}")

        if (auc > bestMetric) {
          bestMetric = auc
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
    model.eval()
    val nTask = taskNames.size

    val taskMetrics = mutable.Map[String, (AUC, LogLoss, Accuracy)]()
    for (name <- taskNames) {
      taskMetrics(name) = (new AUC(), new LogLoss(), new Accuracy())
    }

    val iter = dataLoader.iterator
    while (iter.hasNext) {
      val batch = iter.next()
      val sparseFeats = batch.sparseFeatures
      val taskLabelsMap = batch.taskLabels

      if (sparseFeats.isEmpty || taskLabelsMap.isEmpty) { /* skip */ }
      else {
        // Get predictions - handle both Tensor and Map[String, Tensor] returns
        val outputsAny = model match {
          case m: MMOE => m.forward(sparseFeats)
          case m: SharedBottom => m.forward(sparseFeats)
          case m: PLE => m.forward(sparseFeats)
          case m: ESMM => m.forward(sparseFeats)
          case m: AITM => m.forward(sparseFeats)
          case m: OMoE => m.forward(sparseFeats)
          case m: SingleTaskModel => m.forward(sparseFeats)
          case m: MetaHeac => m.forwardByName(sparseFeats)
          case _ => throw new IllegalArgumentException(s"Unknown model type: ${model.getClass.getName}")
        }

        for (i <- 0 until nTask) {
          val taskName = taskNames(i)
          // Extract prediction for this task - handle Tensor and Map types
          val pred = outputsAny match {
            case tensor: Tensor => tensor.select(1, i)
            case map: java.util.Map[String, Tensor] @unchecked => map.get(taskName)
            case map: Map[String, Tensor] @unchecked => map(taskName)
          }
          val label = taskLabelsMap.get(taskName)
          if (label != null) {
            // Convert logits to probabilities for metric computations
            val labelArr = label.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
            val predProbTensor = if (pred != null) pred.sigmoid() else pred
            val predArr = predProbTensor.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
            // Skip empty arrays (can happen for tiny datasets or unexpected shapes)
            if (labelArr.length == 0 || predArr.length == 0) {
              // nothing to update for this batch/task
            } else {
              val (auc, logloss, acc) = taskMetrics(taskName)
              auc.update(predArr, labelArr)
              logloss.update(predArr, labelArr)
              acc.update(predArr, labelArr)
            }
          }
        }
      }
    }

    model.train(true)

    taskMetrics.flatMap { case (name, (auc, logloss, acc)) =>
      Map(
        s"${name}_AUC" -> auc.compute(),
        s"${name}_LogLoss" -> logloss.compute(),
        s"${name}_Accuracy" -> acc.compute()
      )
    }.toMap
  }
}