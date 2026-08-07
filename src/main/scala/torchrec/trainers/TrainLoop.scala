package torchrec.trainers

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

import scala.collection.mutable

import torchrec.data.DataLoader
import torchrec.utils.DeviceSupport
import torchrec.Implicits._
import org.bytedeco.pytorch.optim.options.AdamOptions
/**
 * Generic training loop with callbacks support.
 * Works with any model that accepts a Batch and returns a Tensor.
 */
class TrainLoop(
  model: Module,
  optimizer: Optimizer,
  device: String = DeviceSupport.backend,
  gradientClip: Option[Float] = None
) {
  private var trainStep = 0

  /**
   * Single training step.
   * @param batch The input batch (Any to allow flexibility; cast inside lossFn)
   * @param lossFn Function that computes the loss from a batch
   * @return (loss tensor, metrics map)
   */
  def step(batch: Any, lossFn: Any => Tensor): (Tensor, Map[String, Float]) = {
    optimizer.zero_grad()
    val loss = lossFn(batch)
    loss.backward()

    // Gradient clipping
    gradientClip.foreach { clipValue =>
      val dev = new Device(device)
      var begin = model.parameters().begin()
      var end = model.parameters().end()
       while (!begin.equals(end)) {
         val p = begin.get()
         if (p.grad() != null) {
           val grad = p.grad()
           val norm = grad.norm()
           if (norm.itemSafe().toFloat > clipValue) {
             grad.div_(new Scalar((norm.itemSafe() / clipValue)))
           }
         }
         begin.increment()
       }

    }

    optimizer.step()
    trainStep += 1
    val metrics = Map("loss" -> loss.itemSafe().toFloat)
    (loss, metrics)
  }

  /**
   * Train one epoch over a data loader.
   * @param dataLoader The training data loader
   * @param lossFn Function that computes the loss from a batch
   * @return Aggregated metrics for the epoch
   */
  def trainEpoch(
    dataLoader: DataLoader,
    lossFn: DataLoader => Any => Tensor
  ): Map[String, Float] = {
    model.train(true)
    val totalLoss = mutable.Map[String, Double]()
    var numBatches = 0

    val lossFnPerBatch = lossFn(dataLoader)

    val iter = dataLoader.iterator
    while (iter.hasNext) {
      val batch = iter.next()
      optimizer.zero_grad()
      val loss = lossFnPerBatch(batch)
      loss.backward()

      gradientClip.foreach { clipValue =>
        var begin = model.parameters().begin()
        var end = model.parameters().end()
       while(!begin.equals(end)){
         val p = begin.get()
         if (p.grad() != null) {
           val grad = p.grad()
           val norm = grad.norm()
           if (norm.itemSafe().toFloat > clipValue) {
             grad.div_(new Scalar((norm.itemSafe() / clipValue)))
           }
         }
         begin.increment()
       }

      }

      optimizer.step()
      trainStep += 1

      val batchLoss = loss.itemSafe().toFloat
      totalLoss("loss") = totalLoss.getOrElse("loss", 0.0) + batchLoss
      numBatches += 1
    }

    if (numBatches == 0) Map("loss" -> 0.0f)
    else Map("loss" -> (totalLoss("loss") / numBatches).toFloat)
  }

  /**
   * Evaluate a data loader with a provided evaluation function.
   * @param dataLoader The evaluation data loader
   * @param evalFn Function that computes metrics from a batch, returns Map[metricName, Float]
   * @return Aggregated metrics across all batches
   */
  def evaluate(
    dataLoader: DataLoader,
    evalFn: DataLoader => Any => Map[String, Float]
  ): Map[String, Float] = {
    model.eval()
    val accumulators = mutable.Map[String, Double]()
    var numBatches = 0

    val evalFnPerBatch = evalFn(dataLoader)

    val iter = dataLoader.iterator
    while (iter.hasNext) {
      val batch = iter.next()
      val batchMetrics = evalFnPerBatch(batch)
      for ((name, value) <- batchMetrics) {
        accumulators(name) = accumulators.getOrElse(name, 0.0) + value
      }
      numBatches += 1
    }

    model.train(true)

    if (numBatches == 0) Map.empty
    else accumulators.map { case (name, sum) => name -> (sum / numBatches).toFloat }.toMap
  }

  def getTrainStep: Int = trainStep
  def reset(): Unit = { trainStep = 0 }
}
