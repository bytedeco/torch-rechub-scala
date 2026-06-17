package torchrec.basic.layers

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch

/**
 * Prediction layer.
 *
 * Parameters
 * ----------
 * taskType : {'classification', 'regression'}
 *     Classification applies sigmoid to logits; regression returns logits.
 *
 * Shape
 * -----
 * Input: ``(B, *)``
 * Output: ``(B, *)`` with sigmoid applied if classification, otherwise unchanged.
 */
class PredictionLayer(
  taskType: String = "classification"
) extends Module {

  if (taskType != "classification" && taskType != "regression") {
    throw new IllegalArgumentException("taskType must be classification or regression")
  }

  def forward(x: Tensor): Tensor = {
    if (taskType == "classification") {
      torch.sigmoid(x)
    } else {
      x
    }
  }
}

/**
 * PredictionLayer companion object with factory methods.
 */
object PredictionLayer {
  def apply(taskType: String = "classification"): PredictionLayer = {
    new PredictionLayer(taskType)
  }
}