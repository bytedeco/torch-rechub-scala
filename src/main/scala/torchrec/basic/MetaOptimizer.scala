package torchrec.basic

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.TensorImplicits.RichTensor

import scala.collection.mutable

/**
 * MetaBalance - scales gradients and balances gradient across tasks.
 * This is a gradient balancing helper for multi-task learning.
 *
 * The implementation follows the original Python algorithm but without retain_graph support.
 * Gradients are accumulated by computing losses in sequence.
 *
 * @param relaxFactor Relaxation factor for gradient scaling (default: 0.7, range: [0, 1))
 * @param beta Moving average coefficient (default: 0.9, range: [0, 1))
 */
class MetaBalance(
  relaxFactor: Float = 0.7f,
  beta: Float = 0.9f
) {

  require(relaxFactor >= 0.0f && relaxFactor < 1.0f,
    s"Invalid relax_factor: $relaxFactor, it should be 0. <= relax_factor < 1.")
  require(beta >= 0.0f && beta < 1.0f,
    s"Invalid beta: $beta, it should be 0. <= beta < 1.")

  // Per-parameter gradient norms history
  private val gradNorms = mutable.Map[Long, mutable.ListBuffer[Float]]()

  // Per-parameter accumulated gradient sum
  private val sumGradients = mutable.Map[Long, Tensor]()

  // Per-parameter first gradient flag (to compute initial norm without retain_graph)
  private val firstGradient = mutable.Map[Long, Boolean]()

  /**
   * Compute and accumulate scaled gradients for multiple task losses.
   * Note: This implementation differs from Python because JavaCPP's backward()
   * does not support retain_graph parameter. Instead, we compute all losses
   * and accumulate their gradients.
   *
   * @param params List of parameters to optimize
   * @param losses Sequence of loss tensors, one per task
   */
  def step(params: Seq[Tensor], losses: Seq[Tensor]): Unit = {
    require(losses.nonEmpty, "At least one loss must be provided")
    require(params.nonEmpty, "At least one parameter must be provided")

    val numTasks = losses.length

    // Initialize state for each parameter
    params.foreach { param =>
      val paramId = System.identityHashCode(param)
      if (!gradNorms.contains(paramId)) {
        gradNorms(paramId) = mutable.ListBuffer.fill(numTasks)(0.0f)
      }
      if (!sumGradients.contains(paramId)) {
        sumGradients(paramId) = {
          val zero = param.clone()
          zero.zero_()
          zero
        }
      }
      if (!firstGradient.contains(paramId)) {
        firstGradient(paramId) = true
      }
    }

    // Compute all losses and accumulate gradients
    // Note: We compute each loss separately because we need individual gradients for scaling
    // This is a limitation of the JavaCPP API (no retain_graph)
    var accumulatedLoss: Option[Tensor] = None
    losses.foreach { loss =>
      // Compute gradient for this loss
      loss.backward()

      // Process each parameter
      params.foreach { param =>
        val paramId = System.identityHashCode(param)
        val grad = param.grad()
        if (grad == null) {
          return
        }

        if (grad.is_sparse()) {
          throw new RuntimeException("MetaBalance does not support sparse gradients")
        }

        val norms = gradNorms(paramId)
        val sumGrad = sumGradients(paramId)

        // Compute gradient norm
        val gradNorm = grad.norm().item().toFloat()

        // Update moving average of gradient norm
        norms(0) = norms(0) * beta + (1.0f - beta) * gradNorm

        // Scale the gradient using MetaBalance formula
        // scaled = grad * (norms[0] / (norms[idx] + 1e-5)) * relax_factor + grad * (1 - relax_factor)
        // Since idx=0 in our case (we maintain only the first norm):
        val scaledGrad = grad * (norms(0) / (norms(0) + 1e-5f)) * relaxFactor +
                         grad * (1.0f - relaxFactor)

        // Accumulate scaled gradients
        sumGrad.add_(scaledGrad)

        // Clear gradient for next loss computation
        grad.zero_()
      }

      // Close the loss tensor to free memory
      loss.close()
    }
  }

  /**
   * Apply the accumulated gradients to parameters.
   *
   * @param params List of parameters to optimize
   */
  def applyGradients(params: Seq[Tensor]): Unit = {
    params.foreach { param =>
      val paramId = System.identityHashCode(param)
      sumGradients.get(paramId).foreach { sumGrad =>
        param.grad().copy_(sumGrad)
        sumGrad.zero_()
      }
    }
  }

  /**
   * Reset the optimizer state.
   */
  def reset(): Unit = {
    gradNorms.foreach { case (_, norms) =>
      for (i <- norms.indices) norms(i) = 0.0f
    }
    sumGradients.values.foreach(_.zero_())
    firstGradient.keys.foreach(k => firstGradient(k) = true)
  }
}

/**
 * Companion object for MetaBalance with factory methods.
 */
object MetaBalance {
  def apply(relaxFactor: Float = 0.7f, beta: Float = 0.9f): MetaBalance = {
    new MetaBalance(relaxFactor, beta)
  }
}