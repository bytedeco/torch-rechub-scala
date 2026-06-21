package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.utils.DeviceSupport

/**
 * Regulation Module for EDCN
 * Reference: EDCN paper, KDD 2021
 *
 * Provides learnable per-field gating mechanism with temperature-controlled softmax.
 * Splits input by fields, applies field-specific gates, and returns two gated outputs.
 *
 * @param numFields Number of feature fields
 * @param feaDims List of embedding dimensions per field
 * @param tau Temperature coefficient for softmax (default=1.0)
 * @param useRegulation Whether to use regulation (default=true)
 */
class RegulationModule(
  numFields: Int,
  feaDims: List[Int],
  tau: Float = 1.0f,
  useRegulation: Boolean = true
) extends Module {

  if (useRegulation) {
    require(feaDims.size == numFields, s"feaDims size ${feaDims.size} must match numFields $numFields")
  }

  // Learnable gate parameters
  // g1 and g2 are per-field scalars (initialized to 1.0)
  private val g1 = if (useRegulation) {
    torch.ones(Array(numFields.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  } else {
    torch.zeros(Array(numFields.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  }
  private val g2 = if (useRegulation) {
    torch.ones(Array(numFields.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  } else {
    torch.zeros(Array(numFields.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  }

  if (useRegulation) {
    register_parameter("g1", g1)
    register_parameter("g2", g2)
  }

  private val device: String = DeviceSupport.backend

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    this.to(dev, false)
  }

  /**
   * Forward pass of RegulationModule
   *
   * @param x Input tensor (B, total_dim) where total_dim = sum(feaDims)
   * @return (out1, out2) tuple of gated tensors (B, total_dim)
   */
  def forward(x: Tensor): (Tensor, Tensor) = {
    if (!useRegulation) {
      return (x, x)
    }

    // Compute softmax gates with temperature
    // g1_scaled[i] = (g1[i] / tau).softmax()
    // g2_scaled[i] = (g2[i] / tau).softmax()
    val tauTensor = new Scalar(tau.toFloat)
    val g1Scaled = g1.div(tauTensor).softmax(0)
    val g2Scaled = g2.div(tauTensor).softmax(0)

    // Build per-field scaling tensors
    // Each field's embedding dim may differ, so we need to expand appropriately
    val g1List = scala.collection.mutable.ListBuffer[Tensor]()
    val g2List = scala.collection.mutable.ListBuffer[Tensor]()

    for (i <- 0 until numFields) {
      val fieldG1 = g1Scaled.select(0, i).unsqueeze(0)  // (1,)
      val fieldG2 = g2Scaled.select(0, i).unsqueeze(0)  // (1,)
      // Use expand to build per-field scaling tensors to avoid repeat dimension issues
      val repeatedG1 = fieldG1.expand(1, feaDims(i))
      val repeatedG2 = fieldG2.expand(1, feaDims(i))
      g1List += repeatedG1
      g2List += repeatedG2
    }

    val g1Tensor = torch.cat(new TensorVector(g1List.toSeq*), 1)  // (1, total_dim)
    val g2Tensor = torch.cat(new TensorVector(g2List.toSeq*), 1)  // (1, total_dim)

    // Broadcast to batch size and apply to input
    val batchSize = x.size(0)
    val g1Broadcast = g1Tensor.expand(batchSize, -1).to(x.device(), ScalarType.Float)
    val g2Broadcast = g2Tensor.expand(batchSize, -1).to(x.device(), ScalarType.Float)

    val out1 = g1Broadcast.mul(x)
    val out2 = g2Broadcast.mul(x)

    (out1, out2)
  }
}