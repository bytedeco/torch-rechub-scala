package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import torchrec.utils.DeviceSupport

/**
 * Multi-head Self-Attention based Interacting Layer, used in AutoInt model.
 *
 * Parameters
 * ----------
 * embedDim : int
 *   The embedding dimension.
 * numHeads : int
 *   The number of attention heads (default=2).
 * dropout : float
 *   The dropout rate (default=0.0).
 * residual : bool
 *   Whether to use residual connection (default=True).
 *
 * Shape
 * -----
 * Input: (batch_size, num_fields, embed_dim)
 * Output: (batch_size, num_fields, embed_dim)
 */
class InteractingLayer(
  embedDim: Int,
  numHeads: Int = 2,
  dropout: Float = 0.0f,
  residual: Boolean = true,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0, s"embed_dim ($embedDim) must be divisible by num_heads ($numHeads)")

  private val headDim = embedDim / numHeads
  private val scale = 1.0f / math.sqrt(headDim).toFloat

  private val wQ = new LinearImpl(embedDim, embedDim)
  register_module("W_Q", wQ)
  wQ.to(new org.bytedeco.pytorch.Device(device), false)

  private val wK = new LinearImpl(embedDim, embedDim)
  register_module("W_K", wK)
  wK.to(new org.bytedeco.pytorch.Device(device), false)

  private val wV = new LinearImpl(embedDim, embedDim)
  register_module("W_V", wV)
  wV.to(new org.bytedeco.pytorch.Device(device), false)

  private val wRes = if (residual) {
    val r = new LinearImpl(embedDim, embedDim)
    register_module("W_Res", r)
    r.to(new org.bytedeco.pytorch.Device(device), false)
    Some(r)
  } else None

  private val dropoutLayer = if (dropout > 0) {
    val d = new DropoutImpl(dropout)
    Some(d)
  } else None

  def forward(x: Tensor): Tensor = {
    val batchSize = x.size(0).toInt
    val numFields = x.size(1).toInt

    // Linear projections
    val Q = wQ.forward(x)
    val K = wK.forward(x)
    val V = wV.forward(x)

    // Reshape for multi-head attention: (batch, num_heads, num_fields, head_dim)
    val qReshaped = Q.reshape(batchSize, numFields, numHeads, headDim).transpose(1, 2)
    val kReshaped = K.reshape(batchSize, numFields, numHeads, headDim).transpose(1, 2)
    val vReshaped = V.reshape(batchSize, numFields, numHeads, headDim).transpose(1, 2)

    // Scaled dot-product attention
    val attnScores = torch.matmul(qReshaped, kReshaped.transpose(-2, -1)).mul(new Scalar(scale))
    val attnWeights = attnScores.softmax(-1)

    val finalWeights = if (dropoutLayer.isDefined) {
      dropoutLayer.get.forward(attnWeights)
    } else {
      attnWeights
    }

    // Apply attention to values
    val attnOutput = torch.matmul(finalWeights, vReshaped)

    // Concatenate heads: (batch, num_fields, embed_dim)
    val attnOutputFinal = attnOutput.transpose(1, 2).contiguous().reshape(batchSize, numFields, embedDim)

    // Residual connection
    val withResidual = if (wRes.isDefined) {
      attnOutputFinal.add(wRes.get.forward(x))
    } else {
      attnOutputFinal
    }

    torch.relu(withResidual)
  }
}

/**
 * InteractingLayer companion object with factory methods.
 */
object InteractingLayer {
  def apply(
    embedDim: Int,
    numHeads: Int = 2,
    dropout: Float = 0.0f,
    residual: Boolean = true,
    device: String = DeviceSupport.backend
  ): InteractingLayer = {
    new InteractingLayer(embedDim, numHeads, dropout, residual, device)
  }
}