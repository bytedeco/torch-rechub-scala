package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.utils.DeviceSupport

/**
 * AUGRU Cell - Attention Update Gate GRU Cell
 * Reference: DIEN paper, AAAI 2019, Eq.16
 *
 * Unlike standard GRU, AUGRU has an attention-scaled update gate:
 * u_hat = a * u  (where a is the attention score)
 * h_new = (1 - u_hat) * h + u_hat * h_hat
 */
class AUGRU_Cell(
  embedDim: Int,
  device: String = DeviceSupport.backend
) extends Module {

  // Update gate weights
  private val Wu = new LinearImpl(embedDim, embedDim)
  private val Uu = new LinearImpl(embedDim, embedDim)
  private val bu = torch.zeros(Array(embedDim.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))

  // Reset gate weights
  private val Wr = new LinearImpl(embedDim, embedDim)
  private val Ur = new LinearImpl(embedDim, embedDim)
  private val br = torch.zeros(Array(embedDim.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))

  // Hidden state proposal weights
  private val Wh = new LinearImpl(embedDim, embedDim)
  private val Uh = new LinearImpl(embedDim, embedDim)
  private val bh = torch.zeros(Array(embedDim.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))

  // Register modules
  register_module("Wu", Wu)
  register_module("Uu", Uu)
  register_module("Wr", Wr)
  register_module("Ur", Ur)
  register_module("Wh", Wh)
  register_module("Uh", Uh)

  // Register buffers (biases)
  register_buffer("bu", bu)
  register_buffer("br", br)
  register_buffer("bh", bh)

  // Move to device
  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    this.to(dev, false)
  }

  def forward(x: Tensor, h_1: Tensor, a: Tensor): Tensor = {
    // x: (batch, embed_dim), h_1: (batch, embed_dim), a: (batch, 1) attention score
    // Update gate: u = sigmoid(Wu @ x + Uu @ h_1 + bu)
    val u = torch.sigmoid(Wu.forward(x).add(Uu.forward(h_1)).add(bu))

    // Reset gate: r = sigmoid(Wr @ x + Ur @ h_1 + br)
    val r = torch.sigmoid(Wr.forward(x).add(Ur.forward(h_1)).add(br))

    // Hidden state proposal: h_hat = tanh(Wh @ x + r * (Uh @ h_1) + bh)
    val hUh = Uh.forward(h_1).mul(r)
    val h_hat = torch.tanh(Wh.forward(x).add(hUh).add(bh))

    // Attention-scaled update gate: u_hat = a * u (paper Eq.16)
    val u_hat = a.mul(u)

    // New hidden state: (1 - u_hat) * h_1 + u_hat * h_hat
    val oneMinusUh = torch.ones_like(u_hat).sub(u_hat)
    oneMinusUh.mul(h_1).add(u_hat.mul(h_hat))
  }
}

/**
 * AUGRU - Attention Update Gate GRU
 * Implements interest evolving layer from DIEN
 */
class AUGRU(
  embedDim: Int,
  device: String = DeviceSupport.backend
) extends Module {

  val embed_dim: Int = embedDim
  private val augruCell = new AUGRU_Cell(embedDim, device)
  register_module("augru_cell", augruCell)

  // Attention projection matrix
  private val Wa = new LinearImpl(embedDim, embedDim)
  register_module("Wa", Wa)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    this.to(dev, false)
  }

  def forward(x: Tensor, item: Tensor, mask: Tensor): (Tensor, Tensor) = {
    // Compute attention scores
    val waOut = Wa.forward(x)
    val itemUnsq = item.unsqueeze(1)
    val scores = waOut.mul(itemUnsq).sum(2)

    // Apply mask
    val maskedScores = scores.masked_fill(mask.bitwise_not(), new Scalar(Float.NegativeInfinity))

    // Softmax
    var attn = torch.softmax(maskedScores, 1)

    // Handle NaN rows
    val nanRows = attn.isnan().any(1)
    if (nanRows.count_nonzero().item().toInt() > 0) {
      val numValid = mask.sum(1).unsqueeze(1)
      attn = torch.where(nanRows.unsqueeze(1), torch.ones_like(attn).div(numValid), attn)
    }

    attn = attn.unsqueeze(2)

    // Initialize hidden state
    val batchSize = x.size(0).toInt
    val h = torch.zeros(Array(batchSize.toLong, embedDim.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(x.device(), ScalarType.Float)

    // Run AUGRU cell step by step
    val outs = scala.collection.mutable.ListBuffer[Tensor]()
    for (i <- 0 until x.size(1).toInt) {
      val stepInput = x.select(1, i)
      val stepAttn = attn.select(1, i)
      val newH = augruCell.forward(stepInput, h, stepAttn)
      outs += newH.unsqueeze(1)
    }

    val outputTensor = torch.cat(new TensorVector(outs.toSeq*), 1)
    (outputTensor, h)
  }
}