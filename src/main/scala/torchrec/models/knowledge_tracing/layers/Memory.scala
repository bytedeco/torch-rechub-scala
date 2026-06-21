package torchrec.models.knowledge_tracing.layers

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.utils.DeviceSupport

/**
 * Erase-and-Add gate used by DKVMN, GKT.
 */
class EraseAddGate(
  dim: Int,
  device: String = DeviceSupport.backend
) extends Module {

  private val erase = new LinearImpl(dim, dim)
  private val add = new LinearImpl(dim, dim)
  private val weight = {
    val w = torch.ones(Array(dim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    register_parameter("weight", w)
    w
  }

  register_module("erase", erase)
  register_module("add", add)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    erase.to(dev, false); add.to(dev, false)
  }

  def forward(x: Tensor, eraseVector: Tensor, addVector: Tensor, attention: Tensor): Tensor = {
    // x: (batch, seq, dim) or (batch, dim)
    // eraseVector: (batch, seq, dim) or (batch, dim)
    // addVector: (batch, seq, dim) or (batch, dim)
    // attention: (batch, seq) or (batch,)
    val eraseGate = torch.sigmoid(erase.forward(eraseVector))
    val addGate = torch.tanh(add.forward(addVector))

    val expandedAttn = if (attention.dim() == 2L) {
      attention.unsqueeze(2)
    } else {
      attention.unsqueeze(1)
    }

    // Memory update: M_new = M * (1 - attn * erase_gate) + attn * add_gate
    val erased = x.mul(x.neg().add(new Scalar(1.0)).mul(expandedAttn.unsqueeze(2).mul(eraseGate)))
    erased.add(expandedAttn.unsqueeze(2).mul(addGate))
  }
}

/**
 * Key-Value Memory for DKVMN and DeepIRT.
 */
class KeyValueMemory(
  numSlots: Int,
  keyDim: Int,
  valueDim: Int,
  device: String = DeviceSupport.backend
) extends Module {

  // Key memory matrix: (numSlots, keyDim)
  private val keyMemory = {
    val init = torch.randn(Array(numSlots.toLong, keyDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .mul(new Scalar(scala.math.sqrt(1.0 / keyDim.toDouble).toFloat))
    register_parameter("key_memory", init)
    init
  }

  // Value memory: (numSlots, valueDim)
  private val valueMemoryInit = {
    val init = torch.zeros(Array(numSlots.toLong, valueDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    register_parameter("value_memory_init", init)
    init
  }

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    this.to(dev, false)
  }

  /**
   * Compute attention weights over memory slots.
   * @param key Input key tensor (batch, seq, keyDim)
   * @return Attention weights (batch, seq, numSlots)
   */
  def computeAttention(key: Tensor): Tensor = {
    // (batch, seq, numSlots, 1) - each slot
    val km = keyMemory.unsqueeze(0).unsqueeze(2)  // (1, keyDim, numSlots) -> need (1, 1, numSlots, keyDim)
    val kmT = keyMemory.view(1L, 1L, numSlots.toLong, keyDim.toLong)  // (1, 1, numSlots, keyDim)
    val key4d = key.unsqueeze(2)  // (batch, seq, 1, keyDim)

    // Compute similarity: (batch, seq, numSlots)
    val scores = torch.matmul(key4d, kmT.transpose(2, 3)).squeeze(3)

    // Softmax over slots
    scores.softmax(2)
  }

  /**
   * Read from memory using attention weights.
   * @param contentKey Content key for reading (batch, seq, keyDim)
   * @param attn Attention weights (batch, seq, numSlots)
   * @param valueMem Current value memory (batch, numSlots, valueDim)
   * @return Read output (batch, seq, valueDim)
   */
  def read(contentKey: Tensor, attn: Tensor, valueMem: Tensor): Tensor = {
    // valueMem: (batch, numSlots, valueDim)
    // attn: (batch, seq, numSlots)
    val attnEx = attn.unsqueeze(3)  // (batch, seq, numSlots, 1)
    valueMem.unsqueeze(0).mul(attnEx).sum(2)  // (batch, seq, valueDim)
  }

  /**
   * Write to memory using attention weights.
   * @param contentValue Content value to write (batch, seq, valueDim)
   * @param attn Attention weights (batch, seq, numSlots)
   * @param currentMem Current value memory (batch, numSlots, valueDim)
   * @param eraseVector Erase vector (batch, seq, valueDim)
   * @return Updated value memory (batch, numSlots, valueDim)
   */
  def write(
    contentValue: Tensor,
    attn: Tensor,
    currentMem: Tensor,
    eraseVector: Tensor
  ): Tensor = {
    // Erase: currentMem * (1 - attn_mean * erase_vector)
    val attnMean = attn.mean(1).unsqueeze(2)  // (batch, numSlots, 1)
    val eraseGate = torch.sigmoid(eraseVector.mean(1)).unsqueeze(1)  // (batch, 1, valueDim)

    val erased = currentMem.mul(attnMean.mul(eraseGate).neg().add(new Scalar(1.0)))

    // Add: attn_mean * content_value
    val addValue = attnMean.mul(contentValue.mean(1).unsqueeze(1))  // (batch, numSlots, valueDim)

    erased.add(addValue)
  }

  def getInitialMemory(): Tensor = valueMemoryInit
  def getKeyMemory(): Tensor = keyMemory
}
