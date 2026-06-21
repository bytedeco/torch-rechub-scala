package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import torchrec.utils.DeviceSupport

/**
 * Squeeze-and-Excitation Network Layer
 * Used in FiBiNet for feature gating
 */
class SENETLayer(
  numFields: Int,
  reduction: Int = 3,
  device: String = DeviceSupport.backend
) extends Module {

  private var inputDim: Long = 0

  // Lazy layers
  private var _fc1: Option[LinearImpl] = None
  private var _fc2: Option[LinearImpl] = None

  private def ensureInitialized(inDim: Long): Unit = {
    if (inDim <= 0) {
      throw new IllegalArgumentException(s"SENETLayer received invalid inDim=$inDim. numFields must be > 0.")
    }
    if (_fc1.isEmpty || inputDim != inDim) {
      _fc1.foreach(_.close())
      _fc2.foreach(_.close())

      inputDim = inDim
      val reducedDim = math.max(1, inDim / reduction).toLong

      val fc1New = new LinearImpl(inDim, reducedDim)
      register_module("fc1", fc1New)
      val fc2New = new LinearImpl(reducedDim, inDim)
      register_module("fc2", fc2New)

      val dev = new org.bytedeco.pytorch.Device(device)
      fc1New.to(dev, false)
      fc2New.to(dev, false)

      _fc1 = Some(fc1New)
      _fc2 = Some(fc2New)
    }
  }

  def setInputDim(dim: Long): this.type = {
    ensureInitialized(dim)
    this
  }

  def forward(x: Tensor): Tensor = {
    // x: (batch_size, num_fields, embed_dim)
    val inDim = x.size(2)
    ensureInitialized(inDim)

    val batchSize = x.size(0)
    val embedDim = x.size(2)

    // Squeeze: global average pooling
    val squeezed = x.mean(1) // (batch, embed_dim)

    // Excitation: FC -> ReLU -> FC -> Sigmoid
    val excited = _fc1.get.forward(squeezed).relu()
    val out = _fc2.get.forward(excited).sigmoid()

    // Scale: multiply back to original shape
    x.mul(out.view(batchSize, 1, embedDim))
  }
}

/**
 * Context-Adjusted SE (CARD) - SE with context vector
 */
class ContextSENET(
  reduction: Int = 3,
  device: String = DeviceSupport.backend
) extends Module {

  private var inputDim: Long = 0
  private var contextDim: Long = 0

  // Lazy layers
  private var _context_proj: Option[LinearImpl] = None
  private var _fc1: Option[LinearImpl] = None
  private var _fc2: Option[LinearImpl] = None

  def setInputDim(dim: Long, ctxDim: Long): this.type = {
    if (_context_proj.isEmpty || inputDim != dim || contextDim != ctxDim) {
      _context_proj.foreach(_.close())
      _fc1.foreach(_.close())
      _fc2.foreach(_.close())

      inputDim = dim
      contextDim = ctxDim
      val reducedDim = math.max(1, dim / reduction).toLong

      val cp = new LinearImpl(ctxDim, dim)
      register_module("context_proj", cp)
      val fc1New = new LinearImpl(dim, reducedDim)
      register_module("fc1", fc1New)
      val fc2New = new LinearImpl(reducedDim, dim)
      register_module("fc2", fc2New)

      val dev = new org.bytedeco.pytorch.Device(device)
      cp.to(dev, false)
      fc1New.to(dev, false)
      fc2New.to(dev, false)

      _context_proj = Some(cp)
      _fc1 = Some(fc1New)
      _fc2 = Some(fc2New)
    }
    this
  }

  def forward(x: Tensor, context: Tensor): Tensor = {
    // x: (batch, num_fields, embed)
    // context: (batch, context_dim)
    val inDim = x.size(2)
    val ctxDim = context.size(1)
    if (_context_proj.isEmpty) setInputDim(inDim, ctxDim)

    val batchSize = x.size(0)
    val embedDim = x.size(2)

    // Project context to field dimension
    val projected = _context_proj.get.forward(context) // (batch, embed)
    val weighted = x.mul(projected.view(batchSize, 1, embedDim))

    // Squeeze
    val squeezed = weighted.mean(1)

    // Excitation
    val excited = _fc1.get.forward(squeezed).relu()
    val out = _fc2.get.forward(excited).sigmoid()

    weighted.mul(out.view(batchSize, 1, embedDim))
  }
}
