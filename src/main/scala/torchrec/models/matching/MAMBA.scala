package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.mutable

/**
 * MAMBA: State Space Model for Sequential Recommendation
 *
 * A simplified implementation of the Mamba/SSM architecture for recommendation.
 * Uses a selective state space model with input-dependent dynamics.
 * This is a Scala 3 adaptation of the Mamba SSM architecture.
 *
 * Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
 *            (Gu & Dao, 2023) -- adapted for recommendation.
 *
 * Architecture:
 *   Token Input → Input Projection (x, z) → Conv1D (local context)
 *   → SSM Dynamics (A, B, C, dt) → Output Projection → [CLS] pooling → MLP
 *
 * @param vocabSize         Item vocabulary size
 * @param embedDim          Embedding dimension
 * @param dState            SSM state dimension (default 16)
 * @param numLayers         Number of MAMBA layers
 * @param maxSeqLen         Maximum sequence length
 * @param mlpDims           MLP hidden dimensions
 * @param dropout           Dropout rate
 * @param device            Device
 */
class MAMBA(
  vocabSize: Long,
  embedDim: Int = 64,
  dState: Int = 16,
  numLayers: Int = 2,
  maxSeqLen: Int = 50,
  mlpDims: List[Long] = List(128L, 64L),
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  private val dInner = embedDim * 2  // Inner dimension

  // Token embedding
  private val tokenEmbedding = new EmbeddingImpl(new EmbeddingOptions(vocabSize, embedDim))
  tokenEmbedding.to(new Device(device),false)
  register_module("tokenEmbedding", tokenEmbedding)

  // Learned [CLS] token
  private val clsTensor = {
    val t = torch.zeros(
      Array(1L, 1L, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    t.fill_(new Scalar(0.02f))
    if (device != "cpu") {
      val dev = new org.bytedeco.pytorch.Device(device)
      t.to(dev, ScalarType.Float)
    } else t
  }
  register_parameter("clsToken", clsTensor)

  // MAMBA layers
  private val layers = (0 until numLayers).map { i =>
    val layer = new MAMBABlock(embedDim, dState, dropout, device)
    register_module(s"mamba_$i", layer)
    layer
  }

  // Layer norm over last dim
  private val layerNorm = {
    val ln = new LayerNormImpl(new LongVector(Array(embedDim.toLong) *))
    register_module("layerNorm", ln)
    ln
  }

  // Dropout
  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropout", dropoutLayer)

  // Final MLP
  private val mlp = new MLP(embedDim, mlpDims, 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    tokenEmbedding.to(dev, false)
    layers.foreach(_.to(dev, false))
    layerNorm.to(dev, false)
    dropoutLayer.to(dev, false)
    mlp.to(dev, false)
  }

  def forward(
    seqTokens: Tensor,   // (batch, seq_len) -- item IDs
    positions: Tensor     // (batch, seq_len) -- position indices
  ): Tensor = {
    val batchSize = seqTokens.size(0).toInt

    // Token embeddings: (batch, seq_len, embed_dim)
    val tokenEmb = tokenEmbedding.forward(seqTokens.toType(ScalarType.Long))

    // Prepend [CLS]
    val clsBatched = clsTensor.expand(batchSize.toLong, 1L, embedDim.toLong)
    val tokenEmbWithCls = torch.cat(new TensorVector(clsBatched, tokenEmb), 1)
    println("  [Model] MAMBA forward .... ")

    // MAMBA layers
    var x = tokenEmbWithCls
    x = dropoutLayer.forward(x)
    layers.foreach { layer =>
      println("  [Model] MAMBA layer foreach ....")

      x = layer.forward(x).add(x)  // Residual
    }

    // Layer norm
    val normed = layerNorm.forward(x)  // (batch, seq_len+1, embed_dim)

    // Extract [CLS] representation
    val clsRep = normed.select(1, 0)  // (batch, embed_dim)

    // MLP
    val logits = mlp.forward(clsRep)  // (batch, 1)
    logits
  }
}

/**
 * A single MAMBA Block with simplified SSM dynamics.
 *
 * The key insight of Mamba: use input-dependent (selective) SSM parameters.
 * Instead of using fixed A, B, C matrices, they are computed from the input.
 *
 * Structure:
 *   LayerNorm → Input Projection (x, z) → SSM-inspired Recurrence → Gated Residual → Linear
 */
class MAMBABlock(
  embedDim: Int,
  dState: Int,
  dropout: Float,
  device: String
) extends Module {

  private val dInner = embedDim  // dInner == embedDim (SSM state matches embedding)

  // Input projection: embedDim -> dInner * 2 (for x channel and gate)
  private val xProj = new LinearImpl(embedDim, dInner * 2)
  xProj.to(new Device(device),false)
  register_module("xProj", xProj)

  // SSM parameters projection: dInner -> dt_rank + 2*dState
  private val dtRank = scala.math.ceil(embedDim / 16.0).toInt
  private val ssmProj = new LinearImpl(dInner, dtRank + 2 * dState)
  ssmProj.to(new Device(device),false)
  register_module("ssmProj", ssmProj)

  // dt projection: dt_rank -> dInner
  private val dtProj = new LinearImpl(dtRank, dInner)
  dtProj.to(new Device(device),false)
  register_module("dtProj", dtProj)

  // SSM D parameter (skip connection) - must match dInner exactly
  private val dParam = {
    val t = torch.ones(Array(dInner.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    register_parameter("dParam", t)
    t
  }

  // Hidden state projection: dInner -> dState (for SSM recurrence)
  private val hProj = new LinearImpl(dInner, dState)
  hProj.to(new Device(device),false)
  register_module("hProj", hProj)

  // Hidden state output projection: dState -> dInner (after SSM recurrence)
  private val hProjOut = new LinearImpl(dState, dInner)
  hProjOut.to(new Device(device),false)
  register_module("hProjOut", hProjOut)

  // Output projection: dInner -> embedDim
  private val outProj = new LinearImpl(dInner, embedDim)
  outProj.to(new Device(device),false)
  register_module("outProj", outProj)

  // Layer norm over embed_dim
  private val norm = {
    val ln = new LayerNormImpl(new LongVector(Array(embedDim.toLong) *))
    register_module("norm", ln)
    ln
  }

  // Dropout
  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropout", dropoutLayer)

  // A matrix (SSM state transition) - initialized to negative for stability
  // Shape: [dState, dState].
  private val aLog = {
    val vals = (0 until dState * dState).map { i =>
      val d = i % dState + 1
      -math.log(d.toFloat + 1.0f)
    }.toArray
    val t = torch.tensor(vals, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .view(dState.toLong, dState.toLong)
    register_parameter("aLog", t)
    t
  }

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    xProj.to(dev, false)
    ssmProj.to(dev, false)
    dtProj.to(dev, false)
    outProj.to(dev, false)
    norm.to(dev, false)
    dropoutLayer.to(dev, false)
    aLog.to(dev, ScalarType.Float)
    hProj.to(dev, false)
    hProjOut.to(dev, false)
  }

  def forward(x: Tensor): Tensor = {
    // x: (batch, seq_len, embed_dim)
    val batchSize = x.size(0).toInt
    val seqLen = x.size(1).toInt

    // Layer norm
    val normed = norm.forward(x)  // (batch, seq_len, embed_dim)

    // Input projection: (batch, seq_len, embed_dim) -> (batch, seq_len, dInner*2)
    val xz = xProj.forward(normed)  // [batch, seq, dInner*2]
    val xHalf = xz.narrow(2, 0, dInner)    // [batch, seq, dInner]
    val zHalf = xz.narrow(2, dInner, dInner) // [batch, seq, dInner]
    val gate = zHalf.sigmoid()  // [batch, seq, dInner]

    // Local SSM-inspired dynamics: simplified selective SSM
    // Project to SSM parameters
    val ssmParams = ssmProj.forward(xHalf)  // (batch, seq_len, dt_rank + 2*dState)

    val dtPart = ssmParams.narrow(2, 0, dtRank)
    val bPart = ssmParams.narrow(2, dtRank, dState)
    val cPart = ssmParams.narrow(2, dtRank + dState, dState)

    // dt: discretize using SiLU
    val dt = dtProj.forward(dtPart)  // (batch, seq_len, dInner)
    val dtAct = torch.silu(dt)  // SiLU discretization

    // A matrix (from log A)
    val a = torch.exp(aLog)  // (dState, dState) - always positive

    // Project xHalf to SSM state dimension dState for recurrence
    val hInput = hProj.forward(xHalf)  // (batch, seq_len, dState)

    // Simplified SSM: element-wise gating with A-decay
    // Use recurrent approximation: h_t = a_avg * h_{t-1} + x_t
    val aAvg = a.mean(0).unsqueeze(0)  // (1, dState) — average A across rows
    val aRep = aAvg.expand(batchSize.toLong, seqLen.toLong, dState.toLong)

    // Recurrence: apply A decay along sequence
    val hInputT = hInput.transpose(0, 1)  // (seq_len, batch, dState)
    var h = hInputT.select(0, 0)  // (batch, dState) — init with first position
    val hiddenSeq = mutable.ArrayBuffer[Tensor]()
    hiddenSeq += h

    for (pos <- 1 until seqLen) {
      val xPos = hInputT.select(0, pos)
      val aDecay = aRep.select(1, pos)  // (batch, dState)
      h = h.mul(aDecay).add(xPos)
      hiddenSeq += h
    }

    // Stack hidden states: (seq_len, batch, dState)
    val stackedTensors = mutable.ArrayBuffer[Tensor]()
    for (pos <- 0 until seqLen) {
      stackedTensors += hiddenSeq(pos).unsqueeze(0)
    }
    val hiddenStacked = torch.cat(new TensorVector(stackedTensors.toList*), 0)
    stackedTensors.foreach(_.close())

    val ssmOutPreProj = hiddenStacked.transpose(0, 1)  // (batch, seq_len, dState)
    val ssmOut = hProjOut.forward(ssmOutPreProj)  // (batch, seq_len, dInner)

    // Add skip connection (D parameter) - properly broadcast
    val dParamView = dParam.view(Array(1L, 1L, dInner.toLong)*)
    val skipOut = xHalf.mul(dParamView)  // [batch, seq, dInner] * [1, 1, dInner] -> [batch, seq, dInner]

    // Output projection with gate - shapes match now: [batch, seq, dInner]
    val ssmGated = ssmOut.add(skipOut)
    val gated = ssmGated.mul(gate)  // direct element-wise, no unsqueeze needed
    val out = outProj.forward(gated)

    // Residual
    val result = dropoutLayer.forward(out)
    hiddenStacked.close()
    ssmGated.close()
    result
  }
}
