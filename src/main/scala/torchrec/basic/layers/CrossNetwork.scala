package torchrec.basic.layers

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
import org.bytedeco.pytorch.global.torch.ScalarType

import torchrec.utils.DeviceSupport

/**
 * Cross Network from DCN.
 *
 * Mirrors the Python reference: a `ModuleList` of `Linear(input_dim, 1, bias=False)`
 * projections and a `ParameterList` of per-layer biases.
 *
 * Formula:
 *   x_{l+1} = x_0 * (W_l x_l) + b_l + x_l
 *
 * Shape
 * -----
 * Input:  (batch, inputDim)
 * Output: (batch, inputDim)
 */
class CrossNetwork(
  inputDim: Long,
  numLayers: Int = 3,
  device: String = DeviceSupport.backend
) extends Module {

  // ModuleList of Linear(inputDim, 1, bias=False), mirrors torch.nn.ModuleList.
  private val wLayers: ModuleListImpl = new ModuleListImpl()
  // ParameterList of per-layer bias vectors, mirrors torch.nn.ParameterList.
  private val biasList: ParameterListImpl = new ParameterListImpl()

  private val dev = new org.bytedeco.pytorch.Device(device)

  for (i <- 0 until numLayers) {
    val wOpt = new LinearOptions(inputDim, 1L).bias(false)
    val w = new LinearImpl(wOpt)
    wLayers.push_back(w)
    register_module(s"w_$i", w)

    val b = torch.zeros(Array(inputDim),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(dev, ScalarType.Float)
    biasList.append(b)
    register_parameter(s"b_$i", b)
  }

  override def forward(x0: Tensor): Tensor = {
    var xl = x0
    var i = 0
    while (i < numLayers) {
      val xw = wLayers.get(i).forward(xl)         // (batch, 1)
      val dot = x0.mul(xw)                         // (batch, inputDim)
      xl = dot.add(biasList.get(i)).add(xl)
      i += 1
    }
    xl
  }
}

/**
 * Cross Network V2 from DCN v2.
 *
 * Mirrors the Python reference: a `ModuleList` of `Linear(input_dim, input_dim, bias=False)`
 * plus per-layer biases.
 *
 * Formula:
 *   x_{l+1} = x_0 * (W_l x_l) + b_l + x_l
 *
 * Shape
 * -----
 * Input:  (batch, inputDim)
 * Output: (batch, inputDim)
 */
class CrossNetV2(
  inputDim: Long,
  numLayers: Int = 3,
  device: String = DeviceSupport.backend
) extends Module {

  private val wLayers: ModuleListImpl = new ModuleListImpl()
  private val biasList: ParameterListImpl = new ParameterListImpl()

  private val dev = new org.bytedeco.pytorch.Device(device)

  for (i <- 0 until numLayers) {
    val wOpt = new LinearOptions(inputDim, inputDim).bias(false)
    val w = new LinearImpl(wOpt)
    wLayers.push_back(w)
    register_module(s"w_$i", w)

    val b = torch.zeros(Array(inputDim),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(dev, ScalarType.Float)
    biasList.append(b)
    register_parameter(s"b_$i", b)
  }

  override def forward(x0: Tensor): Tensor = {
    var xl = x0
    var i = 0
    while (i < numLayers) {
      val wxl = wLayers.get(i).forward(xl)         // (batch, inputDim)
      xl = x0.mul(wxl).add(biasList.get(i)).add(xl)
      i += 1
    }
    xl
  }
}

/**
 * CrossNetMix from DCN v2 — Mixture-of-Experts cross network with low-rank
 * nonlinear projections and a softmax gating network per expert.
 *
 * Mirrors the Python reference exactly:
 *   - `U_list[i]`, `V_list[i]`, `C_list[i]` are `ParameterList` of shape
 *     `(num_experts, in_dim, low_rank)` (U/V) or `(num_experts, low_rank, low_rank)` (C)
 *     with Xavier-normal init.
 *   - `gating` is a `ModuleList` of `Linear(in_dim, 1, bias=False)` softmax gates.
 *   - `bias[i]` is `(in_dim, 1)`.
 *
 * Shape
 * -----
 * Input:  (batch, inDim)
 * Output: (batch, inDim)
 */
class CrossNetMix(
  inputDim: Long,
  numLayers: Int = 3,
  lowRank: Int = 32,
  numExperts: Int = 4,
  device: String = DeviceSupport.backend
) extends Module {

  private val uList: ParameterListImpl = new ParameterListImpl()
  private val vList: ParameterListImpl = new ParameterListImpl()
  private val cList: ParameterListImpl = new ParameterListImpl()
  private val biasList: ParameterListImpl = new ParameterListImpl()
  private val gating: ModuleListImpl = new ModuleListImpl()

  private val dev = new org.bytedeco.pytorch.Device(device)

  def makeParam(shape: Array[Long]): Tensor = {
    val t = torch.empty(shape: _*)
    t.to(dev, ScalarType.Float)
    torch.xavier_normal_(t)
    t
  }

  for (i <- 0 until numLayers) {
    // U[i]: (num_experts, inputDim, lowRank)
    val u = makeParam(Array(numExperts.toLong, inputDim, lowRank.toLong))
    uList.append(u)
    register_parameter(s"U_$i", u)

    // V[i]: (num_experts, inputDim, lowRank)
    val v = makeParam(Array(numExperts.toLong, inputDim, lowRank.toLong))
    vList.append(v)
    register_parameter(s"V_$i", v)

    // C[i]: (num_experts, lowRank, lowRank)
    val c = makeParam(Array(numExperts.toLong, lowRank.toLong, lowRank.toLong))
    cList.append(c)
    register_parameter(s"C_$i", c)

    // bias[i]: (inputDim, 1), zero-initialised like the Python `nn.init.zeros_`.
    val b = torch.zeros(Array(inputDim, 1L): _*)
    b.to(dev, ScalarType.Float)
    biasList.append(b)
    register_parameter(s"bias_$i", b)
  }

  for (e <- 0 until numExperts) {
    val gateOpt = new LinearOptions(inputDim, 1L).bias(false)
    val gate = new LinearImpl(gateOpt)
    gating.push_back(gate)
    register_module(s"gate_$e", gate)
  }

  override def forward(x: Tensor): Tensor = {
    // x: (batch, inputDim) -> (batch, inputDim, 1)
    val x0 = x.unsqueeze(2)
    var xl = x0

    var i = 0
    while (i < numLayers) {
      val uLayer = uList.get(i)
      val vLayer = vList.get(i)
      val cLayer = cList.get(i)
      val bLayer = biasList.get(i)

      // Collect per-expert outputs and gating scores.
      val expertOuts = scala.collection.mutable.ListBuffer[Tensor]()
      val expertGates = scala.collection.mutable.ListBuffer[Tensor]()

      var e = 0
      while (e < numExperts) {
        // Gating score: (batch, 1)
        val gScore = gating.get(e).forward(xl.squeeze(2))
        expertGates += gScore

        // Low-rank projection: v_x = V[e]^T · x_l   → (batch, lowRank, 1)
        var vx = torch.matmul(vLayer.select(0, e.toLong).t(), xl)
        vx = vx.tanh()
        vx = torch.matmul(cLayer.select(0, e.toLong), vx)
        vx = vx.tanh()
        // Project back: uv_x = U[e] · v_x            → (batch, inputDim, 1)
        val uvX = torch.matmul(uLayer.select(0, e.toLong), vx)
        // Add bias (inputDim, 1) and Hadamard with x0.
        val dot = x0.mul(uvX.add(bLayer))            // (batch, inputDim, 1)
        expertOuts += dot.squeeze(2)                 // (batch, inputDim)

        e += 1
      }

      // Stack experts: (batch, inputDim, numExperts)
      val outsStacked = torch.stack(new TensorVector(expertOuts.toSeq: _*), 2L)
      // Stack gating: (batch, numExperts, 1)
      val gatesStacked = torch.stack(new TensorVector(expertGates.toSeq: _*), 1L)
      // Mixture-of-experts weighted sum.
      val moeOut = torch.matmul(outsStacked, gatesStacked.softmax(1))
      xl = moeOut.add(xl)

      i += 1
    }

    xl.squeeze(2)
  }
}