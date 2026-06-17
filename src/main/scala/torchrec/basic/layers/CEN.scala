package torchrec.basic.layers

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.utils.DeviceSupport
import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.utils.DeviceSupport.deviceOf


class CEN(
           embedDim: Int,
           numFieldCrosses: Int, // C = 交叉项总数
           reductionRatio: Int,
           device: String = DeviceSupport.backend
         ) extends Module {
  private val dev = deviceOf(device)
  private val tensorOpts = new TensorOptions()
    .dtype(new ScalarTypeOptional(ScalarType.Float))
    .device(new DeviceOptional(dev))

  // 完全对齐Python self.u: shape [C, D]，不再带多余1维
  private val u = torch.rand(
    Array(numFieldCrosses.toLong, embedDim.toLong),
    tensorOpts
  )
  register_parameter("u", u)

  private val mlpAttention = new MLP(
    inputDim = numFieldCrosses,
    hiddenDims = List((numFieldCrosses / reductionRatio).toLong),
    outputDim = numFieldCrosses,
    activation = "relu",
    useBatchNorm = false,
    useLayerNorm = false,
    outputLayer = true,
    device = device
  )
  register_module("mlp_att", mlpAttention)

  def forward(em: Tensor): Tensor = {
    println("===== CEN forward START =====")
    val b = em.size(0)
    val c = em.size(1)
    val d = em.size(2)
    println(s"[DEBUG] 输入em shape = [$b, $c, $d], device=${em.device().str()}")
    println(s"[DEBUG] 层配置 numFieldCrosses=$numFieldCrosses, embedDim=$embedDim, layer dev=${dev.str()}")

    // 前置校验
    require(c == numFieldCrosses,
      s"CEN cross count mismatch: expect $numFieldCrosses, input dim1=$c, em shape [$b,$c,$d]")
    require(d == embedDim,
      s"CEN embed dim mismatch: expect $embedDim, input dim2=$d")
    require(em.device().equals(dev),
      s"CEN device mismatch: layer=$dev, input=${em.device()}")
    println("[DEBUG] 前置维度&设备校验通过")

    // 打印u原始维度
    val u0 = u
    println(s"[DEBUG] u raw shape = [${u0.size(0)}, ${u0.size(1)}], device=${u0.device().str()}")

    // 升维 [C,D] -> [1,C,D]
    val u3d = u.unsqueeze(0)
    println(s"[DEBUG] u3d unsqueeze(0) shape = [${u3d.size(0)}, ${u3d.size(1)}, ${u3d.size(2)}]")

    // 第一处乘法：torch.mul(u3d, em) 极易报错，打印前后
    println("[DEBUG] 准备执行 torch.mul(u3d, em)")
    val mul = torch.mul(u3d, em)
    println(s"[DEBUG] mul 结果 shape = [${mul.size(0)}, ${mul.size(1)}, ${mul.size(2)}]")

    val dVec = torch.relu(mul.sum(-1))
    println(s"[DEBUG] dVec(sum(-1)) shape = [${dVec.size(0)}, ${dVec.size(1)}]")

    val s = mlpAttention.forward(dVec)
    println(s"[DEBUG] mlp_att输出 s shape = [${s.size(0)}, ${s.size(1)}]")

    val sExpand = s.unsqueeze(-1)
    println(s"[DEBUG] sExpand unsqueeze(-1) shape = [${sExpand.size(0)}, ${sExpand.size(1)}, ${sExpand.size(2)}]")

    // 第二处乘法
    println("[DEBUG] 准备执行 torch.mul(sExpand, em)")
    val aem = torch.mul(sExpand, em)
    println(s"[DEBUG] aem shape = [${aem.size(0)}, ${aem.size(1)}, ${aem.size(2)}]")
    val out = aem.reshape(Array(aem.size(0), -1) *)
//    val out = aem.flatten(1l,1l) //big bug 
    println(s"[DEBUG] 最终输出 flatten shape = [${out.size(0)}, ${out.size(1)}]")
    println("===== CEN forward END =====")
    out
  }

}

object CEN {
  def apply(embedDim: Int, numFieldCrosses: Int, reductionRatio: Int, device: String = DeviceSupport.backend): CEN = {
    new CEN(embedDim, numFieldCrosses, reductionRatio, device)
  }
}
