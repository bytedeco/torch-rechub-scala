package torchrec.distributed

import org.bytedeco.pytorch.{*}
import org.bytedeco.pytorch.optim._
import org.bytedeco.pytorch.distributed._
import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.global.torch as pt
import org.bytedeco.pytorch.global.torch.DeviceType

import java.util
import java.util.concurrent.ConcurrentHashMap

/** FSDPTrainer - Quanmian Fenpi Data Bingxing (FSDP) */
class FSDPTrainer(
    val module: Module,
    val processGroup: ProcessGroupWrapper,
    shardingStrategy: FSDPTrainer.ShardingStrategy = FSDPTrainer.ShardingStrategy.FULL_SHARD,
    reshardAfterForward: Boolean = true,
    useFullPrecision: Boolean = true
) extends AutoCloseable {
  import FSDPTrainer.*

  private val shardedParams = new util.ArrayList[Tensor]()
  private val shardedGrads = new util.ArrayList[Tensor]()
  private val paramShapes = new util.ArrayList[Long]()
  private val paramNumels = new util.ArrayList[Long]()
  private var totalParamNumel: Long = 0
  private var shardSize: Long = 0
  private var isFirstForward = true
  private var useFullPrec = useFullPrecision

  private val forward: ModuleForward = ModuleForward.of(module)
  private val isDevicecuda: Boolean = processGroup.getDevice.`type`() == pt.DeviceType.CUDA
  private val extraState: util.Map[String, Object] = new util.HashMap()
  private var numForwardCalls: Long = 0
  private var numBackwardCalls: Long = 0
  
  val device = processGroup.getDevice
  if (device.`type`() == DeviceType.CUDA) {
    module.to(device, true)
  }

  collectParamMetadata()
  shardParameters()
  broadcastFullParameters()

  println(f"[FSDPTrainer] Initialized strategy=$shardingStrategy, shardSize=$shardSize, rank=${processGroup.getRank}%d")

  /** Ji lu Can Shu yuan shuju */
  private def collectParamMetadata(): Unit = {
    paramShapes.clear()
    paramNumels.clear()
    totalParamNumel = 0

    try {
      val params = module.parameters()
      val p = params.begin()
      val end = params.end()

      while (!p.equals(end)) {
        val tensor = p.get()
        if (tensor != null && !tensor.isNull) {
          try {
            val numel = tensor.numel()
            paramNumels.add(numel)
            totalParamNumel += numel

            var dimIdx = 0
            while (dimIdx < tensor.dim) {
              paramShapes.add(tensor.size(dimIdx))
              dimIdx += 1
            }
          } catch {
            case _: Exception =>
          }
        }
        p.increment()
      }
    } catch {
      case _: Exception => totalParamNumel = 0
    }

    shardSize = (totalParamNumel + processGroup.getWorldSize - 1) / processGroup.getWorldSize
  }

  /** Fen Pi Can Shu */
  private def shardParameters(): Unit = {
    val rankVal = processGroup.getRank
    val worldSizeVal = processGroup.getWorldSize

    val flatParams = flattenParameters()

    val start = rankVal * shardSize
    val endVal = Math.min(start + shardSize, totalParamNumel)

    val shard = flatParams.slice(0, new LongOptional(start), new LongOptional(endVal), 1)
    val sharded = shard.clone.detach()
    sharded.requires_grad_(true)

    shardedParams.clear()
    shardedParams.add(sharded)

    shardedGrads.clear()
    shardedGrads.add(pt.zeros_like(sharded))

    flatParams.close()
    shard.close()
  }

  /** Ping Can Shu Dao Dan Zhang Liang */
  private def flattenParameters(): Tensor = {
    val flatList = new util.ArrayList[Tensor]()
    try {
      val params = module.parameters()
      val it = params.begin()
      val end = params.end()

      while (!it.equals(end)) {
        val t = it.get()
        if (t != null && !t.isNull) {
          try {
            flatList.add(t.flatten())
          } catch { case _: Exception => }
        }
        it.increment()
      }
    } catch { case _: Exception => return pt.zeros(1) }

    if (flatList.isEmpty) pt.zeros(1)
    else pt.cat(new TensorVector(flatList.toArray(new Array[Tensor](0)): _*))
  }

  /** Cong rank 0 Guang bo Can Shu */
  private def broadcastFullParameters(): Unit = {
    try {
      val params = module.parameters()
      val it = params.begin()
      val end = params.end()

      while (!it.equals(end)) {
        val t = it.get()
        if (t != null && !t.isNull) {
          processGroup.broadcast(t, 0)
        }
        it.increment()
      }
    } catch { case _: Exception => }
  }

  /** Ping Tidu */
  private def flattenGradients(): Tensor = {
    val gradList = new util.ArrayList[Tensor]()
    try {
      val params = module.parameters()
      val it = params.begin()
      val end = params.end()

      while (!it.equals(end)) {
        val t = it.get()
        if (t != null && !t.isNull) {
          try {
            val g = t.grad()
            if (g != null && !g.isNull && g.defined()) {
              gradList.add(g.flatten())
            }
          } catch { case _: Exception => }
        }
        it.increment()
      }
    } catch { case _: Exception => return pt.zeros(1) }

    if (gradList.isEmpty) pt.zeros(1)
    else pt.cat(new TensorVector(gradList.toArray(new Array[Tensor](0)): _*))
  }

  /** Quan Ju Ji He Fen Pi Can Shu Dao Wan Zheng Can Shu */
  private def allGatherParameters(): Tensor = {
    val worldSizeVal = processGroup.getWorldSize
    val deviceVal = processGroup.getDevice

    val gathered = new util.ArrayList[Tensor]()
    var gIdx = 0
    while (gIdx < worldSizeVal) {
      gathered.add(pt.empty(Array(shardSize)*).to(deviceVal, pt.ScalarType.Float))
      gIdx += 1
    }

    val padded = shardedParams.get(0)
    val paddedInput = if (padded.numel < shardSize) {
      val paddedVec = new TensorVector(padded, pt.zeros(Array(shardSize - padded.numel)*).to(deviceVal, pt.ScalarType.Float))
      pt.cat(paddedVec)
    } else {
      if (!padded.device.equals(deviceVal)) padded.to(deviceVal, pt.ScalarType.Float)
      else padded
    }

    val input = new util.ArrayList[Tensor]()
    input.add(paddedInput)

    processGroup.allgather(gathered, input)

    val fullVector = new TensorVector(gathered.toArray(new Array[Tensor](0)): _*)
    val full = pt.cat(fullVector)
    if (full.numel > totalParamNumel) {
      full.slice(0, new LongOptional(0), new LongOptional(totalParamNumel), 1l)
    } else full
  }

  /** Qian Xiang Chuan Di */
  def forward(input: Tensor): Tensor = {
    val deviceVal = processGroup.getDevice

    var inputAdj = input
    if (input.device.`type`() != deviceVal.`type`) {
      inputAdj = input.to(deviceVal, pt.ScalarType.Float)
    }

    val fullParams = allGatherParameters()
    writeToModule(fullParams)

    numForwardCalls += 1
    val output =  forward.apply(module, inputAdj)
//    val output = module match {
//      case net: BenchmarkNet => net.forward(inputAdj)
//      case _ => nativeForward(inputAdj)
//    }

    if (reshardAfterForward) fullParams.close()

    output
  }

  def nativeForward(input: Tensor): Tensor = {
    numForwardCalls += 1
    forward.apply(module, input)
  }
  
  def forward(input: Tensor, target: Tensor): Tensor = {
    val output = forward(input)
    pt.cross_entropy(output, target)
  }

  /** Wan Zheng Xunlian Bu */
  def step(input: Tensor, target: Tensor, optimizer: Optimizer): Tensor = {
    zeroGrad()

    val output = forward(input)
    val loss = pt.cross_entropy(output, target)
    loss.backward()
    numBackwardCalls += 1
    reduceScatterGradients()

    optimizer.step()

    loss
  }

  def trainingStep(loss: Tensor, optimizer: Optimizer): Tensor = {
    optimizer.zero_grad()
    loss.backward()
    reduceScatterGradients()
    optimizer.step()
    loss
  }

  /** Reduce-Scatter Tidu */
  private def reduceScatterGradients(): Unit = {
    val gradFlat = flattenGradients()
    if (gradFlat == null) return

    val worldSizeVal = processGroup.getWorldSize
    val rankVal = processGroup.getRank

    val splits = new util.ArrayList[Tensor]()
    var i = 0
    while (i < worldSizeVal) {
      val s = i * shardSize
      val e = Math.min(s + shardSize, totalParamNumel)
      splits.add(gradFlat.slice(0, new LongOptional(s), new LongOptional(e), 1))
      i += 1
    }

    val output = new util.ArrayList[Tensor]()
    output.add(pt.empty_like(shardedParams.get(0)))

    processGroup.reduceScatter(output, splits)

    if (shardedGrads.isEmpty) {
      shardedGrads.add(pt.zeros_like(shardedParams.get(0)))
    }
    shardedGrads.get(0).data.copy_(output.get(0).div(new Scalar(worldSizeVal.toFloat)))

    gradFlat.close()
    val outItr = output.iterator()
    while (outItr.hasNext) outItr.next().close()
  }

  /** Xie Ru Ping Can Shu Hui Mo xing */
  private def writeToModule(flatParams: Tensor): Unit = {
    var offset = 0L
    try {
      val params = module.parameters()
      val it = params.begin()
      val end = params.end()

      while (!it.equals(end)) {
        val t = it.get()
        if (t != null && !t.isNull) {
          try {
            val n = t.numel()
            if (offset + n <= flatParams.numel) {
              val src = flatParams.narrow(0, offset, n)
              t.copy_(src)
              src.close()
            }
            offset += n
          } catch { case _: Exception => }
        }
        it.increment()
      }
    } catch { case _: Exception => }
  }

  /** Ling Tidu */
  private def zeroGrad(): Unit = {
    try {
      val params = module.parameters()
      val it = params.begin()
      val end = params.end()

      while (!it.equals(end)) {
        val t = it.get()
        if (t != null && !t.isNull) {
          try {
            val g = t.grad()
            if (g != null && !g.isNull && g.defined()) g.zero_()
          } catch { case _: Exception => }
        }
        it.increment()
      }
    } catch { case _: Exception => }
  }

  def getModule: Module = module

  def getShardedParameters: util.List[Tensor] = shardedParams

  def getShardedGradients: util.List[Tensor] = shardedGrads

  def parameters: Iterable[Tensor] = {
    val list = new util.ArrayList[Tensor]()
    try {
      val params = module.parameters()
      val it = params.begin()
      val end = params.end()

      while (!it.equals(end)) {
        val t = it.get()
        if (t != null && !t.isNull) list.add(t)
        it.increment()
      }
    } catch { case _: Exception => }
    list.toArray(new Array[Tensor](0)).toIterable.asInstanceOf[Iterable[Tensor]]
  }

  def train(): Unit = module.train(true)

  def eval(): Unit = module.eval()

  def isTraining: Boolean = module.is_training

  def setFullPrecision(useFullPrecParam: Boolean): Unit = { useFullPrec = useFullPrecParam }

  def getProcessGroup: ProcessGroupWrapper = processGroup

  def getShardingStrategy: FSDPTrainer.ShardingStrategy = shardingStrategy

  def getRank: Int = processGroup.getRank

  def getWorldSize: Int = processGroup.getWorldSize

  def isMainProcess: Boolean = processGroup.isMainProcess

  def getDevice: Device = device

  def getShardSize: Long = shardSize

  def getTotalParamSize: Long = totalParamNumel

  override def close(): Unit = {
    val itr = shardedParams.iterator()
    while (itr.hasNext) itr.next().close()
    val gItr = shardedGrads.iterator()
    while (gItr.hasNext) gItr.next().close()
    FSDPTrainer.instances.remove(this)
    module.close()
  }

  override def toString: String = s"FSDPTrainer{rank=${processGroup.getRank}, worldSize=${processGroup.getWorldSize}, strategy=$shardingStrategy, shardSize=$shardSize}"
}

object FSDPTrainer {
  enum ShardingStrategy {
    case FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD
  }

  val VERSION = "1.0"

  private val instances = ConcurrentHashMap.newKeySet[FSDPTrainer]()

  def apply(
    module: Module,
    processGroup: ProcessGroupWrapper
  ): FSDPTrainer = new FSDPTrainer(module, processGroup)

  def builder(): Builder = new Builder

  class Builder {
    private var module: Module = null.asInstanceOf[Module]
    private var processGroup: ProcessGroupWrapper = null.asInstanceOf[ProcessGroupWrapper]
    private var shardingStrategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    private var reshardAfterForward = true
    private var useFullPrecision = true

    def module(m: Module): Builder = { module = m; this }
    def processGroup(pg: ProcessGroupWrapper): Builder = { processGroup = pg; this }
    def shardingStrategy(s: ShardingStrategy): Builder = { shardingStrategy = s; this }
    def reshardAfterForward(b: Boolean): Builder = { reshardAfterForward = b; this }
    def useFullPrecision(b: Boolean): Builder = { useFullPrecision = b; this }

    def build(): FSDPTrainer = {
      require(module != null, "module is required")
      require(processGroup != null, "processGroup is required")
      new FSDPTrainer(module, processGroup, shardingStrategy, reshardAfterForward, useFullPrecision)
    }
  }
}