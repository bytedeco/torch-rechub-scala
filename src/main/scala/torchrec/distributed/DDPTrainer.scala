package torchrec.distributed

import org.bytedeco.pytorch.{*}
import org.bytedeco.pytorch.optim._
import org.bytedeco.pytorch.distributed._
import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.global.torch as pt

import java.util
import java.util.concurrent.ConcurrentHashMap

class DDPTrainer(
    val model: Module,
    val processGroup: ProcessGroupWrapper
) extends AutoCloseable {
  import DDPTrainer.*

  val VERSION = "1.0"

  private val forward: ModuleForward = ModuleForward.of(model)
  private val isDevicecuda: Boolean = processGroup.getDevice.`type`() == pt.DeviceType.CUDA
  private val extraState: util.Map[String, Object] = new util.HashMap()
  private var numForwardCalls: Long = 0
  private var numBackwardCalls: Long = 0

  initialize()

  private def initialize(): Unit = {
    val device = processGroup.getDevice
    model.to(device, true)
    if (processGroup.getWorldSize > 1) {
      broadcastInitialParameters()
    }
    println(f"[DDPTrainer] Initialized on rank ${processGroup.getRank}%d with device=$device, worldSize=${processGroup.getWorldSize}%d")
  }

  private def broadcastInitialParameters(): Unit = {
    if (processGroup.getWorldSize <= 1) return
    val params = collectParameters()
    if (!params.isEmpty) {
      val itr = params.iterator()
      while (itr.hasNext) {
        processGroup.broadcast(itr.next(), 0)
      }
    }
  }

  private def collectParameters(): util.List[Tensor] = {
    val params = new util.ArrayList[Tensor]()
    val paramVec = model.parameters()
    val begin = paramVec.begin()
    val end = paramVec.end()
    while (!begin.equals(end)) {
      val p = begin.get()
      if (p != null && !p.isNull) {
        params.add(p.clone())
      }
      begin.increment()
    }
    params
  }

  def forward(input: Tensor): Tensor = {
    numForwardCalls += 1
    forward.apply(model, input)
  }

  def step(input: Tensor, target: Tensor, optimizer: Optimizer): Tensor = {
    val output = forward(input)
    val loss = pt.cross_entropy(output, target)
    optimizer.zero_grad()
    loss.backward()
    numBackwardCalls += 1
    reduceGradients()
    optimizer.step()
    loss
  }

  def training_step(input: Tensor, target: Tensor, optimizer: Optimizer): Tensor =
    step(input, target, optimizer)

  def synchronize(): Unit = reduceGradients()

  private def reduceGradients(): Unit = {
    if (processGroup.getWorldSize <= 1) return
    val gradients = new util.ArrayList[Tensor]()
    try {
      val paramVec = model.parameters()
      val begin = paramVec.begin()
      val end = paramVec.end()
      while (!begin.equals(end)) {
        val p = begin.get()
        if (p != null && !p.isNull) {
          try {
            val grad = p.grad()
            if (grad != null && !grad.isNull && grad.defined()) {
              gradients.add(grad)
            }
          } catch { case _: Exception => }
        }
        begin.increment()
      }
    } catch { case _: Exception => return }

    if (!gradients.isEmpty) {
      processGroup.allreduce(gradients, ReduceOp.RedOpType.SUM)
      val worldSize = processGroup.getWorldSize
      val itr = gradients.iterator()
      while (itr.hasNext) {
        itr.next().div_(new Scalar(worldSize))
      }
    }
  }

  def getModule: Module = model
  def getLocalModule: Module = model
  def getModuleForTraining: Module = model

  def parameters: Iterable[Tensor] = {
    val list = new util.ArrayList[Tensor]()
    val paramVec = model.parameters()
    val begin = paramVec.begin()
    val end = paramVec.end()
    while (!begin.equals(end)) {
      list.add(begin.get())
      begin.increment()
    }
    list.asInstanceOf[Iterable[Tensor]]
  }

  def setParameters(params: util.List[Tensor]): Unit = {
    val paramVec = model.parameters()
    var i = 0
    val begin = paramVec.begin()
    val end = paramVec.end()
    while (!begin.equals(end)) {
      val p = begin.get()
      if (p != null && !p.isNull && i < params.size) {
        p.set_(params.get(i))
        i += 1
      }
      begin.increment()
    }
  }

  def namedBuffers(): util.Map[String, Tensor] = {
    val buffers = new util.HashMap[String, Tensor]()
    val bufVec = model.buffers()
    val begin = bufVec.begin()
    val end = bufVec.end()
    var idx = 0
    while (!begin.equals(end)) {
      buffers.put(s"buffer_$idx", begin.get())
      idx += 1
      begin.increment()
    }
    buffers
  }

  def getTempStateDict: util.Map[String, Object] = extraState
  def loadTempstate_dict(state: util.Map[String, Object]): Unit = extraState.putAll(state)

  def train(): Unit = model.train(true)
  def eval(): Unit = model.eval()
  def isTraining: Boolean = model.is_training

  def getNumForwardCalls: Long = numForwardCalls
  def getNumBackwardCalls: Long = numBackwardCalls
  def resetStats(): Unit = { numForwardCalls = 0; numBackwardCalls = 0 }

  def getProcessGroup: ProcessGroupWrapper = processGroup
  def getRank: Int = processGroup.getRank
  def getWorldSize: Int = processGroup.getWorldSize
  def isMainProcess: Boolean = processGroup.isMainProcess
  def getDevice: Device = processGroup.getDevice

  override def close(): Unit = {
    model.close()
    DDPTrainer.instances.remove(this)
  }

  override def toString: String = s"DDPTrainer{rank=${processGroup.getRank}, worldSize=${processGroup.getWorldSize}, device=${processGroup.getDevice}, forwardCalls=$numForwardCalls}"
}

object DDPTrainer {
  val VERSION = "1.0"

  private val instances = ConcurrentHashMap.newKeySet[DDPTrainer]()

  def create(model: Module, pg: ProcessGroupWrapper): DDPTrainer = builder().module(model).processGroup(pg).build()

  def builder(): Builder = new Builder

  class Builder {
    private var module: Module = null.asInstanceOf[Module]
    private var processGroup: ProcessGroupWrapper = null.asInstanceOf[ProcessGroupWrapper]
    private var broadcastBuffers: Boolean = true
    private var gradientAsBucketstore: Boolean = true
    private var bucketCapKb: Int = 25 * 1024

    def module(m: Module): Builder = { module = m; this }
    def processGroup(pg: ProcessGroupWrapper): Builder = { processGroup = pg; this }
    def broadcastBuffers(b: Boolean): Builder = { broadcastBuffers = b; this }
    def gradientAsBucketstore(b: Boolean): Builder = { gradientAsBucketstore = b; this }
    def bucketCapKb(kb: Int): Builder = { bucketCapKb = kb; this }

    def build(): DDPTrainer = {
      require(module != null, "module is required")
      require(processGroup != null, "processGroup is required")
      val trainer = new DDPTrainer(module, processGroup)
      instances.add(trainer)
      trainer
    }
  }
}
