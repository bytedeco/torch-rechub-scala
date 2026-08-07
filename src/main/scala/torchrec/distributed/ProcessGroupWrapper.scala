package torchrec.distributed

import org.bytedeco.javacpp.chrono.Milliseconds
import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.nn.modules._
import org.bytedeco.pytorch.nn.modules.container._
import org.bytedeco.pytorch.nn.options._
import org.bytedeco.pytorch.optim._
import org.bytedeco.pytorch.data.datasets._
import org.bytedeco.pytorch.data.options._
import org.bytedeco.pytorch.data.sampler._
import org.bytedeco.pytorch.distributed._
import org.bytedeco.pytorch.global.torch as pt
import org.bytedeco.pytorch.distributed.ProcessGroupNCCL

import java.util
import java.util.concurrent.ConcurrentHashMap

class ProcessGroupWrapper(
    options: ProcessGroupWrapper.Options,
    val rank: Int,
    val worldSize: Int,
    store: DistributedStore
) extends AutoCloseable {
  import ProcessGroupWrapper.*

  private val backendType: BackendType = resolveBackend(options.backendType)
  private val timeoutVal = new Milliseconds(options.timeoutMs)

  val device: Device = if (backendType == BackendType.NCCL && pt.cuda_is_available()) {
    new Device(pt.DeviceType.CUDA, rank.toByte)
  } else {
    new Device(pt.DeviceType.CPU, 0.toByte)
  }

  private val (processGroupVal, backendName) = initializePG(backendType, options, rank, worldSize, store, timeoutVal, device)
  private val processGroup = processGroupVal

  println(f"[Rank $rank%d] ProcessGroup initialized with backend=$backendName, device=$device")

  private def initializePG(
    bType: BackendType,
    opts: Options,
    r: Int,
    ws: Int,
    st: DistributedStore,
    to: Milliseconds,
    dev: Device
  ): (org.bytedeco.pytorch.distributed.Backend, String) = bType match {
    case BackendType.NCCL if pt.cuda_is_available() =>
      val pgOpts = ProcessGroupNCCL.Options.create(true)
      pgOpts.timeout(to)
      (new ProcessGroupNCCL(st.getNativeStore, r, ws, pgOpts), "nccl")
    case _ =>
      val pgOpts = ProcessGroupGloo.Options.create()
      pgOpts.timeout(to)
      val devices = new GlooDeviceVector()
      devices.push_back(ProcessGroupGloo.createDeviceForHostname(opts.masterAddr))
      pgOpts.devices(devices)
      (new ProcessGroupGloo(st.getNativeStore, r, ws, pgOpts), "gloo")
  }

  def allreduce(tensors: util.List[Tensor]): Work = allreduce(tensors, ReduceOp.RedOpType.SUM)

  def allreduce(tensors: util.List[Tensor], op: ReduceOp.RedOpType): Work = {
    val opts = new AllreduceOptions()
    opts.reduceOp(new ReduceOp(op))
    processGroup.allreduce(toTensorVector(tensors), opts)
  }

  def allreduce(tensor: Tensor): Work = allreduce(util.Collections.singletonList(tensor))

  def broadcast(tensor: Tensor, rootRank: Int): Work = {
    val opts = new BroadcastOptions()
    opts.rootRank(rootRank)
    processGroup.broadcast(toTensorVector(tensor), opts)
  }

  def broadcast(tensors: util.List[Tensor], rootRank: Int): Work = {
    val opts = new BroadcastOptions()
    opts.rootRank(rootRank)
    processGroup.broadcast(toTensorVector(tensors), opts)
  }

  def allgather(outputTensors: util.List[Tensor], inputTensors: util.List[Tensor]): Work = {
    val opts = new AllgatherOptions()
    processGroup.allgather(toTensorVector(outputTensors), toTensorVector(inputTensors), opts)
  }

  def allgather(outputTensors: util.List[Tensor], inputTensor: Tensor): Work =
    allgather(outputTensors, util.Collections.singletonList(inputTensor))

  def reduceScatter(outputTensors: util.List[Tensor], inputTensors: util.List[Tensor]): Work = {
    val opts = new ReduceScatterOptions()
    opts.reduceOp(new ReduceOp(ReduceOp.RedOpType.SUM))
    processGroup.reduce_scatter(toTensorVector(outputTensors), toTensorVector(inputTensors), opts)
  }

  def send(tensor: Tensor, dstRank: Int, tag: Int = 0): Unit = {
    val work = processGroup.send(toTensorVector(tensor), dstRank, tag)
    work._wait()
  }

  def recv(tensor: Tensor, srcRank: Int, tag: Int = 0): Tensor = {
    val work = processGroup.recv(toTensorVector(tensor), srcRank, tag)
    work._wait()
    tensor
  }

  def recvAnysource(tensor: Tensor, tag: Int = 0): Tensor = {
    val work = processGroup.recvAnysource(toTensorVector(tensor), tag)
    work._wait()
    tensor
  }

  def barrier(): Work = processGroup.barrier(new BarrierOptions())

  def averageGradients(gradients: util.List[Tensor]): Unit = {
    if (gradients.isEmpty) return
    allreduce(gradients, ReduceOp.RedOpType.SUM)
    val itr = gradients.iterator()
    while (itr.hasNext) {
      itr.next().div_(new Scalar(worldSize))
    }
  }

  def syncParameters(parameters: util.List[Tensor], rootRank: Int): Unit = {
    broadcast(parameters, rootRank)
  }

  def getNativeGroup: org.bytedeco.pytorch.distributed.Backend = processGroup
  def getRank: Int = rank
  def getWorldSize: Int = worldSize
  def getBackend: BackendType = backendType
  def getBackendName: String = backendName
  def getDevice: Device = device
  def isMainProcess: Boolean = rank == 0

  override def close(): Unit = {
    if (backendName.equalsIgnoreCase("nccl")) {
      processGroup.waitForPendingWorks()
    }
    processGroup.shutdown()
    ProcessGroupWrapper.instances.remove(this)
  }

  override def toString: String = s"ProcessGroupWrapper{backend=$backendName, rank=$rank, worldSize=$worldSize, device=$device}"

  private def resolveBackend(requested: BackendType): BackendType = {
    if (requested == BackendType.AUTO) {
      if (pt.cuda_is_available()) BackendType.NCCL else BackendType.GLOO
    } else if (requested == BackendType.NCCL && !pt.cuda_is_available()) {
      System.err.println("WARNING: NCCL requested but CUDA not available, falling back to GLOO")
      BackendType.GLOO
    } else requested
  }

  private def toTensorVector(tensors: util.List[Tensor]): TensorVector =
    new TensorVector(tensors.toArray(new Array[Tensor](0)): _*)

  private def toTensorVector(tensor: Tensor): TensorVector = new TensorVector(tensor)
}

object ProcessGroupWrapper {

  class Options(
    var backendType: BackendType = BackendType.AUTO,
    var timeoutMs: Int = 300000,
    var masterAddr: String = "127.0.0.1",
    var masterPort: Int = 29500
  ) {
    def this() = this(BackendType.AUTO, 300000, "127.0.0.1", 29500)

    def backend(b: BackendType): Options = { backendType = b; this }
    def timeout(ms: Int): Options = { timeoutMs = ms; this }
    def masterAddr(addr: String): Options = { masterAddr = addr; this }
    def masterPort(port: Int): Options = { masterPort = port; this }

    def getBackend: BackendType = backendType
    def getTimeoutMs: Int = timeoutMs
    def getMasterAddr: String = masterAddr
    def getMasterPort: Int = masterPort
  }

  private val instances = ConcurrentHashMap.newKeySet[ProcessGroupWrapper]()

  def getOrCreate(
    groupName: String,
    opts: Options,
    r: Int,
    ws: Int,
    store: DistributedStore
  ): ProcessGroupWrapper = {
    instances.synchronized {
      val itr = instances.iterator()
      while (itr.hasNext) {
        val w = itr.next()
        if (w != null && w.getRank == r && w.getWorldSize == ws) return w
      }
      val w = new ProcessGroupWrapper(opts, r, ws, store)
      instances.add(w)
      w
    }
  }

  def create(opts: Options, r: Int, ws: Int, store: DistributedStore): ProcessGroupWrapper = {
    val w = new ProcessGroupWrapper(opts, r, ws, store)
    instances.add(w)
    w
  }

  def create(r: Int, ws: Int, store: DistributedStore): ProcessGroupWrapper =
    create(new Options(), r, ws, store)
}
