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

import java.util
import java.util.concurrent.ConcurrentHashMap

class DistributedStore(
    private val options: DistributedStore.Options,
    private val rank: Int,
    private val worldSize: Int
) extends AutoCloseable {
  import DistributedStore.*

  private val prefix: String = s"_rank_${rank}_"
  private val selectedType: StoreType = resolveType(options)
  private val store: Store = selectedType match {
    case StoreType.TCP => createTCPStore(options, rank, worldSize)
    case _ => createFileStore(rank, worldSize)
  }

  private def resolveType(opts: Options): StoreType = {
    if (opts.getType == StoreType.AUTO) {
      val isMultiMachine = opts.getMasterAddr != "127.0.0.1" && opts.getMasterAddr != "localhost"
      if (isMultiMachine) StoreType.TCP else StoreType.FILE
    } else opts.getType
  }

  private def createFileStore(r: Int, ws: Int): Store = {
    val storePath = s"/tmp/pytorch_ddp_store_${System.getProperty("user.name")}"
    new FileStore(storePath, ws)
  }

  private def createTCPStore(opts: Options, r: Int, ws: Int): Store = {
    val tcpOpts = new TCPStoreOptions()
    tcpOpts.port(opts.getMasterPort.toShort)
    tcpOpts.isServer(false)
    tcpOpts.numWorkers().put(opts.getNumWorkers)
    tcpOpts.timeout(new Milliseconds(opts.getTimeoutMs))
    new TCPStore(opts.getMasterAddr, tcpOpts)
  }

  def set(key: String, value: String): Unit = store.set(prefix + key, value)

  def set(key: String, value: Array[Byte]): Unit = store.set(prefix + key, new ByteVector(value.toSeq: _*))

  def getString(key: String): String = {
    try {
      val bv = store.get(prefix + key)
      if (bv == null || bv.empty()) null.asInstanceOf[String]
      else new String(bv.get(), 0, bv.size().toInt)
    } catch { case _: RuntimeException => null.asInstanceOf[String] }
  }

  def getBytes(key: String): Array[Byte] = {
    try {
      val bv = store.get(prefix + key)
      if (bv == null || bv.empty()) null.asInstanceOf[Array[Byte]]
      else bv.get()
    } catch { case _: RuntimeException => null.asInstanceOf[Array[Byte]] }
  }

  def add(key: String, value: Long): Long = store.add(prefix + key, value)

  def getInteger(key: String): Int = store.add(prefix + key, 0).toInt

  def delete(key: String): Unit = store.deleteKey(prefix + key)

  def exists(key: String): Boolean = store.check(new StringVector(prefix + key))

  def waitFor(key: String): Unit = store.`_wait`(new StringVector(prefix + key))

  def waitFor(key: String, timeoutMs: Int): Unit =
    store.`_wait`(new StringVector(prefix + key), new Milliseconds(timeoutMs))

  def getNativeStore: Store = store
  def getRank: Int = rank
  def getWorldSize: Int = worldSize
  def getType: StoreType = selectedType

  override def close(): Unit = DistributedStore.instances.remove(this)

  override def toString: String = s"DistributedStore{type=$selectedType, rank=$rank, worldSize=$worldSize}"
}

object DistributedStore {

  class Options(
    var storeType: StoreType = StoreType.AUTO,
    var timeoutMs: Int = 300000,
    var masterAddr: String = "127.0.0.1",
    var masterPort: Int = 29500,
    var numWorkers: Int = 1
  ) {
    def this() = this(StoreType.AUTO, 300000, "127.0.0.1", 29500, 1)

    def `type`(t: StoreType): Options = { storeType = t; this }
    def timeout(ms: Int): Options = { timeoutMs = ms; this }
    def masterAddr(addr: String): Options = { masterAddr = addr; this }
    def masterPort(port: Int): Options = { masterPort = port; this }
    def numWorkers(n: Int): Options = { numWorkers = n; this }

    def getType: StoreType = storeType
    def getTimeoutMs: Int = timeoutMs
    def getMasterAddr: String = masterAddr
    def getMasterPort: Int = masterPort
    def getNumWorkers: Int = numWorkers
  }

  private val instances = ConcurrentHashMap.newKeySet[DistributedStore]()

  def getOrCreate(name: String, opts: Options, r: Int, ws: Int): DistributedStore = {
    val key = s"$name:$r:$ws"
    instances.synchronized {
      val found = instances.iterator()
      while (found.hasNext) {
        val s = found.next()
        if (s != null && s.getRank == r && s.getWorldSize == ws) return s
      }
      val newStore = new DistributedStore(opts, r, ws)
      instances.add(newStore)
      newStore
    }
  }

  def create(opts: Options, r: Int, ws: Int): DistributedStore = {
    val store = new DistributedStore(opts, r, ws)
    instances.add(store)
    store
  }

  def create(r: Int, ws: Int): DistributedStore = create(new Options(), r, ws)
}
