package torchrec

import org.bytedeco.pytorch.global.torch
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
import org.bytedeco.pytorch.global.torch.ScalarType
import org.bytedeco.javacpp.{FloatPointer, DoublePointer, LongPointer}
import org.bytedeco.pytorch.optim.options.AdamOptions
package object torchrec {
  type Tensor = org.bytedeco.pytorch.Tensor
  type Module = org.bytedeco.pytorch.nn.Module
  type ScalarType = org.bytedeco.pytorch.global.torch.ScalarType
  type TensorVector = org.bytedeco.pytorch.TensorVector
  type TensorOptions = org.bytedeco.pytorch.TensorOptions
  type ScalarTypeOptional = org.bytedeco.pytorch.ScalarTypeOptional
  type Device = org.bytedeco.pytorch.Device
  type Scalar = org.bytedeco.pytorch.Scalar
  type LongPointer = org.bytedeco.javacpp.LongPointer
  type FloatPointer = org.bytedeco.javacpp.FloatPointer
  type DoublePointer = org.bytedeco.javacpp.DoublePointer
  type IntPointer = org.bytedeco.javacpp.IntPointer
  type BytePointer = org.bytedeco.javacpp.BytePointer
  type Pointer = org.bytedeco.javacpp.Pointer
  type Loader = org.bytedeco.javacpp.Loader
  type PointerScope = org.bytedeco.javacpp.PointerScope
  // Layers / nn modules
  type LinearImpl = org.bytedeco.pytorch.nn.modules.LinearImpl
  type LayerNormImpl = org.bytedeco.pytorch.nn.modules.LayerNormImpl
  type BatchNorm1dImpl = org.bytedeco.pytorch.nn.modules.BatchNorm1dImpl
  type BatchNorm2dImpl = org.bytedeco.pytorch.nn.modules.BatchNorm2dImpl
  type GroupNormImpl = org.bytedeco.pytorch.nn.modules.GroupNormImpl
  type InstanceNorm2dImpl = org.bytedeco.pytorch.nn.modules.InstanceNorm2dImpl
  type DropoutImpl = org.bytedeco.pytorch.nn.modules.DropoutImpl
  type IdentityImpl = org.bytedeco.pytorch.nn.modules.IdentityImpl
  type ReLUImpl = org.bytedeco.pytorch.nn.modules.ReLUImpl
  type LeakyReLUImpl = org.bytedeco.pytorch.nn.modules.LeakyReLUImpl
  type PReLUImpl = org.bytedeco.pytorch.nn.modules.PReLUImpl
  type SigmoidImpl = org.bytedeco.pytorch.nn.modules.SigmoidImpl
  type TanhImpl = org.bytedeco.pytorch.nn.modules.TanhImpl
  type GELUImpl = org.bytedeco.pytorch.nn.modules.GELUImpl
  type SiLUImpl = org.bytedeco.pytorch.nn.modules.SiLUImpl
  type EmbeddingImpl = org.bytedeco.pytorch.nn.modules.EmbeddingImpl
  type EmbeddingBagImpl = org.bytedeco.pytorch.nn.modules.EmbeddingBagImpl
  type Conv1dImpl = org.bytedeco.pytorch.nn.modules.Conv1dImpl
  type Conv2dImpl = org.bytedeco.pytorch.nn.modules.Conv2dImpl
  type FlattenImpl = org.bytedeco.pytorch.nn.modules.FlattenImpl
  type GRUCellImpl = org.bytedeco.pytorch.nn.modules.GRUCellImpl
  type GRUImpl = org.bytedeco.pytorch.nn.modules.GRUImpl
  type LSTMImpl = org.bytedeco.pytorch.nn.modules.LSTMImpl
  // Container modules
  type SequentialImpl = org.bytedeco.pytorch.nn.modules.container.SequentialImpl
  type ModuleListImpl = org.bytedeco.pytorch.nn.modules.container.ModuleListImpl
  type ModuleDictImpl = org.bytedeco.pytorch.nn.modules.container.ModuleDictImpl
  type ParameterListImpl = org.bytedeco.pytorch.nn.modules.container.ParameterListImpl
  // Options
  type LinearOptions = org.bytedeco.pytorch.nn.options.LinearOptions
  type BatchNormOptions = org.bytedeco.pytorch.nn.options.BatchNormOptions
  type GRUOptions = org.bytedeco.pytorch.nn.options.GRUOptions
  type LSTMOptions = org.bytedeco.pytorch.nn.options.LSTMOptions
  type Conv1dOptions = org.bytedeco.pytorch.nn.options.Conv1dOptions
  type Conv2dOptions = org.bytedeco.pytorch.nn.options.Conv2dOptions
  type CrossEntropyLossOptions = org.bytedeco.pytorch.nn.options.CrossEntropyLossOptions
  type EmbeddingOptions = org.bytedeco.pytorch.nn.options.EmbeddingOptions
  // Optim
  type Adam = org.bytedeco.pytorch.optim.Adam
  type AdamOptions = org.bytedeco.pytorch.optim.options.AdamOptions
  type SGD = org.bytedeco.pytorch.optim.SGD
  type SGDOptions = org.bytedeco.pytorch.optim.options.SGDOptions
  type Optimizer = org.bytedeco.pytorch.optim.Optimizer
  // Data
  type TensorDataset = org.bytedeco.pytorch.data.datasets.TensorDataset
  type DataLoaderOptions = org.bytedeco.pytorch.data.options.DataLoaderOptions
  type FullDataLoaderOptions = org.bytedeco.pytorch.data.options.FullDataLoaderOptions
  // Distributed
  type ProcessGroupGloo = org.bytedeco.pytorch.distributed.ProcessGroupGloo
  type FileStore = org.bytedeco.pytorch.distributed.FileStore
  type TCPStore = org.bytedeco.pytorch.distributed.TCPStore
  type TCPStoreOptions = org.bytedeco.pytorch.distributed.TCPStoreOptions
  type Store = org.bytedeco.pytorch.distributed.Store
  type Work = org.bytedeco.pytorch.distributed.Work
  type AllgatherOptions = org.bytedeco.pytorch.distributed.AllgatherOptions
  type AllreduceOptions = org.bytedeco.pytorch.distributed.AllreduceOptions
  type BroadcastOptions = org.bytedeco.pytorch.distributed.BroadcastOptions
  type ReduceOp = org.bytedeco.pytorch.distributed.ReduceOp
  type ReduceScatterOptions = org.bytedeco.pytorch.distributed.ReduceScatterOptions
  type ProcessGroupNCCL = org.bytedeco.pytorch.distributed.ProcessGroupNCCL
  // Data loaders / samplers
  type BatchSizeSampler = org.bytedeco.pytorch.data.sampler.BatchSizeSampler
  type SequentialSampler = org.bytedeco.pytorch.data.sampler.SequentialSampler
  type RandomSampler = org.bytedeco.pytorch.data.sampler.RandomSampler
  type StreamSampler = org.bytedeco.pytorch.data.sampler.StreamSampler

  def cpu(): String = "cpu"
  def cuda(device: Int = 0): String = s"cuda:$device"

  def randn(sizes: Long*): Tensor =
    torch.randn(sizes.toArray, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  def rand(sizes: Long*): Tensor =
    torch.rand(sizes.toArray, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  def zeros(sizes: Long*): Tensor =
    torch.zeros(sizes.toArray, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  def ones(sizes: Long*): Tensor =
    torch.ones(sizes.toArray, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))

  def tensor(data: Array[Float], sizes: Long*): Tensor = {
    val flat = torch.tensor(data, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    if (sizes.length == 1 && sizes(0) == data.length) flat
    else flat.reshape(sizes.toArray: _*)
  }

  def tensor(data: Array[Int], sizes: Long*): Tensor = {
    val floatData = data.map(_.toFloat)
    tensor(floatData, sizes*)
  }

  def tensor(data: Array[Long], sizes: Long*): Tensor = {
    val n = data.length
    val f = data.map(_.toFloat)
    val flat = torch.tensor(f, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    val t = flat.toType(ScalarType.Long)
    flat.close()
    if (sizes.length == 1 && sizes(0) == n) t
    else { val r = t.reshape(sizes.toArray: _*); t.close(); r }
  }

  def longTensor(data: Array[Long]): Tensor = {
    val n = data.length
    val f = data.map(_.toFloat)
    val flat = torch.tensor(f, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    val t = flat.toType(ScalarType.Long)
    flat.close()
    t
  }

  def arange(start: Int, end: Int, step: Int = 1): Tensor =
    torch.arange(new Scalar(start), new Scalar(end), new Scalar(step))

  def toTensorVector(tensors: Seq[Tensor]): TensorVector = {
    val vec = new TensorVector(tensors.size.toLong)
    tensors.foreach(vec.push_back)
    vec
  }

  def toParameterVector(params: Seq[Tensor]): TensorVector = {
    val pv = new TensorVector(params.size.toLong)
    params.foreach(pv.push_back)
    pv
  }

  implicit class DeviceContext(val device: String) extends AnyVal {
    def asImplicit: String = device
  }

  object Implicits {
    implicit val cpuDevice: String = "cpu"
  }
}
