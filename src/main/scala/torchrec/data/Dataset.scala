package torchrec.data

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import scala.jdk.CollectionConverters._
import scala.collection.mutable
import torchrec.Implicits._
import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * Base trait for datasets
 */
trait Dataset {
  def size: Long
  def get(index: Long): Batch
}

object Dataset {
  import org.bytedeco.pytorch._

  // Wrap a JavaDataset (Example-based) as a Scala Dataset
  private class JavaDatasetWrapper(javaDs: JavaDataset, order: Seq[String]) extends Dataset {
    override def size: Long = try { javaDs.size().get() } catch { case _: Throwable => 0L }

    override def get(index: Long): Batch = {
      try {
        val ex = javaDs.get(index)
        Batch.fromExample(ex, order)
      } catch {
        case _: Throwable => Batch.fromExample(javaDs.get(0L), order)
      }
    }
  }

  // Wrap a JavaTensorDataset (TensorExample-based) as a Scala Dataset
  private class JavaTensorDatasetWrapper(javaDs: JavaTensorDataset, order: Seq[String]) extends Dataset {
    override def size: Long = try { javaDs.size().get() } catch { case _: Throwable => 0L }

    override def get(index: Long): Batch = {
      try {
        val te = javaDs.get(index)
        Batch.fromTensorExample(te, order)
      } catch {
        case _: Throwable => Batch.fromTensorExample(javaDs.get(0L), order)
      }
    }
  }

  // Stream dataset wrapper: sequential access backed by JavaStreamDataset.get_batch
  private class JavaStreamDatasetWrapper(javaDs: JavaStreamDataset, order: Seq[String], fetchSize: Int = 32) extends Dataset {
    private var buffer = Seq.empty[Batch]
    private var baseIndex: Long = 0L

    override def size: Long = try { javaDs.size().get() } catch { case _: Throwable => 0L }

    override def get(index: Long): Batch = synchronized {
      if (index < baseIndex) {
        // cannot seek backwards in stream; not supported
        throw new UnsupportedOperationException("JavaStreamDataset wrapper does not support seeking backwards")
      }

      while (index >= baseIndex + buffer.length) {
        val vec = try { javaDs.get_batch(fetchSize) } catch { case _: Throwable => new ExampleVector() }
        val batches = Batch.fromExampleVector(vec, order)
        buffer = buffer ++ batches
      }

      buffer((index - baseIndex).toInt)
    }
  }

  // Stateful dataset wrapper: uses get_batch on JavaStatefulDataset and supports reset
  private class JavaStatefulDatasetWrapper(javaDs: JavaStatefulDataset, order: Seq[String], fetchSize: Int = 32) extends Dataset {
    private var buffer = Seq.empty[Batch]
    private var baseIndex: Long = 0L

    override def size: Long = try { javaDs.size().get() } catch { case _: Throwable => 0L }

    override def get(index: Long): Batch = synchronized {
      if (index < baseIndex) {
        try { javaDs.reset() } catch { case _: Throwable => () }
        buffer = Seq.empty
        baseIndex = 0L
      }

      while (index >= baseIndex + buffer.length) {
        val opt = try { javaDs.get_batch(fetchSize) } catch { case _: Throwable => null }
        val vec = if (opt == null) new ExampleVector() else try { opt.get() } catch { case _: Throwable => new ExampleVector() }
        val batches = Batch.fromExampleVector(vec, order)
        buffer = buffer ++ batches
      }

      buffer((index - baseIndex).toInt)
    }
  }

  // Factory helpers
  def fromJavaDataset(javaDs: JavaDataset, order: Seq[String]): Dataset = new JavaDatasetWrapper(javaDs, order)

  def fromJavaTensorDataset(javaDs: JavaTensorDataset, order: Seq[String]): Dataset = new JavaTensorDatasetWrapper(javaDs, order)

  def fromJavaStreamDataset(javaDs: JavaStreamDataset, order: Seq[String], fetchSize: Int = 32): Dataset = new JavaStreamDatasetWrapper(javaDs, order, fetchSize)

  def fromJavaStatefulDataset(javaDs: JavaStatefulDataset, order: Seq[String], fetchSize: Int = 32): Dataset = new JavaStatefulDatasetWrapper(javaDs, order, fetchSize)

  // Convenience: convert Scala Dataset to JavaDataset using the existing adapters
  def toJavaDataset(backing: Dataset): JavaDataset = JavaDatasetAdapters.createDatasetAdapter(backing)

  def toJavaTensorDataset(backing: Dataset): JavaTensorDataset = JavaTensorDatasetAdapters.createTensorDatasetAdapter(backing)

  def toJavaStatefulDataset(backing: Dataset): JavaStatefulDataset = JavaDatasetAdapters.createStatefulDatasetAdapter(backing)

  def toJavaStreamDataset(backing: Dataset): JavaStreamDataset = JavaDatasetAdapters.createStreamDatasetAdapter(backing)
}

/**
 * Batch containing features and labels
 */
case class Batch(
  sparseFeatures: Map[String, Tensor],
  denseFeatures: Map[String, Tensor] = Map.empty,
  sequenceFeatures: Map[String, Tensor] = Map.empty,
  labels: Option[Tensor] = None,
  // Sequence model fields
  tokens: Option[Tensor] = None,
  positions: Option[Tensor] = None,
  timeDiffs: Option[Tensor] = None,
  targets: Option[Tensor] = None,
  // Two-tower matching fields
  itemFeatures: Map[String, Tensor] = Map.empty,
  negItemFeatures: Option[Map[String, Tensor]] = None,
  // Multi-task fields
  taskLabels: Option[Map[String, Tensor]] = None
) {
  def to(device: String): Batch = {
    val d = new Device(device)
    def move(t: Tensor): Tensor = t.to(d, t.dtype())
    Batch(
      sparseFeatures.map { case (k, v) => k -> move(v) },
      denseFeatures.map { case (k, v) => k -> move(v) },
      sequenceFeatures.map { case (k, v) => k -> move(v) },
      labels.map(move),
      tokens.map(move),
      positions.map(move),
      timeDiffs.map(move),
      targets.map(move),
      itemFeatures.map { case (k, v) => k -> move(v) },
      negItemFeatures.map(_.map { case (k, v) => k -> move(v) }),
      taskLabels.map(_.map { case (k, v) => k -> move(v) })
    )
  }

  def numSamples: Long = {
    sparseFeatures.headOption.map(_._2.size(0))
      .orElse(denseFeatures.headOption.map(_._2.size(0)))
      .orElse(sequenceFeatures.headOption.map(_._2.size(0)))
      .orElse(tokens.map(_.size(0)))
      .getOrElse(0)
  }
}

object Batch {
  import org.bytedeco.pytorch.global.torch.ScalarType

  private def packBatchToIndexTensor(batch: Batch, sparseOrder: Seq[String], denseOrder: Seq[String] = Seq.empty, includeLabel: Boolean = false): Tensor = {
    val sparseVals = sparseOrder.map { name =>
      batch.sparseFeatures.get(name) match {
        case Some(t) =>
          val arr = t.toFloatArray
          if (arr.nonEmpty) arr(0) else 0.0f
        case None => 0.0f
      }
    }

    val denseVals = denseOrder.map { name =>
      batch.denseFeatures.get(name) match {
        case Some(t) =>
          val arr = t.toFloatArray
          if (arr.nonEmpty) arr(0) else 0.0f
        case None => 0.0f
      }
    }

    val labelVals = if (includeLabel) {
      batch.labels match {
        case Some(t) =>
          val arr = t.toFloatArray
          if (arr.nonEmpty) Array(arr(0)) else Array(0.0f)
        case None => Array(0.0f)
      }
    } else Array.emptyFloatArray

    val vals = (sparseVals ++ denseVals ++ labelVals).toArray
    if (vals.isEmpty) {
      torch.zeros(Array(1L): _*)
    } else {
      val ft = torchrec.Implicits.tensor(vals, Array(vals.length.toLong))
      ft.toType(ScalarType.Long)
    }
  }

  private def unpackIndexTensorToMaps(t: Tensor, sparseOrder: Seq[String], denseOrder: Seq[String], includeLabel: Boolean): (Map[String, Tensor], Map[String, Tensor], Option[Tensor]) = {
    val arr = try { t.toFloatArray } catch { case _: Throwable => Array.emptyFloatArray }
    val sparseLen = sparseOrder.length
    val denseLen = denseOrder.length
    val labelLen = if (includeLabel) 1 else 0
    val sparseVals = arr.slice(0, math.min(sparseLen, arr.length))
    val denseVals = arr.slice(sparseLen, math.min(sparseLen + denseLen, arr.length))
    val labelVals = if (labelLen == 1 && arr.length >= sparseLen + denseLen + 1) Some(arr(sparseLen + denseLen)) else None

    val sparseMap = sparseOrder.zipWithIndex.map { case (name, i) =>
      val v = if (i < sparseVals.length) sparseVals(i) else 0.0f
      val tt = torchrec.Implicits.tensor(Array(v), Array(1L)).toType(ScalarType.Long)
      name -> tt
    }.toMap

    val denseMap = denseOrder.zipWithIndex.map { case (name, i) =>
      val v = if (i < denseVals.length) denseVals(i) else 0.0f
      val tt = torchrec.Implicits.tensor(Array(v), Array(1L)).toType(ScalarType.Long)
      name -> tt
    }.toMap

    val labelTensor = labelVals.map(v => torchrec.Implicits.tensor(Array(v), Array(1L)).toType(ScalarType.Long))
    (sparseMap, denseMap, labelTensor)
  }

  // Convert a single Batch to an Example (Example.data is a 1-D Long tensor of feature indices)
  def toExample(batch: Batch, sparseOrder: Seq[String], denseOrder: Seq[String] = Seq.empty, includeLabel: Boolean = false): Example = {
    new Example(packBatchToIndexTensor(batch, sparseOrder, denseOrder, includeLabel))
  }

  def toTensorExample(batch: Batch, sparseOrder: Seq[String], denseOrder: Seq[String] = Seq.empty, includeLabel: Boolean = false): TensorExample = {
    new TensorExample(packBatchToIndexTensor(batch, sparseOrder, denseOrder, includeLabel))
  }

  // Convert Example/TensorExample back to Batch using provided feature order
  def fromExample(ex: Example, sparseOrder: Seq[String], denseOrder: Seq[String] = Seq.empty, includeLabel: Boolean = false): Batch = {
    val data = try { ex.data() } catch { case _: Throwable => ex.data() }
    val (sparse, dense, label) = unpackIndexTensorToMaps(data, sparseOrder, denseOrder, includeLabel)
    Batch(sparse, dense, Map.empty, label)
  }

  def fromTensorExample(te: TensorExample, sparseOrder: Seq[String], denseOrder: Seq[String] = Seq.empty, includeLabel: Boolean = false): Batch = {
    val data = try { te.data() } catch { case _: Throwable => te.data() }
    val (sparse, dense, label) = unpackIndexTensorToMaps(data, sparseOrder, denseOrder, includeLabel)
    Batch(sparse, dense, Map.empty, label)
  }

  // Vectors
  def fromExampleVector(vec: ExampleVector, order: Seq[String]): Seq[Batch] = {
    val n = vec.size().toInt
    (0 until n).map(i => fromExample(vec.get(i), order))
  }

  def fromTensorExampleVector(vec: TensorExampleVector, order: Seq[String]): Seq[Batch] = {
    val n = vec.size().toInt
    (0 until n).map(i => fromTensorExample(vec.get(i), order))
  }

  def toExampleVector(batches: Seq[Batch], order: Seq[String]): ExampleVector = {
    val vec = new ExampleVector()
    batches.foreach(b => vec.push_back(toExample(b, order)))
    vec
  }

  def toTensorExampleVector(batches: Seq[Batch], order: Seq[String]): TensorExampleVector = {
    val vec = new TensorExampleVector()
    batches.foreach(b => vec.push_back(toTensorExample(b, order)))
    vec
  }
}

/**
 * TensorDataset - wraps in-memory tensors
 */
class TensorDataset(
  val sparseFeatures: Map[String, Tensor],
  val denseFeatures: Map[String, Tensor] = Map.empty,
  val labels: Option[Tensor] = None
) extends Dataset {

  override def size: Long = {
    sparseFeatures.headOption.map(_._2.size(0)).getOrElse(0)
  }

  override def get(index: Long): Batch = {
    // Ensure index is within bounds
    val safeIndex = index.min(size - 1).max(0)
    // Return contiguous copies to avoid view-related tensor issues
    def getFeature(v: Tensor): Tensor = {
      val sliced = v.narrow(0, safeIndex, 1)
      val result = if (sliced.dim() == 0) sliced.unsqueeze(0) else sliced
      result.contiguous().clone()
    }
    Batch(
      sparseFeatures.map { case (k, v) => k -> getFeature(v) },
      denseFeatures.map { case (k, v) => k -> getFeature(v) },
      Map.empty,
      labels.map(l => getFeature(l))
    )
  }

  // Convenience: produce a JavaTensorDataset adapter for use with JavaCPP DataLoaders
  def asJavaTensorDataset(): JavaTensorDataset = JavaTensorDatasetAdapters.createTensorDatasetAdapter(this)
}

/**
 * SequenceDataset for sequence models
 */
class SequenceDataset(
  val features: Map[String, Tensor] = Map.empty,
  val sequenceFeatures: Map[String, Tensor] = Map.empty,
  val labels: Option[Tensor] = None,
  val positions: Option[Tensor] = None,
  val timeDiffs: Option[Tensor] = None,
  val tokens: Option[Tensor] = None,
  val targets: Option[Tensor] = None,
  val itemFeatures: Option[Map[String, Tensor]] = None
) extends Dataset {

  override def size: Long = {
    tokens.map(_.size(0))
      .orElse(features.headOption.map(_._2.size(0)))
      .orElse(sequenceFeatures.headOption.map(_._2.size(0)))
      .getOrElse(0)
  }

  override def get(index: Long): Batch = {
    Batch(
      features.map { case (k, v) => k -> v.select(0, index) },
      Map.empty,
      sequenceFeatures.map { case (k, v) => k -> v.select(0, index) },
      labels.map(_.select(0, index)),
      tokens.map(_.select(0, index)),
      positions.map(_.select(0, index)),
      timeDiffs.map(_.select(0, index)),
      targets.map(_.select(0, index)),
      itemFeatures.map(m => m.map { case (k, v) => k -> v.select(0, index) }).getOrElse(Map.empty)
    )
  }

  def asJavaDataset(): JavaDataset = JavaDatasetAdapters.createDatasetAdapter(this)
  def asJavaStatefulDataset(): JavaStatefulDataset = JavaDatasetAdapters.createStatefulDatasetAdapter(this)
}

/**
 * MatchingDataset for two-tower models
 */
class MatchingDataset(
  val userFeatures: Map[String, Tensor],
  val itemFeatures: Map[String, Tensor],
  val labels: Option[Tensor] = None,
  val negItemFeatures: Option[Map[String, Tensor]] = None,
  val tokens: Option[Tensor] = None,
  val positions: Option[Tensor] = None
) extends Dataset {

  override def size: Long = {
    // Return the max of user and item feature counts so the dataset can be
    // iterated either by users or by items (used for embedding extraction).
    val u = userFeatures.headOption.map(_._2.size(0)).getOrElse(0L)
    val it = itemFeatures.headOption.map(_._2.size(0)).getOrElse(0L)
    math.max(u, it)
  }

  override def get(index: Long): Batch = {
    // Return contiguous copies to avoid view-related tensor issues
    def safeNarrow(v: Tensor, idx: Long): Tensor = {
      val safeIdx = math.min(idx, v.size(0) - 1)
      val sliced = v.narrow(0, safeIdx, 1)
      val result = if (sliced.dim() == 0) sliced.unsqueeze(0) else sliced
      result.contiguous().clone()
    }

    Batch(
      userFeatures.map { case (k, v) => k -> safeNarrow(v, index) },
      Map.empty,
      Map.empty,
      labels.map(l => safeNarrow(l, index)),
      tokens.map(_.select(0, index).contiguous().clone()),
      positions.map(_.select(0, index).contiguous().clone()),
      None,
      None,
      itemFeatures.map { case (k, v) => k -> safeNarrow(v, index) },
      negItemFeatures.map(m => m.map { case (k, v) => k -> safeNarrow(v, index) }),
      None
    )
  }

  def asJavaDataset(): JavaDataset = JavaDatasetAdapters.createDatasetAdapter(this)
  def asJavaTensorDataset(): JavaTensorDataset = JavaTensorDatasetAdapters.createTensorDatasetAdapter(this)
}

/**
 * MultiTaskDataset for multi-task learning
 */
class MultiTaskDataset(
  val features: Map[String, Tensor],
  val taskLabels: Map[String, Tensor]
) extends Dataset {

  override def size: Long = {
    features.headOption.map(_._2.size(0)).getOrElse(0)
  }

  override def get(index: Long): Batch = {
    Batch(
      features.map { case (k, v) => k -> v.select(0, index) },
      Map.empty,
      Map.empty,
      None,
      taskLabels = Some(taskLabels.map { case (k, v) => k -> v.select(0, index) })
    )
  }

  def asJavaDataset(): JavaDataset = JavaDatasetAdapters.createDatasetAdapter(this)
}