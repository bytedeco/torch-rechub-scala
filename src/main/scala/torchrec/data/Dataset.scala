package torchrec.data

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.nn.modules._
import org.bytedeco.pytorch.nn.modules.container._
import org.bytedeco.pytorch.nn.options._
import org.bytedeco.pytorch.optim._
import org.bytedeco.pytorch.data._
import org.bytedeco.pytorch.data.datasets._
import org.bytedeco.pytorch.data.options._
import org.bytedeco.pytorch.data.sampler._
import org.bytedeco.pytorch.distributed._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.c10.SizeTArrayRef

import scala.jdk.CollectionConverters._
import scala.collection.mutable
import torchrec.Implicits._

/**
 * Base trait for datasets
 */
trait Dataset {
  def size: Long
  def get(index: Long): Batch
}

object Dataset {

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

  // Factory helpers
  def fromJavaTensorDataset(javaDs: JavaTensorDataset, order: Seq[String]): Dataset = new JavaTensorDatasetWrapper(javaDs, order)

  // Convenience: convert Scala Dataset to JavaTensorDataset using a basic adapter
  def toJavaTensorDataset(backing: Dataset): JavaTensorDataset = {
    new JavaTensorDataset() {
      override def get(index: Long): TensorExample = {
        val b = backing.get(index)
        Batch.toTensorExample(b, backing match {
          case td: TensorDataset => td.sparseFeatures.keys.toSeq.sorted
          case sd: SequenceDataset => sd.features.keys.toSeq.sorted
          case md: MatchingDataset => md.userFeatures.keys.toSeq.sorted
          case _ => Seq.empty[String]
        })
      }
      override def size(): org.bytedeco.pytorch.SizeTOptional = new org.bytedeco.pytorch.SizeTOptional(backing.size)
      override def get_batch(indices: SizeTArrayRef): TensorExampleVector = {
        val vec = new TensorExampleVector()
        val len = indices.size().toInt
        var i = 0
        while (i < len) {
          val idx = indices.get(i)
          val b = backing.get(idx)
          vec.push_back(Batch.toTensorExample(b, backing match {
            case td: TensorDataset => td.sparseFeatures.keys.toSeq.sorted
            case sd: SequenceDataset => sd.features.keys.toSeq.sorted
            case md: MatchingDataset => md.userFeatures.keys.toSeq.sorted
            case _ => Seq.empty[String]
          }))
          i += 1
        }
        vec
      }
    }
  }
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

  // Convert a single Batch to a TensorExample
  def toTensorExample(batch: Batch, sparseOrder: Seq[String], denseOrder: Seq[String] = Seq.empty, includeLabel: Boolean = false): TensorExample = {
    new TensorExample(packBatchToIndexTensor(batch, sparseOrder, denseOrder, includeLabel))
  }

  // Convert TensorExample back to Batch using provided feature order
  def fromTensorExample(te: TensorExample, sparseOrder: Seq[String], denseOrder: Seq[String] = Seq.empty, includeLabel: Boolean = false): Batch = {
    val data = try { te.data() } catch { case _: Throwable => te.data() }
    val (sparse, dense, label) = unpackIndexTensorToMaps(data, sparseOrder, denseOrder, includeLabel)
    Batch(sparse, dense, Map.empty, label)
  }

  // Vectors
  def fromTensorExampleVector(vec: TensorExampleVector, order: Seq[String]): Seq[Batch] = {
    val n = vec.size().toInt
    (0 until n).map(i => fromTensorExample(vec.get(i), order))
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
  def asJavaTensorDataset(): JavaTensorDataset = Dataset.toJavaTensorDataset(this)
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

  def asJavaTensorDataset(): JavaTensorDataset = Dataset.toJavaTensorDataset(this)
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
}