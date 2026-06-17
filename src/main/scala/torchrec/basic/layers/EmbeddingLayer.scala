package torchrec.basic.layers

import torchrec.basic.features._
import torchrec.Implicits._
import torchrec.Implicits.SeqTensorRichSeq

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import torchrec.utils.DeviceSupport
import scala.jdk.CollectionConverters._
import scala.collection.mutable

/**
 * Embedding layer for sparse and sequence features
 * Uses ModuleDictImpl like PyTorch's nn.ModuleDict to store embedding tables
 */
class EmbeddingLayer(
  val features: List[Feature],
  val embedDim: Int = 8,
  val device: String = DeviceSupport.backend,
  val paddingIdx: Option[Long] = None,
  val sparse: Boolean = false
) extends Module {

  // Use ModuleDictImpl like PyTorch's nn.ModuleDict
  private val embedDict = new ModuleDictImpl()

  // Map for direct access (mirrors the ModuleDict content)
  private val embeddingTables = mutable.Map[String, EmbeddingImpl]()

  // Build embedding tables and register them to the Module
  features.foreach {
    case f: SparseFeature =>
      val key = s"embed_${f.name}"
      if (!f.sharedWith.exists(name => embeddingTables.contains(name)) && !embeddingTables.contains(key)) {
        val options = new EmbeddingOptions(f.vocabSize, f.embedDim)
        f.paddingIdx.foreach { idx =>
          options.padding_idx().put(idx)
        }
        val embedding = new EmbeddingImpl(options)
        if (device != "cpu") {
          embedding.to(new org.bytedeco.pytorch.Device(device), false)
        }
        register_module(key, embedding)
        embeddingTables(key) = embedding
      }

    case f: SequenceFeature =>
      val key = s"embed_seq_${f.name}"
      if (!f.sharedWith.exists(name => embeddingTables.contains(name)) && !embeddingTables.contains(key)) {
        val options = new EmbeddingOptions(f.vocabSize, f.embedDim)
        if (f.paddingIdx != 0) {
          options.padding_idx().put(f.paddingIdx)
        }
        val embedding = new EmbeddingImpl(options)
        if (device != "cpu") {
          embedding.to(new org.bytedeco.pytorch.Device(device), false)
        }
        register_module(key, embedding)
        embeddingTables(key) = embedding
      }

    case _ => // DenseFeature - no embedding table
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    sequenceFeats: Map[String, Tensor] = Map.empty,
    squeeze: Boolean = true
  ): Tensor = {
    val embeddingList = mutable.ListBuffer[Tensor]()

    // Process sparse features
    sparseFeats.foreach { case (name, indices) =>
      val embedKey = s"embed_$name"
      embeddingTables.get(embedKey).foreach { embed =>
        val embedDev = embed.weight().device()
        val idx1d = if (indices.dim() == 2L && indices.size(1) == 1L) {
          indices.squeeze(1L)
        } else {
          indices
        }
        val idxOnDev = if (idx1d.device().equals(embedDev)) {
          idx1d.toType(ScalarType.Long)
        } else {
          idx1d.toType(ScalarType.Long).to(embedDev, ScalarType.Long)
        }
        val emb = embed.forward(idxOnDev)
        val emb3d = if (emb.dim() == 2L) emb.unsqueeze(1L) else emb
        embeddingList += emb3d
      }
    }

    // Process sequence features
    sequenceFeats.foreach { case (name, indices) =>
      val embedKey = s"embed_seq_$name"
      embeddingTables.get(embedKey).foreach { embed =>
        val embedDev = embed.weight().device()
        val idxOnDev = if (indices.device().equals(embedDev)) {
          indices.toType(ScalarType.Long)
        } else {
          indices.toType(ScalarType.Long).to(embedDev, ScalarType.Long)
        }
        val emb = embed.forward(idxOnDev)
        val pooledEmb = poolSequence(emb, idxOnDev, "mean")
        embeddingList += pooledEmb
      }
    }

    if (embeddingList.isEmpty) {
      throw new IllegalArgumentException("No embeddings found for given features")
    }

    val batchSize = embeddingList.head.size(0).toInt
    val embedDims = embeddingList.map(e => if (e.dim() == 3L) e.size(2).toInt else e.size(1).toInt)
    val totalDim = embedDims.sum

    // Manually concatenate embeddings
    val resultArr = new Array[Float](batchSize * totalDim)
    var offset = 0
    for ((emb, f) <- embeddingList.zipWithIndex) {
      val embCpu = try {
        if (emb.is_cuda()) emb.cpu() else emb
      } catch {
        case _: Exception => emb.to(new Device("cpu"), ScalarType.Float)
      }
      val arr = embCpu.toFloatArray
      val thisDim = embedDims(f)
      var i = 0
      while (i < batchSize) {
        var k = 0
        while (k < thisDim) {
          resultArr(i * totalDim + offset + k) = arr(i * thisDim + k)
          k += 1
        }
        i += 1
      }
      offset += thisDim
    }
    val embedDev = embeddingList.head.device()
    val flattened = torchrec.TorchRec.tensor(resultArr, batchSize.toLong, totalDim.toLong).to(embedDev, ScalarType.Float)
    if (squeeze && embeddingList.size == 1) {
      flattened.squeeze(1L)
    } else {
      flattened
    }
  }

  private def poolSequence(emb: Tensor, indices: Tensor, pooling: String): Tensor = {
    val padIdx = paddingIdx.getOrElse(0L)
    if (indices.dim() == 1L) {
      return emb
    }
    pooling match {
      case "mean" =>
        val padTensor = torch.full(Array(1L), new Scalar(padIdx)).to(indices.device(), ScalarType.Long)
        val mask = indices.ne(padTensor).toType(ScalarType.Float)
        val sum = emb.mul(mask.unsqueeze(2L)).sum(1L)
        val count = mask.sum(1L).unsqueeze(1L)
        sum.div(count)

      case "sum" =>
        emb.sum(1L)

      case "max" =>
        val maxPair = torch.max(emb, 1L)
        maxPair.get0

      case "last" =>
        emb.mean(1L)

      case _ => emb.mean(1L)
    }
  }

  /** Get embedding for a single feature */
  def getEmbedding(name: String, indices: Tensor): Tensor = {
    val embedKey = s"embed_$name"
    embeddingTables.get(embedKey) match {
      case Some(embed) =>
        val idx1d = if (indices.dim() == 2L && indices.size(1) == 1L) {
          indices.squeeze(1L)
        } else if (indices.dim() == 1L) {
          indices
        } else {
          indices
        }
        val embedDev = embed.weight().device()
        val idxOnDev = if (idx1d.device().equals(embedDev)) {
          idx1d.toType(ScalarType.Long)
        } else {
          idx1d.toType(ScalarType.Long).to(embedDev, ScalarType.Long)
        }
        embed.forward(idxOnDev)
      case None =>
        throw new IllegalArgumentException(s"No embedding table for: $name")
    }
  }

  def to(device: String): this.type = {
    this
  }
}