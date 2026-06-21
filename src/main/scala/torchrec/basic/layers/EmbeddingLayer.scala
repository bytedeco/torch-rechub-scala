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

  private def sanitizeModuleName(name: String): String = {
    val cleaned = name.replaceAll("[^A-Za-z0-9_]+", "_")
    cleaned.stripPrefix("_").stripSuffix("_")
  }

  // Produce a canonical base name for a feature by stripping common prefixes
  // like seq_, embed_, feat_ to avoid double-prefixing when building table keys.
  private def baseName(name: String): String = {
    if (name == null) return ""
    val withoutPrefix = name.replaceAll("^(seq_|embed_|feat_)+", "")
    sanitizeModuleName(withoutPrefix)
  }

  // Use ModuleDictImpl like PyTorch's nn.ModuleDict
  private val embedDict = new ModuleDictImpl()

  // Map for direct access (mirrors the ModuleDict content)
  private val embeddingTables = mutable.Map[String, EmbeddingImpl]()
  // Track warnings printed for unknown feature sets to avoid flooding logs
  private val warnedMissingSparse = mutable.Set[String]()
  private val warnedMissingSeq = mutable.Set[String]()
  // Track warnings about invalid indices so we don't spam logs
  private val warnedInvalidIndices = mutable.Set[String]()

  private def safeDevice(t: Tensor): Device = {
    try t.device() catch { case _: Throwable => new Device("cpu") }
  }

  // Build embedding tables and register them to the Module
    features.foreach {
    case f: SparseFeature =>
      val key = s"embed_${baseName(f.name)}"
      if (!f.sharedWith.exists(name => embeddingTables.contains(name)) && !embeddingTables.contains(key)) {
        val options = new EmbeddingOptions(f.vocabSize, f.embedDim)
        f.paddingIdx.foreach { idx =>
          options.padding_idx().put(idx)
        }
        val embedding = new EmbeddingImpl(options)
        if (device != "cpu") {
          embedding.to(new org.bytedeco.pytorch.Device(device), false)
        }
        register_module(sanitizeModuleName(key), embedding)
        embeddingTables(key) = embedding
      }

    case f: SequenceFeature =>
      val key = s"embed_seq_${baseName(f.name)}"
      if (!f.sharedWith.exists(name => embeddingTables.contains(name)) && !embeddingTables.contains(key)) {
        val options = new EmbeddingOptions(f.vocabSize, f.embedDim)
        if (f.paddingIdx != 0) {
          options.padding_idx().put(f.paddingIdx)
        }
        val embedding = new EmbeddingImpl(options)
        if (device != "cpu") {
          embedding.to(new org.bytedeco.pytorch.Device(device), false)
        }
        register_module(sanitizeModuleName(key), embedding)
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

    // Filter incoming features to only those we have tables for, avoid noisy per-feature warnings
    val filteredSparse = sparseFeats.filter { case (name, _) => embeddingTables.contains(s"embed_${baseName(name)}") }
    val missingSparse = sparseFeats.keySet -- filteredSparse.keySet
    if (missingSparse.nonEmpty) {
      val key = missingSparse.mkString(",")
      if (!warnedMissingSparse.contains(key)) {
        System.err.println(s"[WARNING] Ignoring unknown sparse features: ${missingSparse.mkString(", ")}. Available tables: ${embeddingTables.keys.mkString(", ")}")
        warnedMissingSparse += key
      }
    }

    // Process sparse features that we actually have tables for
    filteredSparse.foreach { case (name, indices) =>
      val embedKey = s"embed_${baseName(name)}"
      embeddingTables.get(embedKey).foreach { embed =>
        try {
          val embedDev = embed.weight().device()
          val idx1d = if (indices.dim() == 2L && indices.size(1) == 1L) {
            indices.squeeze(1L)
          } else {
            indices
          }
          val idxDev = safeDevice(idx1d)
          var idxOnDev = if (idxDev.equals(embedDev)) {
            idx1d.toType(ScalarType.Long)
          } else {
            idx1d.toType(ScalarType.Long).to(embedDev, ScalarType.Long)
          }
          // Defensive: clamp indices into [0, num_embeddings-1] to avoid device-side gather OOB
          try {
            val numEmb = embed.weight().size(0)
            val maxIdx = numEmb - 1
            val anyLow = idxOnDev.lt(new Scalar(0L)).any().item().toDouble
            val anyHigh = idxOnDev.gt(new Scalar(maxIdx)).any().item().toDouble
            if ((anyLow != 0.0 || anyHigh != 0.0) && !warnedInvalidIndices.contains(embedKey)) {
              System.err.println(s"[WARNING] EmbeddingLayer: indices for '$embedKey' contain out-of-range values. Clamping to [0,$maxIdx].")
              warnedInvalidIndices += embedKey
            }
            idxOnDev = idxOnDev.clamp(min = new ScalarOptional(new Scalar(0L)), max = new ScalarOptional(new Scalar(maxIdx)))
          } catch { case _: Throwable => /* ignore clamp failures and proceed */ }
          val emb = embed.forward(idxOnDev)
          val emb3d = if (emb.dim() == 2L) emb.unsqueeze(1L) else emb
          embeddingList += emb3d
        } catch {
          case e: Exception =>
            System.err.println(s"[WARNING] Failed to embed feature '$name': ${e.getMessage}")
        }
      }
    }

    // Filter and process sequence features
    val filteredSeq = sequenceFeats.filter { case (name, _) => embeddingTables.contains(s"embed_seq_${baseName(name)}") }
    val missingSeq = sequenceFeats.keySet -- filteredSeq.keySet
    if (missingSeq.nonEmpty) {
      val key = missingSeq.mkString(",")
      if (!warnedMissingSeq.contains(key)) {
        System.err.println(s"[WARNING] Ignoring unknown sequence features: ${missingSeq.mkString(", ")}. Available tables: ${embeddingTables.keys.mkString(", ")}")
        warnedMissingSeq += key
      }
    }
    filteredSeq.foreach { case (name, indices) =>
      val embedKey = s"embed_seq_${baseName(name)}"
      embeddingTables.get(embedKey).foreach { embed =>
        try {
          val embedDev = embed.weight().device()
          val idxDev = safeDevice(indices)
          var idxOnDev = if (idxDev.equals(embedDev)) {
            indices.toType(ScalarType.Long)
          } else {
            indices.toType(ScalarType.Long).to(embedDev, ScalarType.Long)
          }
          // Defensive clamp for sequence indices as well
          try {
            val numEmb = embed.weight().size(0)
            val maxIdx = numEmb - 1
            val anyLow = idxOnDev.lt(new Scalar(0L)).any().item().toDouble
            val anyHigh = idxOnDev.gt(new Scalar(maxIdx)).any().item().toDouble
            if ((anyLow != 0.0 || anyHigh != 0.0) && !warnedInvalidIndices.contains(embedKey)) {
              System.err.println(s"[WARNING] EmbeddingLayer: sequence indices for '$embedKey' contain out-of-range values. Clamping to [0,$maxIdx].")
              warnedInvalidIndices += embedKey
            }
            idxOnDev = idxOnDev.clamp(min = new ScalarOptional(new Scalar(0L)), max = new ScalarOptional(new Scalar(maxIdx)))
          } catch { case _: Throwable => }
           val emb = embed.forward(idxOnDev)
           val pooledEmb = poolSequence(emb, idxOnDev, "mean")
           // Ensure sequence pooled embeddings have shape (batch, 1, embedDim)
           val emb3d = if (pooledEmb.dim() == 2L) pooledEmb.unsqueeze(1L) else pooledEmb
           embeddingList += emb3d
        } catch {
          case e: Exception =>
            System.err.println(s"[WARNING] Failed to embed sequence feature '$name': ${e.getMessage}")
        }
      }
    }

    if (embeddingList.isEmpty) {
      val availableTables = embeddingTables.keys.mkString(", ")
      val inputFeats = (sparseFeats.keys ++ sequenceFeats.keys).mkString(", ")
      throw new IllegalArgumentException(
        s"No embeddings found for given features. Input features: [$inputFeats], Available embedding tables: [$availableTables]"
      )
    }

    val batchSize = embeddingList.head.size(0).toInt
    val embedDims = embeddingList.map(e => if (e.dim() == 3L) e.size(2).toInt else e.size(1).toInt)
    val totalDim = embedDims.sum

    // Use GPU cat for concatenation - much more efficient than manual CPU copy
    val embeddingsArr = new Array[Tensor](embeddingList.size)
    embeddingList.copyToArray(embeddingsArr)
    // Ensure all embeddings are on the same device before concatenation
    val targetDev = safeDevice(embeddingList.head)
//    println(s"[EmbeddingLayer] concatenating ${embeddingsArr.length} embeddings on targetDev=$targetDev")
    val vec = new TensorVector()
    var idx = 0
    embeddingsArr.foreach { t =>
      try {
//        println(s"[EmbeddingLayer] embedding[$idx] device=${t.device()} shape=${t.sizes().toString}")
      } catch { case _: Throwable => println(s"[EmbeddingLayer] embedding[$idx] <error obtaining device/shape>") }
      val onDev = if (safeDevice(t).equals(targetDev)) t else t.to(targetDev, t.dtype())
      vec.push_back(onDev)
      idx += 1
    }
    val concatenated = torch.cat(vec, 1L)
    // Defensive reshape: sometimes individual embeddings may carry an unexpected
    // extra dimension (e.g. sequence pooling returned flattened values) which
    // causes concatenated.numel() != batchSize * totalDim and leads to a
    // confusing view/reshape error at runtime. Compute the real per-batch
    // feature width from the actual number of elements when possible so the
    // library is more robust to upstream shape variations and gives better
    // diagnostic output.
    val actualNumel = concatenated.numel().toLong
    val expectedNumel = batchSize.toLong * totalDim.toLong
    val finalTotalDim = if (actualNumel % batchSize.toLong == 0L) (actualNumel / batchSize.toLong).toInt else totalDim
    if (actualNumel != expectedNumel) {
      try { System.err.println(s"[EmbeddingLayer DEBUG] concatenated.numel()=$actualNumel expected=$expectedNumel; falling back to per-batch dim=$finalTotalDim") } catch { case _: Throwable => () }
    }
    val flattened = try {
      concatenated.contiguous().view(batchSize.toLong, finalTotalDim)
    } catch {
      case e: Throwable =>
        // Provide a clearer error message with shapes of all embeddings to aid debugging
        try {
          val shapes = embeddingList.map(e => if (e == null) "null" else try e.sizes().toString catch { case _: Throwable => "<unknown>" }).mkString(",")
          System.err.println(s"[EmbeddingLayer ERROR] Failed to view concatenated into (batch=$batchSize, dim=$finalTotalDim). concatenated.sizes=${concatenated.sizes()} embeddingShapes=[$shapes]")
        } catch { case _: Throwable => () }
        throw e
    }
    if (squeeze && embeddingList.size == 1) flattened.squeeze(1L) else flattened
  }

  /**
   * Forward pass returning 3D tensor (batch, num_fields, embed_dim)
   * This is needed for models like AFM, AutoInt that require field dimension
   */
  def forward3D(
    sparseFeats: Map[String, Tensor],
    sequenceFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    val embeddingList = mutable.ListBuffer[Tensor]()

    // Process sparse features
    sparseFeats.foreach { case (name, indices) =>
      val embedKey = s"embed_${baseName(name)}"
      embeddingTables.get(embedKey).foreach { embed =>
        val embedDev = embed.weight().device()
        val idx1d = if (indices.dim() == 2L && indices.size(1) == 1L) {
          indices.squeeze(1L)
        } else {
          indices
        }
        val idxDev = safeDevice(idx1d)
        var idxOnDev = if (idxDev.equals(embedDev)) {
          idx1d.toType(ScalarType.Long)
        } else {
          idx1d.toType(ScalarType.Long).to(embedDev, ScalarType.Long)
        }
        try {
          val numEmb = embed.weight().size(0)
          val maxIdx = numEmb - 1
          val anyLow = idxOnDev.lt(new Scalar(0L)).any().item().toDouble
          val anyHigh = idxOnDev.gt(new Scalar(maxIdx)).any().item().toDouble
          if ((anyLow != 0.0 || anyHigh != 0.0) && !warnedInvalidIndices.contains(embedKey)) {
            System.err.println(s"[WARNING] EmbeddingLayer.forward3D: indices for '$embedKey' contain out-of-range values. Clamping to [0,$maxIdx].")
            warnedInvalidIndices += embedKey
          }
          idxOnDev = idxOnDev.clamp(min = new ScalarOptional(new Scalar(0L)), max = new ScalarOptional(new Scalar(maxIdx)))
        } catch { case _: Throwable => }
        val emb = embed.forward(idxOnDev)
        val emb3d = if (emb.dim() == 2L) emb.unsqueeze(1L) else emb
        embeddingList += emb3d
      }
    }

    // Process sequence features
    sequenceFeats.foreach { case (name, indices) =>
      val embedKey = s"embed_seq_${baseName(name)}"
      embeddingTables.get(embedKey).foreach { embed =>
        val embedDev = embed.weight().device()
        val idxDev = safeDevice(indices)
        var idxOnDev = if (idxDev.equals(embedDev)) {
          indices.toType(ScalarType.Long)
        } else {
          indices.toType(ScalarType.Long).to(embedDev, ScalarType.Long)
        }
        try {
          val numEmb = embed.weight().size(0)
          val maxIdx = numEmb - 1
          val anyLow = idxOnDev.lt(new Scalar(0L)).any().item().toDouble
          val anyHigh = idxOnDev.gt(new Scalar(maxIdx)).any().item().toDouble
          if ((anyLow != 0.0 || anyHigh != 0.0) && !warnedInvalidIndices.contains(embedKey)) {
            System.err.println(s"[WARNING] EmbeddingLayer.forward3D: sequence indices for '$embedKey' contain out-of-range values. Clamping to [0,$maxIdx].")
            warnedInvalidIndices += embedKey
          }
          idxOnDev = idxOnDev.clamp(min = new ScalarOptional(new Scalar(0L)), max = new ScalarOptional(new Scalar(maxIdx)))
        } catch { case _: Throwable => }
        val emb = embed.forward(idxOnDev)
        val pooledEmb = poolSequence(emb, idxOnDev, "mean")
        val emb3d = if (pooledEmb.dim() == 2L) pooledEmb.unsqueeze(1L) else pooledEmb
        embeddingList += emb3d
      }
    }

    if (embeddingList.isEmpty) {
      throw new IllegalArgumentException("No embeddings found for given features")
    }

    // Stack embeddings along field dimension: (batch, num_fields, embed_dim)
    val embeddingsArr = new Array[Tensor](embeddingList.size)
    embeddingList.copyToArray(embeddingsArr)
    // Align devices
    val targetDev = embeddingList.head.device()
//    println(s"[EmbeddingLayer.forward3D] concatenating ${embeddingsArr.length} embeddings on targetDev=$targetDev")
    val vec = new TensorVector()
    var j = 0
    embeddingsArr.foreach { t =>
      try {
//        println(s"[EmbeddingLayer.forward3D] embedding[$j] device=${t.device()} shape=${t.sizes().toString}")
      } catch { case _: Throwable => println(s"[EmbeddingLayer.forward3D] embedding[$j] <error obtaining device/shape>") }
      val onDev = if (t.device().equals(targetDev)) t else t.to(targetDev, t.dtype())
      vec.push_back(onDev)
      j += 1
    }
    torch.cat(vec, 1L)
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
    val embedKey = s"embed_${baseName(name)}"
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
        var idxOnDev = if (idx1d.device().equals(embedDev)) {
          idx1d.toType(ScalarType.Long)
        } else {
          idx1d.toType(ScalarType.Long).to(embedDev, ScalarType.Long)
        }
        try {
          val numEmb = embed.weight().size(0)
          val maxIdx = numEmb - 1
          val anyLow = idxOnDev.lt(new Scalar(0L)).any().item().toDouble
          val anyHigh = idxOnDev.gt(new Scalar(maxIdx)).any().item().toDouble
          if ((anyLow != 0.0 || anyHigh != 0.0) && !warnedInvalidIndices.contains(embedKey)) {
            System.err.println(s"[WARNING] EmbeddingLayer.getEmbedding: indices for '$embedKey' contain out-of-range values. Clamping to [0,$maxIdx].")
            warnedInvalidIndices += embedKey
          }
          idxOnDev = idxOnDev.clamp(min = new ScalarOptional(new Scalar(0L)), max = new ScalarOptional(new Scalar(maxIdx)))
        } catch { case _: Throwable => }
        embed.forward(idxOnDev)
      case None =>
        throw new IllegalArgumentException(s"No embedding table for: $name")
    }
  }

  /** Get embedding for a single sequence feature. */
  def getSequenceEmbedding(name: String, indices: Tensor): Tensor = {
    val embedKey = s"embed_seq_${baseName(name)}"
    embeddingTables.get(embedKey) match {
      case Some(embed) =>
        val embedDev = embed.weight().device()
        var idxOnDev = if (indices.device().equals(embedDev)) {
          indices.toType(ScalarType.Long)
        } else {
          indices.toType(ScalarType.Long).to(embedDev, ScalarType.Long)
        }
        // Defensive clamping: ensure all indices are within [0, num_embeddings-1]
        // to prevent device-side assert in the CUDA gather kernel.
        val numEmb = embed.weight().size(0)
        val maxIdx = numEmb - 1
        val anyLow = idxOnDev.lt(new Scalar(0L)).any().item().toDouble
        val anyHigh = idxOnDev.gt(new Scalar(maxIdx)).any().item().toDouble
        if (anyLow != 0.0 || anyHigh != 0.0) {
          if (!warnedInvalidIndices.contains(embedKey)) {
            System.err.println(s"[WARNING] EmbeddingLayer.getSequenceEmbedding: indices for '$embedKey' contain out-of-range values. Clamping to [0,$maxIdx].")
            warnedInvalidIndices += embedKey
          }
          idxOnDev = idxOnDev.clamp(min = new ScalarOptional(new Scalar(0L)), max = new ScalarOptional(new Scalar(maxIdx)))
        }
        embed.forward(idxOnDev)
      case None =>
        throw new IllegalArgumentException(s"No sequence embedding table for: $name")
    }
  }

  /**
   * Forward pass for sequence features only, without pooling.
   * Returns 3D tensor (batch, seqLen, embedDim).
   * Used by BST model that needs raw sequence embeddings for self-attention.
   */
  def forwardSeqRaw(
    sequenceFeats: Map[String, Tensor]
  ): Tensor = {
    val embeddingList = mutable.ListBuffer[Tensor]()

    // Process sequence features without pooling
    sequenceFeats.foreach { case (name, indices) =>
      val embedKey = s"embed_seq_${baseName(name)}"
      embeddingTables.get(embedKey).foreach { embed =>
        val embedDev = embed.weight().device()
        var idxOnDev = if (indices.device().equals(embedDev)) {
          indices.toType(ScalarType.Long)
        } else {
          indices.toType(ScalarType.Long).to(embedDev, ScalarType.Long)
        }
        // Defensive clamping: ensure all indices are within [0, num_embeddings-1]
        // to prevent device-side assert in the CUDA gather kernel.
        val numEmb = embed.weight().size(0)
        val maxIdx = numEmb - 1
        val anyLow = idxOnDev.lt(new Scalar(0L)).any().item().toDouble
        val anyHigh = idxOnDev.gt(new Scalar(maxIdx)).any().item().toDouble
        if (anyLow != 0.0 || anyHigh != 0.0) {
          if (!warnedInvalidIndices.contains(embedKey)) {
            System.err.println(s"[WARNING] EmbeddingLayer.forwardSeqRaw: sequence indices for '$embedKey' contain out-of-range values. Clamping to [0,$maxIdx].")
            warnedInvalidIndices += embedKey
          }
          idxOnDev = idxOnDev.clamp(min = new ScalarOptional(new Scalar(0L)), max = new ScalarOptional(new Scalar(maxIdx)))
        }
        val emb = embed.forward(idxOnDev)
        embeddingList += emb
      }
    }

    if (embeddingList.isEmpty) {
      throw new IllegalArgumentException("No sequence embeddings found for given features")
    }

    // Stack embeddings along field dimension: (batch, num_features, seqLen, embedDim)
    // Always return 4D tensor for consistency
    val embeddingsArr = new Array[Tensor](embeddingList.size)
    embeddingList.copyToArray(embeddingsArr)

    if (embeddingList.size == 1) {
      // Single feature: return (batch, seqLen, embedDim)
      embeddingsArr.head
    } else {
      // Multiple sequence features: concatenate along the embedding dimension (last dim)
      // so result is (batch, seqLen, num_features * embedDim)
      val targetDev = embeddingList.head.device()
      val vec = new TensorVector()
      embeddingsArr.foreach { t =>
        val onDev = if (t.device().equals(targetDev)) t else t.to(targetDev, t.dtype())
        vec.push_back(onDev)
      }
      torch.cat(vec, 2)
    }
  }

  def to(device: String): this.type = {
    this
  }
}