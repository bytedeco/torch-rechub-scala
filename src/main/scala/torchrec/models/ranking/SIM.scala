package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits.SeqTensorRichSeq
import torchrec.utils.DeviceSupport


import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Search-based Interest Model (SIM)
 * Reference: "Search-based User Interest Modeling with Sequential Behavior Data"
 * Alibaba, CIKM 2020
 *
 * SIM handles long user behavior sequences by:
 * 1. Category-based hard/soft filtering to reduce sequence length
 * 2. Attention-based aggregation of filtered history
 * 3. Combining user interest with target item for CTR prediction
 *
 * @param features         User/item sparse features
 * @param seqFeatures      Sequence features (item history)
 * @param cateFeatures     Sequence features (category history for filtering)
 * @param timeFeatures     Sequence features (time information)
 * @param embedDim         Embedding dimension
 * @param attentionUnits   Attention hidden units
 * @param mode             Filter mode: "hard" (category match) or "soft" (similarity threshold)
 * @param threshold        Soft mode similarity threshold [0, 1)
 * @param mlpDims          MLP hidden dimensions
 * @param dropout          Dropout rate
 * @param device           Device
 */
class SIM(
  features: List[Feature],
  seqFeatures: List[SequenceFeature],
  cateFeatures: List[SequenceFeature],
  timeFeatures: List[SequenceFeature],
  embedDim: Int = 8,
  attentionUnits: Int = 36,
  mode: String = "hard",
  threshold: Float = 0.8f,
  mlpDims: List[Long] = List(256L, 128L, 64L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(seqFeatures.nonEmpty, "seqFeatures cannot be empty")
  require(mode == "hard" || mode == "soft", "mode must be 'hard' or 'soft'")

  // Embedding layers
  private val featureEmbedding = new EmbeddingLayer(features, embedDim, device)
  register_module("featureEmbedding", featureEmbedding)

  private val seqEmbedding = new EmbeddingLayer(seqFeatures, embedDim, device)
  register_module("seqEmbedding", seqEmbedding)

  private val cateEmbedding = new EmbeddingLayer(cateFeatures, embedDim, device)
  register_module("cateEmbedding", cateEmbedding)

  private val timeEmbedding = new EmbeddingLayer(timeFeatures, embedDim, device)
  register_module("timeEmbedding", timeEmbedding)

  private val sparseDim = Features.calcSparseDim(features)

  // Attention unit: combines item and category embeddings
  private val attentionNet = new ActivationUnit(embedDim * 2, attentionUnits, "dice", device)
  register_module("attentionNet", attentionNet)

  // Projection for filtered sequence
  private val seqProj = new LinearImpl(embedDim * 2, embedDim)
  seqProj.to(new Device(device),false)
  register_module("seqProj", seqProj)

  // Final MLP
  // Input: sparse_features + aggregated_seq (with attention)
  private val totalDim = sparseDim + embedDim
  private val mlp = new MLP(totalDim, mlpDims, 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  private def normalizeSequenceEmb(raw: Tensor, batchSize: Int, seqLen: Int): Tensor = {
    if (raw.dim() == 4L && raw.size(1) == 1L) {
      raw.squeeze(1L)
    } else if (raw.dim() == 3L) {
      raw
    } else if (raw.dim() == 2L) {
      val total = raw.size(1)
      val expected = batchSize.toLong * seqLen.toLong * embedDim.toLong
      if (seqLen > 0 && total == seqLen.toLong * embedDim.toLong) {
        raw.view(batchSize.toLong, seqLen.toLong, embedDim.toLong)
      } else if (seqLen > 0 && raw.numel() == expected) {
        raw.view(batchSize.toLong, seqLen.toLong, embedDim.toLong)
      } else if (total == embedDim.toLong) {
        raw.unsqueeze(1L)
      } else {
        raw
      }
    } else {
      raw
    }
  }

  private def normalizeTargetEmb(raw: Tensor, batchSize: Int): Tensor = {
    if (raw.dim() == 3L && raw.size(1) == 1L) {
      raw.squeeze(1L)
    } else if (raw.dim() == 2L) {
      if (raw.size(1) == embedDim.toLong) raw else if (raw.numel() == batchSize.toLong * embedDim.toLong) raw.view(batchSize.toLong, embedDim.toLong) else raw
    } else {
      raw
    }
  }

  private def safeSequenceEmbedding(layer: EmbeddingLayer, name: String, indices: Tensor, batchSize: Int): Tensor = {
    val raw = layer.getSequenceEmbedding(name, indices)
    if (raw.dim() == 3L) raw
    else if (raw.dim() == 2L && raw.size(1) == embedDim.toLong) raw.unsqueeze(1L)
    else if (raw.dim() == 2L && raw.numel() == batchSize.toLong * indices.size(1) * embedDim.toLong) raw.view(batchSize.toLong, indices.size(1), embedDim.toLong)
    else raw
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    seqFeats: Map[String, Tensor],
    cateFeats: Map[String, Tensor],
    timeFeats: Map[String, Tensor],
    targetFeats: Map[String, Tensor]
  ): Tensor = {
    // Get sparse embeddings: (batch, sparse_dim)
    val featEmb = featureEmbedding.forward(sparseFeats)

    // Get target embeddings for attention: (batch, 1, embed_dim)
    val batchSize = seqFeats.values.headOption.map(_.size(0)).orElse(targetFeats.values.headOption.map(_.size(0))).getOrElse(1L).toInt
    val targetRaw = targetFeats.headOption match {
      case Some((name, tensor)) => seqEmbedding.getSequenceEmbedding(name, tensor)
      case None => throw new IllegalArgumentException("targetFeats cannot be empty")
    }
    val targetEmb = normalizeTargetEmb(targetRaw, batchSize)

    // Get sequence embeddings: (batch, seq_len, embed_dim)
    val seqEmbs = seqFeatures.map { f =>
      // use sequence-specific API
      normalizeSequenceEmb(safeSequenceEmbedding(seqEmbedding, f.name, seqFeats(f.name), batchSize), batchSize, seqFeats(f.name).size(1).toInt)
    }
    val seqEmb = if (seqEmbs.length == 1) seqEmbs.head else {
      val vec = new TensorVector()
      seqEmbs.foreach(vec.push_back)
      torch.cat(vec, 1)
    }

    // Get category embeddings for filtering: (batch, seq_len, embed_dim)
    val cateEmbs = cateFeatures.map { f =>
      cateEmbedding.getSequenceEmbedding(f.name, cateFeats(f.name))
    }
    val cateEmb = if (cateEmbs.length == 1) cateEmbs.head else {
      val vec = new TensorVector()
      cateEmbs.foreach(vec.push_back)
      torch.cat(vec, 1)
    }

    // Get time embeddings: (batch, seq_len, embed_dim)
    val timeEmbs = timeFeatures.map { f =>
      normalizeSequenceEmb(safeSequenceEmbedding(timeEmbedding, f.name, timeFeats(f.name), batchSize), batchSize, timeFeats(f.name).size(1).toInt)
    }
    val timeEmb = if (timeEmbs.length == 1) timeEmbs.head else {
      val vec = new TensorVector()
      timeEmbs.foreach(vec.push_back)
      torch.cat(vec, 1)
    }

    val seqLen = seqEmb.size(1).toInt

    // Target item + time embedding for matching
    val targetExpanded = targetEmb.unsqueeze(1).repeat(1, seqLen, 1)

    // Cat target item and time for matching: (batch, seq_len, embed_dim * 2)
    val targetCatTime = {
      val vec = new TensorVector()
      vec.push_back(targetExpanded)
      vec.push_back(timeEmb)
      torch.cat(vec, 2)
    }

    // Concatenate item and category for filtering: (batch, seq_len, embed_dim * 2)
    val seqCatCate = {
      val vec = new TensorVector()
      vec.push_back(seqEmb)
      vec.push_back(cateEmb)
      torch.cat(vec, 2)
    }

    // Apply category-based filtering
    val filteredSeq = if (mode == "hard") {
      // Hard mode: keep items whose category matches target category
      // We use cosine similarity between (target_item, target_time) and (seq_item, seq_cate)
      // as a soft proxy for hard category matching
      hardFilter(seqCatCate, targetCatTime, threshold, batchSize, seqLen)
    } else {
      // Soft mode: use similarity threshold
      softFilter(seqCatCate, targetCatTime, threshold, batchSize, seqLen)
    }

    // Apply attention to aggregate filtered sequence
    val attendedSeq = applyAttention(filteredSeq, targetExpanded, batchSize, seqLen)

    // Combine sparse features with attended sequence
    val combined = {
      val vec = new TensorVector()
      vec.push_back(featEmb)
      vec.push_back(attendedSeq)
      torch.cat(vec, 1L)
    }

    // MLP prediction
    val logits = mlp.forward(combined)
    logits
  }

  /** Hard filter: keep items with high category similarity using dot product */
  private def hardFilter(
    seqCatCate: Tensor,
    targetCatTime: Tensor,
    threshold: Float,
    batchSize: Int,
    seqLen: Int
  ): Tensor = {
    // Similarity between each history item and target
    val sim = seqCatCate.mul(targetCatTime).sum(2).unsqueeze(2)  // (batch, seq_len, 1)
    val mask = sim.gt(new Scalar(threshold)).toType(ScalarType.Float)  // (batch, seq_len, 1)
    // Zero out low-similarity items
    seqCatCate.mul(mask)
  }

  /** Soft filter: dot-product similarity weighted filtering */
  private def softFilter(
    seqCatCate: Tensor,
    targetCatTime: Tensor,
    threshold: Float,
    batchSize: Int,
    seqLen: Int
  ): Tensor = {
    // Use dot product as similarity score (simpler than cosine)
    val dotProd = seqCatCate.mul(targetCatTime).sum(2L).unsqueeze(2L)
    // Soft filter: just return dot-product weighted seq
    dotProd.mul(seqCatCate)
  }

  /** Apply attention over filtered sequence to get aggregated interest representation */
  private def applyAttention(
    filteredSeq: Tensor,
    target: Tensor,
    batchSize: Int,
    seqLen: Int
  ): Tensor = {
    // Project filteredSeq down to embedDim for attention
    val projected = seqProj.forward(filteredSeq)  // (batch, seq_len, embed_dim)

    // Scaled dot-product attention
    val attnWeights = projected.mul(target).sum(2).unsqueeze(2)  // (batch, seq_len, 1)
    val scale = new Scalar(scala.math.sqrt(embedDim.toDouble).toFloat)
    val attnNorm = attnWeights.div(scale).softmax(1)  // (batch, seq_len, 1)

    // Weighted sum of projected sequence
    val attended = projected.mul(attnNorm).sum(1)  // (batch, embed_dim)
    attended
  }
}
