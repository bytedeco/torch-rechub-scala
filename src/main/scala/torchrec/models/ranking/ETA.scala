package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * End-to-End Target Attention (ETA)
 * Reference: "End-to-End User Behavior Modeling with Target Attention at Alibaba"
 * An improvement over SIM that uses Locality Sensitive Hashing (LSH) for
 * efficient retrieval of relevant items from long-term history.
 *
 * ETA replaces the two-stage approach of SIM with end-to-end target attention:
 * 1. Hash target item embedding to retrieve top-K similar historical items
 * 2. Apply target attention to aggregate retrieved items
 *
 * @param features         User/item sparse features
 * @param seqFeatures      Sequence features (short + long term history)
 * @param embedDim         Embedding dimension
 * @param hashSize         Hash size for LSH retrieval
 * @param attentionUnits   Attention hidden units
 * @param topK             Number of retrieved items to keep
 * @param mlpDims          MLP hidden dimensions
 * @param dropout          Dropout rate
 * @param device           Device
 */
class ETA(
  features: List[Feature],
  seqFeatures: List[SequenceFeature],
  embedDim: Int = 8,
  hashSize: Int = 64,
  attentionUnits: Int = 36,
  topK: Int = 20,
  mlpDims: List[Long] = List(256L, 128L, 64L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(seqFeatures.nonEmpty, "seqFeatures cannot be empty")

  // Embedding layers
  private val featureEmbedding = new EmbeddingLayer(features, embedDim, device)
  register_module("featureEmbedding", featureEmbedding)

  private val seqEmbedding = new EmbeddingLayer(seqFeatures, embedDim, device)
  register_module("seqEmbedding", seqEmbedding)

  private val sparseDim = Features.calcSparseDim(features)

  // Hash projection for LSH
  private val hashProjection = {
    val lp = new LinearImpl(embedDim, hashSize)
    lp.to(new Device(device), false)
    lp
  }
  register_module("hashProjection", hashProjection)

  // Activation unit for target attention
  private val attentionNet = new ActivationUnit(embedDim, attentionUnits, "dice", device)
  register_module("attentionNet", attentionNet)

  // Final MLP
  // Input: sparse_features + long_term_interest + short_term_interest
  private val totalDim = sparseDim + embedDim * 2
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

  private def safeSequenceEmbedding(name: String, indices: Tensor, batchSize: Int): Tensor = {
    val raw = seqEmbedding.getSequenceEmbedding(name, indices)
    if (raw.dim() == 3L) raw
    else if (raw.dim() == 2L && raw.size(1) == embedDim.toLong) raw.unsqueeze(1L)
    else if (raw.dim() == 2L && raw.numel() == batchSize.toLong * indices.size(1) * embedDim.toLong) raw.view(batchSize.toLong, indices.size(1), embedDim.toLong)
    else raw
  }

  private def alignLike(ref: Tensor, t: Tensor): Tensor = {
    try {
      val refDev = ref.device()
      if (t.device().equals(refDev)) t.toType(ScalarType.Float) else t.to(refDev, ScalarType.Float)
    } catch {
      case _: Throwable => t.toType(ScalarType.Float)
    }
  }

  private def onModelDevice(t: Tensor): Tensor = {
    try {
      if (device == "cpu") t.toType(ScalarType.Float) else t.to(new Device(device), ScalarType.Float)
    } catch {
      case _: Throwable => t.toType(ScalarType.Float)
    }
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    seqFeats: Map[String, Tensor],
    targetFeats: Map[String, Tensor]
  ): Tensor = {
    // Get sparse embeddings: (batch, sparse_dim)
    val featEmb = onModelDevice(featureEmbedding.forward(sparseFeats))

    // Get target embeddings: (batch, embed_dim)
    val batchSize = seqFeats.values.headOption.map(_.size(0)).orElse(targetFeats.values.headOption.map(_.size(0))).getOrElse(1L).toInt
    val targetRaw = targetFeats.headOption match {
      case Some((name, tensor)) => seqEmbedding.getSequenceEmbedding(name, tensor)
      case None => throw new IllegalArgumentException("targetFeats cannot be empty")
    }
    val targetEmb = onModelDevice(normalizeTargetEmb(targetRaw, batchSize))
    val targetFlat = if (targetEmb.dim() == 2L) targetEmb else targetEmb.mean(1)

    // Get sequence embeddings: (batch, seq_len, embed_dim)
    val seqEmbs = seqFeatures.map { f =>
      normalizeSequenceEmb(safeSequenceEmbedding(f.name, seqFeats(f.name), batchSize), batchSize, seqFeats(f.name).size(1).toInt)
    }
    val seqEmb = if (seqEmbs.length == 1) seqEmbs.head else {
      val vec = new TensorVector()
      seqEmbs.foreach(vec.push_back)
      torch.cat(vec, 1)
    }
    val seqEmbAligned = onModelDevice(seqEmb)

    val seqLen = seqEmb.size(1).toInt

    // LSH: hash target item
    val targetHashed = hashProjection.forward(targetFlat)
    // Sign-based LSH: positive -> 1, negative -> 0
    val targetSign = targetHashed.ge(new Scalar(0.0f)).toType(ScalarType.Float)

    // Hash all history items
    val seqFlat = seqEmbAligned.view(batchSize * seqLen, embedDim)
    val seqHashed = hashProjection.forward(seqFlat)
    val seqSign = seqHashed.ge(new Scalar(0.0f)).toType(ScalarType.Float)
    val seqHashed2D = seqSign.view(batchSize, seqLen, hashSize)

    // Compute Hamming similarity: count matching bits
    val matchMask = targetSign.unsqueeze(1).eq(seqHashed2D).toType(ScalarType.Float)
    val hammingSim = matchMask.sum(2L)  // (batch, seq_len)

    // Target attention over full sequence (simplified ETA without topk gather)
    val targetExpanded = onModelDevice(targetFlat.unsqueeze(1).repeat(1L, seqLen.toLong, 1L))  // (batch, seq_len, embed)

    // Scaled dot-product attention
    val attnScores = seqEmbAligned.mul(targetExpanded).sum(2L)  // (batch, seq_len)
    val combinedScores = attnScores.add(hammingSim)  // combine attention + LSH
    val attnWeights = combinedScores.unsqueeze(2).div(new Scalar(scala.math.sqrt(embedDim.toDouble).toFloat)).softmax(1L)  // (batch, seq_len, 1)

    // Attention-weighted sequence aggregation
    val attendedLong = seqEmbAligned.mul(attnWeights).sum(1L)  // (batch, embed)

    // Short-term: mean pool of sequence
    val attendedShort = seqEmbAligned.mean(1L)  // (batch, embed)

    // Combine all
    val combined = {
      val vec = new TensorVector()
      vec.push_back(onModelDevice(featEmb))
      vec.push_back(onModelDevice(attendedLong))
      vec.push_back(onModelDevice(attendedShort))
      torch.cat(vec, 1L)
    }

    // MLP prediction
    val logits = mlp.forward(combined)
    logits
  }
}
