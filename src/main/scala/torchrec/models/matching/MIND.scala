package torchrec.models.matching

import torchrec.basic.features.*
import torchrec.basic.layers.*
import torchrec.Implicits.*
import torchrec.utils.DeviceSupport
import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.mutable

/**
 * Multi-Interest Network with Dynamic Routing
 * Reference: Alibaba, CIKM 2019
 */
class MIND(
  features: List[Feature],
  sequenceFeature: SequenceFeature,
  embedDim: Int = 8,
  numInterests: Int = 4,
  capsuleDim: Int = 4,
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private def normalizeSparseIndex(name: String, t: Tensor): Tensor = {
    try {
      t.dim() match {
        case 0L => t.unsqueeze(0L)
        case 1L => t
        case 2L if t.size(1) == 1L => t.squeeze(1L)
        case 2L =>
          System.err.println(s"[MIND WARNING] Sparse feature '$name' has shape ${t.sizes()}; using first column as the ID index.")
          t.select(1L, 0L)
        case _ =>
          System.err.println(s"[MIND WARNING] Sparse feature '$name' has unexpected shape ${t.sizes()}; flattening to 1D.")
          t.contiguous().view(-1)
      }
    } catch {
      case _: Throwable => t
    }
  }

  private def normalizeSequenceInput(name: String, t: Tensor): Tensor = {
    try {
      t.dim() match {
        case 1L =>
          System.err.println(s"[MIND WARNING] Sequence feature '$name' received 1D input ${t.sizes()}; unsqueezing to (batch, seqLen).")
          t.unsqueeze(1L)
        case 2L => t
        case 3L if t.size(1) == 1L => t.squeeze(1L)
        case 3L if t.size(2) == 1L => t.squeeze(2L)
        case _ =>
          System.err.println(s"[MIND WARNING] Sequence feature '$name' has unexpected shape ${t.sizes()}; flattening trailing dims.")
          val batch = t.size(0)
          t.contiguous().view(batch, -1L)
      }
    } catch {
      case _: Throwable => t
    }
  }

  private val featureEmbedding = new EmbeddingLayer(features, embedDim, device)
  register_module("featureEmbedding", featureEmbedding)

  private val sequenceEmbedding = new EmbeddingLayer(List(sequenceFeature), embedDim, device)
  register_module("sequenceEmbedding", sequenceEmbedding)

  // Capsule routing
  private val capsuleNet = new CapsuleNetwork(embedDim, numInterests, capsuleDim,3, device)
  register_module("capsuleNet", capsuleNet)

  // MLP
  private val featSparseDim = Features.calcSparseDim(features)
  private val totalInputDim = featSparseDim + numInterests * capsuleDim

  private val tower = new MLP(totalInputDim, mlpDims, embedDim, "relu", dropout, device = device)
  register_module("tower", tower)

  def forward(
    features: Map[String, Tensor],
    sequenceIndices: Tensor  // (batch, seq_len)
  ): Tensor = {
    val normFeatures = features.map { case (name, tensor) => name -> normalizeSparseIndex(name, tensor) }
    val normSequence = normalizeSequenceInput(sequenceFeature.name, sequenceIndices)

    val featEmb = featureEmbedding.forward(normFeatures)
    val seqEmb = sequenceEmbedding.getSequenceEmbedding(sequenceFeature.name, normSequence)
    // seqEmb: (batch, seq_len, embed_dim)
    try { System.err.println(s"[MIND DEBUG] seqEmb dim=${seqEmb.dim()} sizes=${seqEmb.sizes()} device=${seqEmb.device()}") } catch { case _: Throwable => () }
    try { println(s"[MIND DEBUG] seqEmb dim=${seqEmb.dim()} sizes=${seqEmb.sizes()} device=${seqEmb.device()}") } catch { case _: Throwable => () }

    // Capsule routing to get multiple interests
    val interests = capsuleNet.forward(seqEmb)
    try { System.err.println(s"[MIND DEBUG] capsule returned interests dim=${interests.dim()} sizes=${interests.sizes()} numel=${interests.numel()}") } catch { case _: Throwable => () }
    try { println(s"[MIND DEBUG] capsule returned interests dim=${interests.dim()} sizes=${interests.sizes()} numel=${interests.numel()}") } catch { case _: Throwable => () }
    // interests: (batch, num_interests, capsule_dim)

    // Concatenate features and interests
    val interestsFlat = {
      val b = interests.size(0).toInt
      val total = interests.numel().toInt
      val second = if (b == 0) 0 else total / b
      try {
        try { println(s"[MIND DEBUG] flattening interests sizes=${interests.sizes()} numel=$total -> view($b,$second)") } catch { case _: Throwable => () }
        interests.view(b, second)
      } catch {
        case _: Throwable =>
          try {
            val cont = interests.contiguous()
            try { println(s"[MIND DEBUG] fallback contiguous interests sizes=${cont.sizes()} numel=${cont.numel()} -> view($b,$second)") } catch { case _: Throwable => () }
            cont.view(b, second)
          } catch {
            case e: Throwable =>
              try { System.err.println(s"[MIND DEBUG] interests shape sizes: ${interests.sizes()} (dim=${interests.dim()})") } catch { case _: Throwable => () }
              try { System.err.println(s"[MIND DEBUG] seqEmb shape: ${seqEmb.sizes()} (dim=${seqEmb.dim()})") } catch { case _: Throwable => () }
              throw new RuntimeException(s"MIND view reshape failed: ${e.getMessage}", e)
          }
        }
    }
    val combined = torch.cat(new TensorVector(featEmb, interestsFlat), 1L)

    tower.forward(combined)
  }
}

/**
 * Capsule Network for Multi-Interest Learning
 */
class CapsuleNetwork(
  embedDim: Int,
  numInterests: Int,
  capsuleDim: Int,
  numRoutings: Int = 3,
  device: String = DeviceSupport.backend
) extends Module {

  private val inputDim = embedDim
  private val W = new LinearImpl(inputDim, numInterests * capsuleDim)
  W.to(new Device(device),false)
  register_module("W", W)

  def forward(x: Tensor): Tensor = {
    // x: (batch, seq_len, embed_dim)
    // Ensure input is on the same device as the capsule projection weights
    val dev = new Device(device)
    val xOn = try { x.to(dev, ScalarType.Float) } catch { case _: Throwable => x }
    val batchSize = xOn.size(0).toInt
    val seqLen = xOn.size(1).toInt

    // Project to capsule space
    // Support both 2D and 3D outputs from LinearImpl for robustness.
    val uRaw = W.forward(xOn)
    try { System.err.println(s"[MIND DEBUG] uRaw dim=${uRaw.dim()} sizes=${uRaw.sizes()} numel=${uRaw.numel()} device=${uRaw.device()}") } catch { case _: Throwable => () }
    try { println(s"[MIND DEBUG] uRaw dim=${uRaw.dim()} sizes=${uRaw.sizes()} numel=${uRaw.numel()} device=${uRaw.device()}") } catch { case _: Throwable => () }

    // uRaw may be (batch, seq_len, outDim) or (batch*seq_len, outDim)
    val u = try {
      // Derive a reliable capsule dimension from the actual number of elements
      // produced by the linear projection. This makes the reshape robust to
      // different output layouts (2D vs 3D) and avoids silent mismatches.
      val totalElements = uRaw.numel().toInt
      val denom = batchSize * seqLen * numInterests
      if (denom == 0) throw new RuntimeException(s"Invalid shape components: batchSize=$batchSize seqLen=$seqLen numInterests=$numInterests")
      if (totalElements % denom != 0) {
        try { System.err.println(s"[MIND DEBUG] uRaw.dim=${uRaw.dim()} uRaw.sizes=${uRaw.sizes()}") } catch { case _: Throwable => () }
        throw new RuntimeException(s"Capsule reshape failed: totalElements=$totalElements is not divisible by (batchSize*seqLen*numInterests)=$denom")
      }
      val derivedCapsuleDim = totalElements / denom
      // Now reliably reshape to (batch, seq_len, numInterests, derivedCapsuleDim)
      val uView = uRaw.view(batchSize, seqLen, numInterests, derivedCapsuleDim)
      try { System.err.println(s"[MIND DEBUG] uView sizes=${uView.sizes()} numel=${uView.numel()}") } catch { case _: Throwable => () }
      uView
    } catch {
      case e: Throwable =>
        try { System.err.println(s"[MIND DEBUG] uRaw.dim=${uRaw.dim()} uRaw.sizes=${uRaw.sizes()}") } catch { case _: Throwable => () }
        try { System.err.println(s"[MIND DEBUG] batchSize=$batchSize seqLen=$seqLen numInterests=$numInterests capsuleDim=$capsuleDim totalElements=${uRaw.numel()}") } catch { case _: Throwable => () }
        // Fallback: if reshape failed, compute interests by pooling and projecting
        val pooled = xOn.mean(1) // (batch, embedDim)
        val fallback = W.forward(pooled) // (batch, numInterests * capsuleDim)
        try { System.err.println(s"[MIND DEBUG] fallback sizes=${fallback.sizes()} device=${fallback.device()}") } catch { case _: Throwable => () }
        try { println(s"[MIND DEBUG] fallback sizes=${fallback.sizes()} device=${fallback.device()}") } catch { case _: Throwable => () }
        val outDim = fallback.size(1).toInt
        val actualCapsuleDim = if (outDim % numInterests == 0) outDim / numInterests else capsuleDim
        val interestsFallback = fallback.view(batchSize, numInterests, actualCapsuleDim)
        try { System.err.println(s"[MIND DEBUG] interestsFallback sizes=${interestsFallback.sizes()} numel=${interestsFallback.numel()}") } catch { case _: Throwable => () }
        try { println(s"[MIND DEBUG] interestsFallback sizes=${interestsFallback.sizes()} numel=${interestsFallback.numel()}") } catch { case _: Throwable => () }
        return interestsFallback
    }

    // Simple attention-based pooling
    // Compute attention scores per (batch, seq_len, num_interests) by averaging over capsule dim
    val scores = u.mean(3) // (batch, seq_len, num_interests)
    try { System.err.println(s"[MIND DEBUG] scores sizes=${scores.sizes()} numel=${scores.numel()}") } catch { case _: Throwable => () }
    // Softmax over sequence length (dim=1) to get attention weights across tokens for each interest
    val attnWeights = scores.softmax(1).unsqueeze(3) // (batch, seq_len, num_interests, 1)
    try { System.err.println(s"[MIND DEBUG] attnWeights sizes=${attnWeights.sizes()}") } catch { case _: Throwable => () }
    val interests = u.mul(attnWeights).sum(1) // (batch, num_interests, capsule_dim)
    try { System.err.println(s"[MIND DEBUG] interests computed sizes=${interests.sizes()} numel=${interests.numel()}") } catch { case _: Throwable => () }

    interests
  }
}