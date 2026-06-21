package torchrec.models.matching

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.basic.features._
import torchrec.basic.layers.{EmbeddingLayer, MLP}
import torchrec.utils.DeviceSupport

import scala.math

/**
 * GRU4Rec - GRU-based Session-Based Recommender
 *
 * Full implementation matching the Python torch_rechub GRU4Rec model.
 * Uses dual-tower architecture with GRU for session encoding.
 *
 * Reference:
 *   "Session-Based Recommendations with Recurrent Neural Networks"
 *   Hidasi et al., 2015 - http://arxiv.org/abs/1511.06939
 *
 * Architecture:
 *   - Item Embedding: Lookup table for item IDs
 *   - User Tower: GRU over history sequence + MLP for user representation
 *   - Item Tower: Item embedding lookup with L2 normalization
 *   - Scoring: Dot product (cosine similarity) between user and item embeddings
 *
 * @param userFeatures User context features (sparse)
 * @param historyFeatures Sequence features for history (e.g., click history)
 * @param itemFeatures Item features (sparse)
 * @param negItemFeature Negative item features for contrastive learning
 * @param userParams MLP parameters for user tower: dims, activation, dropout, num_layers
 * @param temperature Temperature factor for similarity score
 * @param device Device for computation
 */
class GRU4Rec(
  userFeatures: List[Feature],
  historyFeatures: List[Feature],
  itemFeatures: List[Feature],
  negItemFeature: Option[Feature] = None,
  userParams: Map[String, Any] = Map(),
  temperature: Float = 1.0f,
  device: String = DeviceSupport.backend
) extends Module {

  require(userFeatures.nonEmpty, "userFeatures cannot be empty")
  require(historyFeatures.nonEmpty, "historyFeatures cannot be empty")
  require(itemFeatures.nonEmpty, "itemFeatures cannot be empty")

  private val targetDevice = new Device(device)

  // =========================================================================
  // Feature dimensions
  // =========================================================================
  private val userDims: Int = (userFeatures ++ historyFeatures).map(_.embedDim).sum
  private val historyDim: Int = historyFeatures.head.embedDim
  // Keep a list of history sequence feature names for runtime checks
  private val historyFeatureNames: List[String] = historyFeatures.map(_.name)
  private val userFeatureNames: Set[String] = userFeatures.map(_.name).toSet
  private val itemFeatureNames: Set[String] = itemFeatures.map(_.name).toSet
  private val numLayers: Int = userParams.getOrElse("num_layers", 2).asInstanceOf[Int]

  // =========================================================================
  // Embedding layer for all features
  // =========================================================================
  private val embedding = new EmbeddingLayer(
    userFeatures ++ itemFeatures ++ historyFeatures ++ negItemFeature.toList,
    historyDim,
    device
  )
  register_module("embedding", embedding)

  // =========================================================================
  // GRU for history sequence encoding using GRUImpl
  // PyTorch equivalent: nn.GRU(input_size=history_dim, hidden_size=history_dim,
  //                           num_layers=num_layers, batch_first=True, bias=False)
  // =========================================================================
  private val gru: GRUImpl = {
    val opts = new GRUOptions(historyDim, historyDim)
    opts.num_layers().put(numLayers)
    opts.batch_first().put(true)
    opts.bias().put(false)
    val g = new GRUImpl(opts)
    g.to(targetDevice, false)
    g
  }
  register_module("gru", gru)

  // =========================================================================
  // User MLP tower
  // =========================================================================
  private val userMlp = {
    val dims = userParams.getOrElse("dims", List(historyDim * 2.toLong)).asInstanceOf[List[Long]]
    val activation = userParams.getOrElse("activation", "relu").asInstanceOf[String]
    val dropout = userParams.getOrElse("dropout", 0.0f).asInstanceOf[Float]
    new MLP(userDims, dims, historyDim, activation, dropout, outputLayer = true, device = device)
  }
  register_module("user_mlp", userMlp)

  // =========================================================================
  // Mode for inference (user_tower or item_tower)
  // =========================================================================
  private var mode: Option[String] = None

  // =========================================================================
  // Initialize weights
  // =========================================================================
  private def _initWeights(): Unit = {
    // GRU weights are initialized by GRUImpl constructor
    // MLP weights initialized by MLP constructor
  }

  // =========================================================================
  // Set inference mode
  // =========================================================================
  def setMode(m: String): Unit = {
    mode = Some(m)
  }

  // =========================================================================
  // User tower: encodes user features + history into user embedding
  // =========================================================================
  /**
   * User tower expecting a combined map. For backward compatibility we
   * split the incoming map into sparse features and sequence features
   * based on known history feature names and delegate to the two-arg
   * implementation.
   */
  def userTower(x: Map[String, Tensor]): Tensor = {
    // split x into sparse and sequence maps
    val (seqMap, sparseMap) = x.partition { case (k, _) => historyFeatureNames.contains(k) }
    // seqMap contains sequence features, sparseMap contains others
    userTower(sparseMap, seqMap)
  }

  /**
   * New userTower that accepts sparse features and sequence features separately.
   * This avoids passing sequence tensors as sparse features which previously
   * caused embedding table lookup mistakes.
   */
  def userTower(sparseFeats: Map[String, Tensor], sequenceFeats: Map[String, Tensor]): Tensor = {
    if (mode.contains("item")) {
      val zero = torch.zeros(Array(1L, historyDim.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      return zero
    }

    // Filter incoming maps to only the features this model was constructed with.
    val filteredSparse = sparseFeats.filter { case (k, _) => userFeatureNames.contains(k) }
    // Get user feature embeddings: [batch_size, user_dims]
    val userEmb = embedding.forward(
      sparseFeats = filteredSparse,
      sequenceFeats = Map(),
      squeeze = true
    )

    // Get history sequence embeddings raw (no pooling) so we obtain [batch, seq_len, history_dim]
    val seqFiltered = sequenceFeats.filter { case (k, _) => historyFeatureNames.contains(k) }
    val rawSeq = embedding.forwardSeqRaw(seqFiltered)
    val historyEmb = if (rawSeq.dim() == 4L) {
      // rawSeq: (batch, num_features, seqLen, embedDim) -> expect single seq feature -> squeeze field dim
      rawSeq.squeeze(1L)
    } else {
      // already (batch, seqLen, embedDim)
      rawSeq
    }

    // Pass through GRU: take only the last hidden state
    val gruOutput = gru.forward(historyEmb)
    val gruHidden = gruOutput.get1 // hidden state tensor
    val lastHidden = gruHidden.select(0, numLayers - 1) // hidden[-1]

    // Concatenate user features with last hidden state: [batch_size, user_dims + history_dim]
    val combined = torch.cat(new TensorVector(userEmb, lastHidden), 1)

    // Pass through MLP user tower: [batch_size, 1, history_dim]
    var userEmbedding = userMlp.forward(combined).unsqueeze(1)

    // L2 normalize: F.normalize(user_embedding, p=2, dim=-1)
    val normOpt = new NormalizeFuncOptions()
    normOpt.p().put(2)
    normOpt.dim().put(-1)
    normOpt.eps().put(1e-8f)
    userEmbedding = torch.normalize(userEmbedding, normOpt)

    if (mode.contains("user")) {
      // Inference embedding mode -> [batch_size, history_dim]
      userEmbedding.squeeze(1)
    } else {
      userEmbedding
    }
  }

  // =========================================================================
  // Item tower: encodes items into embeddings
  // =========================================================================
  def itemTower(x: Map[String, Tensor]): Tensor = {
    if (mode.contains("user")) {
      val zero = torch.zeros(Array(1L, historyDim.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      return zero
    }

    // Get positive item embeddings: [batch_size, 1, history_dim]
    val posEmbedding = embedding.forward(
      sparseFeats = x,
      sequenceFeats = Map(),
      squeeze = false
    )

    // L2 normalize
    val normOpt = new NormalizeFuncOptions()
    normOpt.p().put(2)
    normOpt.dim().put(-1)
    normOpt.eps().put(1e-8f)
    var posEmbNormed = torch.normalize(posEmbedding, normOpt)

    if (mode.contains("item")) {
      // Inference embedding mode -> [batch_size, history_dim]
      return posEmbNormed.squeeze(1)
    }

    // Get negative item embeddings if provided
    val negEmbeddingOpt = negItemFeature.flatMap { negFeat =>
      val negX = Map(negFeat.name -> x.getOrElse(negFeat.name, throw new IllegalArgumentException(s"Missing feature: ${negFeat.name}")))
      Some(embedding.forward(sparseFeats = negX, sequenceFeats = Map(), squeeze = false))
    }

    val finalItemEmbedding = negEmbeddingOpt match {
      case Some(negEmb) =>
        // L2 normalize negative embeddings
        var negEmbNormed = torch.normalize(negEmb, normOpt)
        // Concatenate: [batch_size, 1 + n_neg_items, history_dim]
        torch.cat(new TensorVector(posEmbNormed, negEmbNormed), 1)
      case None =>
        posEmbNormed
    }

    finalItemEmbedding
  }

  // =========================================================================
  // Main forward pass
  // =========================================================================
  def forward(x: Map[String, Tensor]): Tensor = {
    val userEmb = userTower(x)
    val itemEmb = itemTower(x)

    if (mode.contains("user")) return userEmb
    if (mode.contains("item")) return itemEmb

    // Compute dot product: torch.mul(user_emb, item_emb).sum(dim=1)
    // Returns similarity scores
    val scores = torch.mul(userEmb, itemEmb).sum(1L)

    scores
  }
}

/**
 * GRU4Rec companion object with factory methods.
 */
object GRU4Rec {
  def apply(
    userFeatures: List[Feature],
    historyFeatures: List[Feature],
    itemFeatures: List[Feature],
    negItemFeature: Option[Feature] = None,
    userParams: Map[String, Any] = Map("num_layers" -> 2),
    temperature: Float = 1.0f,
    device: String = DeviceSupport.backend
  ): GRU4Rec = {
    new GRU4Rec(userFeatures, historyFeatures, itemFeatures, negItemFeature, userParams, temperature, device)
  }
}