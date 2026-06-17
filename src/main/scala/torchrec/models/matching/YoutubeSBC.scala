package torchrec.models.matching

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.basic.features._
import torchrec.basic.layers.{EmbeddingLayer, MLP}
import torchrec.utils.DeviceSupport

import scala.collection.mutable
import scala.math

/**
 * YoutubeSBC - Sampling-Bias-Corrected Neural Modeling for Matching
 *
 * Full implementation matching the Python torch_rechub YoutubeSBC model.
 * Uses dual-tower architecture with in-batch softmax sampling and bias correction.
 *
 * Reference:
 *   "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"
 *   RecSys'2019 - https://dl.acm.org/doi/10.1145/3298689.3346996
 *
 * Architecture:
 *   - User Tower: Embedding + MLP for user representation
 *   - Item Tower: Embedding + MLP for item representation
 *   - Scoring: Cosine similarity with sampling bias correction
 *   - In-batch softmax loss with temperature
 *
 * @param userFeatures User features for user tower
 * @param itemFeatures Item features for item tower
 * @param sampleWeightFeature Sample weight feature for bias correction
 * @param userParams MLP parameters for user tower: dims, activation, dropout
 * @param itemParams MLP parameters for item tower: dims, activation, dropout
 * @param batchSize Batch size for in-batch sampling indices
 * @param nNeg Number of negative samples per positive (default 3)
 * @param temperature Temperature factor for similarity score
 * @param device Device for computation
 */
class YoutubeSBC(
  userFeatures: List[Feature],
  itemFeatures: List[Feature],
  sampleWeightFeature: Option[Feature] = None,
  userParams: Map[String, Any] = Map(),
  itemParams: Map[String, Any] = Map(),
  batchSize: Int = 128,
  nNeg: Int = 3,
  temperature: Float = 1.0f,
  device: String = DeviceSupport.backend
) extends Module {

  require(userFeatures.nonEmpty, "userFeatures cannot be empty")
  require(itemFeatures.nonEmpty, "itemFeatures cannot be empty")
  require(nNeg > 0, "nNeg must be positive")
  require(batchSize > nNeg, "batchSize must be greater than nNeg")

  private val targetDevice = new Device(device)

  // =========================================================================
  // Feature dimensions
  // =========================================================================
  private val userDims: Int = userFeatures.map(_.embedDim).sum
  private val itemDims: Int = itemFeatures.map(_.embedDim).sum

  // =========================================================================
  // Embedding layer for all features
  // =========================================================================
  private val embedding = new EmbeddingLayer(
    userFeatures ++ itemFeatures ++ sampleWeightFeature.toList,
    userFeatures.head.embedDim,
    device
  )
  register_module("embedding", embedding)

  // =========================================================================
  // User MLP tower
  // =========================================================================
  private val userMlp = {
    val dims = userParams.getOrElse("dims", List(128L)).asInstanceOf[List[Long]]
    val activation = userParams.getOrElse("activation", "relu").asInstanceOf[String]
    val dropout = userParams.getOrElse("dropout", 0.0f).asInstanceOf[Float]
    new MLP(userDims, dims, userFeatures.head.embedDim, activation, dropout, outputLayer = false, device = device)
  }
  register_module("user_mlp", userMlp)

  // =========================================================================
  // Item MLP tower
  // =========================================================================
  private val itemMlp = {
    val dims = itemParams.getOrElse("dims", List(128L)).asInstanceOf[List[Long]]
    val activation = itemParams.getOrElse("activation", "relu").asInstanceOf[String]
    val dropout = itemParams.getOrElse("dropout", 0.0f).asInstanceOf[Float]
    new MLP(itemDims, dims, itemFeatures.head.embedDim, activation, dropout, outputLayer = false, device = device)
  }
  register_module("item_mlp", itemMlp)

  // =========================================================================
  // In-batch sampling indices
  // Python: self.index0 = np.repeat(np.arange(batch_size), n_neg + 1)
  //         self.index1 = np.concatenate([np.arange(i, i + n_neg + 1) for i in range(batch_size)])
  //         self.index1[np.where(self.index1 >= batch_size)] -= batch_size
  // =========================================================================
  private val index0: Array[Long] = {
    val arr = mutable.ArrayBuilder.make[Long]
    for (i <- 0 until batchSize) {
      for (_ <- 0 until nNeg + 1) {
        arr += i.toLong
      }
    }
    arr.result()
  }

  private val index1: Array[Long] = {
    val arr = mutable.ArrayBuilder.make[Long]
    for (i <- 0 until batchSize) {
      for (j <- 0 until nNeg + 1) {
        var idx = (i + j).toLong
        if (idx >= batchSize) idx -= batchSize
        arr += idx
      }
    }
    arr.result()
  }

  // =========================================================================
  // Mode for inference (user_tower or item_tower)
  // =========================================================================
  private var mode: Option[String] = None

  // =========================================================================
  // Set inference mode
  // =========================================================================
  def setMode(m: String): Unit = {
    mode = Some(m)
  }

  // =========================================================================
  // User tower: encodes user features into user embedding
  // =========================================================================
  def userTower(x: Map[String, Tensor]): Tensor = {
    if (mode.contains("item")) {
      return torch.zeros(Array(batchSize.toLong, userFeatures.head.embedDim.toLong),
        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    }

    // [batch_size, num_features*embed_dim]
    val inputUser = embedding.forward(
      sparseFeats = x,
      sequenceFeats = Map(),
      squeeze = true
    )

    // [batch_size, user_params["dims"][-1]]
    userMlp.forward(inputUser)
  }

  // =========================================================================
  // Item tower: encodes item features into item embedding
  // =========================================================================
  def itemTower(x: Map[String, Tensor]): Tensor = {
    if (mode.contains("user")) {
      return torch.zeros(Array(batchSize.toLong, itemFeatures.head.embedDim.toLong),
        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    }

    // [batch_size, num_features*embed_dim]
    val inputItem = embedding.forward(
      sparseFeats = x,
      sequenceFeats = Map(),
      squeeze = true
    )

    // [batch_size, item_params["dims"][-1]]
    itemMlp.forward(inputItem)
  }

  // =========================================================================
  // Main forward pass with sampling bias correction
  // =========================================================================
  def forward(x: Map[String, Tensor]): Tensor = {
    val userEmbedding = userTower(x)
    val itemEmbedding = itemTower(x)

    if (mode.contains("user")) return userEmbedding
    if (mode.contains("item")) return itemEmbedding

    // Compute cosine similarity: torch.cosine_similarity(user_emb.unsqueeze(1), item_emb, dim=2)
    // user_emb: [batch_size, 1, embed_dim], item_emb: [batch_size, embed_dim]
    val userEmbExpanded = userEmbedding.unsqueeze(1) // [batch_size, 1, embed_dim]
    val itemEmbExpanded = itemEmbedding.unsqueeze(0) // [1, batch_size, embed_dim] via broadcasting

    // Cosine similarity manually: (a . b) / (||a|| * ||b||)
    val userNormOpt = new NormalizeFuncOptions()
    userNormOpt.p().put(2)
    userNormOpt.dim().put(-1)
    userNormOpt.eps().put(1e-8f)
    val userNorm = torch.normalize(userEmbExpanded, userNormOpt)

    val itemNormOpt = new NormalizeFuncOptions()
    itemNormOpt.p().put(2)
    itemNormOpt.dim().put(-1)
    itemNormOpt.eps().put(1e-8f)
    val itemNorm = torch.normalize(itemEmbedding, itemNormOpt)
    val cosSim = torch.mul(userNorm, itemNorm.unsqueeze(1)).sum(2) // [batch_size, batch_size]

    // Get sample weight: self.embedding(x, self.sample_weight_feature, squeeze_dim=True).squeeze(1)
    val sampleWeight = sampleWeightFeature.map { feat =>
      val emb = embedding.forward(sparseFeats = Map(feat.name -> x(feat.name)), sequenceFeats = Map(), squeeze = true)
      emb.squeeze(1)
    }.getOrElse {
      torch.ones(Array(batchSize.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    }

    // Sampling Bias Corrected: scores = pred - torch.log(sample_weight)
    // Broadcasting sample_weight from [batch_size] to [batch_size, batch_size]
    val sampleWeightExpanded = sampleWeight.unsqueeze(1) // [batch_size, 1]
    val scores = cosSim.sub(torch.log(sampleWeightExpanded))

    // Apply in-batch sampling indices
    val currentBatchSize = userEmbedding.size(0).toInt
    val (idx0, idx1) = if (currentBatchSize * (nNeg + 1) != index0.length) {
      // Last batch - adjust indices
      val slicedIndex0 = index0.slice(0, currentBatchSize * (nNeg + 1))
      val slicedIndex1 = index1.slice(0, currentBatchSize * (nNeg + 1))
      // Adjust for batch size wrapping
      val adjustedIdx0 = slicedIndex0.map { idx =>
        if (idx >= currentBatchSize) idx - currentBatchSize else idx
      }
      val adjustedIdx1 = slicedIndex1.map { idx =>
        if (idx >= currentBatchSize) idx - currentBatchSize else idx
      }
      (adjustedIdx0, adjustedIdx1)
    } else {
      (index0, index1)
    }

    // Flatten scores and indices for lookup
    val flatScores = scores.view(-1) // [batch_size * batch_size]

    // Build index tensor for advanced indexing
    val idxTensor = torch.tensor(idx1, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
    val batchIdxTensor = torch.tensor(idx0, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))

    // Gather: scores[index0, index1] equivalent
    val gatheredScores = flatScores.index_select(0, batchIdxTensor).index_select(0, idxTensor)

    // Apply temperature: scores = scores / temperature
    val finalScores = gatheredScores.div(new Scalar(temperature))

    // Reshape: [batch_size, 1 + n_neg]
    finalScores.view(currentBatchSize, nNeg + 1)
  }
}

/**
 * YoutubeSBC companion object with factory methods.
 */
object YoutubeSBC {
  def apply(
    userFeatures: List[Feature],
    itemFeatures: List[Feature],
    sampleWeightFeature: Option[Feature] = None,
    userParams: Map[String, Any] = Map("dims" -> List(128L)),
    itemParams: Map[String, Any] = Map("dims" -> List(128L)),
    batchSize: Int = 128,
    nNeg: Int = 3,
    temperature: Float = 1.0f,
    device: String = DeviceSupport.backend
  ): YoutubeSBC = {
    new YoutubeSBC(userFeatures, itemFeatures, sampleWeightFeature, userParams, itemParams, batchSize, nNeg, temperature, device)
  }
}