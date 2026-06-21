package torchrec.models.matching

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.basic.features._
import torchrec.basic.layers.{EmbeddingLayer, MLP, SENETLayer}
import torchrec.utils.DeviceSupport

import scala.math

/**
 * DSSMSENET - Deep Structured Semantic Model with SENET
 *
 * Full implementation matching the Python torch_rechub DSSM with SENET model.
 * Uses dual-tower architecture with Squeeze-and-Excitation networks for
 * adaptive feature weighting.
 *
 * Architecture:
 *   - User Tower: Embedding + SENET + MLP for user representation
 *   - Item Tower: Embedding + SENET + MLP for item representation
 *   - Scoring: Cosine similarity with sigmoid output
 *
 * @param userFeatures User features for user tower
 * @param itemFeatures Item features for item tower
 * @param userParams MLP parameters for user tower: dims, activation, dropout
 * @param itemParams MLP parameters for item tower: dims, activation, dropout
 * @param temperature Temperature factor for similarity score
 * @param device Device for computation
 */
class DSSMSENET(
  userFeatures: List[Feature],
  itemFeatures: List[Feature],
  userParams: Map[String, Any] = Map(),
  itemParams: Map[String, Any] = Map(),
  temperature: Float = 1.0f,
  device: String = DeviceSupport.backend
) extends Module {

  require(userFeatures.nonEmpty, "userFeatures cannot be empty")
  require(itemFeatures.nonEmpty, "itemFeatures cannot be empty")

  private val targetDevice = new Device(device)

  // =========================================================================
  // Feature dimensions
  // =========================================================================
  private val userDims: Int = userFeatures.map(_.embedDim).sum
  private val itemDims: Int = itemFeatures.map(_.embedDim).sum

  // Count number of sparse/sequence features for SENET
  private val userNumFeatures: Int = userFeatures.count {
    case _: SparseFeature => true
    case f: SequenceFeature if f.sharedWith.isEmpty => true
    case _ => false
  }

  private val itemNumFeatures: Int = itemFeatures.count {
    case _: SparseFeature => true
    case f: SequenceFeature if f.sharedWith.isEmpty => true
    case _ => false
  }

  // =========================================================================
  // Separate embedding layers per tower (so each tower only embeds its own
  // features and the MLP input dim matches the tower feature dim).
  // =========================================================================
  private val userEmbeddingLayer = new EmbeddingLayer(
    userFeatures,
    userFeatures.head.embedDim,
    device
  )
  register_module("user_embedding", userEmbeddingLayer)

  private val itemEmbeddingLayer = new EmbeddingLayer(
    itemFeatures,
    itemFeatures.head.embedDim,
    device
  )
  register_module("item_embedding", itemEmbeddingLayer)

  // Feature name sets for routing inputs to the correct tower.
  private val userFeatureNames: Set[String] = userFeatures.map(_.name).toSet
  private val itemFeatureNames: Set[String] = itemFeatures.map(_.name).toSet

  // =========================================================================
  // SENET layers for adaptive feature weighting
  // =========================================================================
  private val userSenet = {
    val s = new SENETLayer(numFields = userNumFeatures, reduction = 3, device = device)
    s
  }
  register_module("user_senet", userSenet)

  private val itemSenet = {
    val s = new SENETLayer(numFields = itemNumFeatures, reduction = 3, device = device)
    s
  }
  register_module("item_senet", itemSenet)

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
  // User tower: encodes user features into user embedding with SENET
  // =========================================================================
  def userTower(x: Map[String, Tensor]): Tensor = {
    if (mode.contains("item")) {
      return torch.zeros(Array(1L, userFeatures.head.embedDim.toLong),
        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    }

    // [batch_size, num_features * embed_dim]
    val userFeats = x.view.filterKeys(userFeatureNames.contains).toMap
    val inputUser = userEmbeddingLayer.forward(
      sparseFeats = userFeats,
      sequenceFeats = Map(),
      squeeze = true
    )

    // [batch_size, num_features, embed_dim]
    // Reshape to (batch_size, num_features, embed_dim) for SENET
    val batchSize = inputUser.size(0)
    val reshapedUser = inputUser.view(batchSize, userNumFeatures, -1)

    // SENET: [batch_size, num_features, embed_dim]
    val senetedUser = userSenet.forward(reshapedUser)

    // [batch_size, num_features * embed_dim]
    val flattenedUser = senetedUser.view(batchSize, -1)

    // [batch_size, user_params["dims"][-1]]
    var userEmbedding = userMlp.forward(flattenedUser)

    // L2 normalize: F.normalize(user_embedding, p=2, dim=1)
    val normOpt = new NormalizeFuncOptions()
    normOpt.p().put(2)
    normOpt.dim().put(1)
    normOpt.eps().put(1e-8f)
    userEmbedding = torch.normalize(userEmbedding, normOpt)

    userEmbedding
  }

  // =========================================================================
  // Item tower: encodes item features into item embedding with SENET
  // =========================================================================
  def itemTower(x: Map[String, Tensor]): Tensor = {
    if (mode.contains("user")) {
      return torch.zeros(Array(1L, itemFeatures.head.embedDim.toLong),
        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    }

    // [batch_size, num_features * embed_dim]
    val itemFeats = x.view.filterKeys(itemFeatureNames.contains).toMap
    val inputItem = itemEmbeddingLayer.forward(
      sparseFeats = itemFeats,
      sequenceFeats = Map(),
      squeeze = true
    )

    // [batch_size, num_features, embed_dim]
    // Reshape to (batch_size, num_features, embed_dim) for SENET
    val batchSize = inputItem.size(0)
    val reshapedItem = inputItem.view(batchSize, itemNumFeatures, -1)

    // SENET: [batch_size, num_features, embed_dim]
    val senetedItem = itemSenet.forward(reshapedItem)

    // [batch_size, num_features * embed_dim]
    val flattenedItem = senetedItem.view(batchSize, -1)

    // [batch_size, item_params["dims"][-1]]
    var itemEmbedding = itemMlp.forward(flattenedItem)

    // L2 normalize: F.normalize(item_embedding, p=2, dim=1)
    val normOpt = new NormalizeFuncOptions()
    normOpt.p().put(2)
    normOpt.dim().put(1)
    normOpt.eps().put(1e-8f)
    itemEmbedding = torch.normalize(itemEmbedding, normOpt)

    itemEmbedding
  }

  // =========================================================================
  // Main forward pass.
  //
  // Returns the temperature-scaled cosine similarity as a *logit* (no sigmoid).
  // Callers that need a probability should apply sigmoid themselves; this keeps
  // the model compatible with BCEWithLogitsLoss and avoids a double-sigmoid.
  // =========================================================================
  def forward(x: Map[String, Tensor]): Tensor = {
    val userEmbedding = userTower(x)
    val itemEmbedding = itemTower(x)

    if (mode.contains("user")) return userEmbedding
    if (mode.contains("item")) return itemEmbedding

    // calculate cosine score: torch.mul(user_embedding, item_embedding).sum(dim=1)
    val cosSim = torch.mul(userEmbedding, itemEmbedding).sum(1L)

    // Apply temperature scaling -> logit
    cosSim.div(new Scalar(temperature))
  }
}

/**
 * DSSMSENET companion object with factory methods.
 */
object DSSMSENET {
  def apply(
    userFeatures: List[Feature],
    itemFeatures: List[Feature],
    userParams: Map[String, Any] = Map("dims" -> List(128L)),
    itemParams: Map[String, Any] = Map("dims" -> List(128L)),
    temperature: Float = 1.0f,
    device: String = DeviceSupport.backend
  ): DSSMSENET = {
    new DSSMSENET(userFeatures, itemFeatures, userParams, itemParams, temperature, device)
  }
}