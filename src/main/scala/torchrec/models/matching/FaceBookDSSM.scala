package torchrec.models.matching

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.basic.features._
import torchrec.basic.layers.{EmbeddingLayer, MLP}
import torchrec.utils.DeviceSupport

import scala.math

/**
 * FaceBookDSSM - Embedding-based Retrieval in Facebook Search
 *
 * Full implementation matching the Python torch_rechub FaceBookDSSM model.
 * Uses dual-tower architecture with positive/negative item scoring.
 *
 * Reference:
 *   "Embedding-based Retrieval in Facebook Search"
 *   KDD'2020 - https://arxiv.org/abs/2006.11632
 *
 * Architecture:
 *   - User Tower: Embedding + MLP for user representation
 *   - Item Tower: Embedding + MLP for item representation
 *   - Scoring: Cosine similarity between user and positive/negative items
 *   - Hinge loss training with pair-wise samples
 *
 * @param userFeatures User features for user tower
 * @param posItemFeatures Positive item features for item tower
 * @param negItemFeatures Negative item features for item tower
 * @param userParams MLP parameters for user tower: dims, activation, dropout
 * @param itemParams MLP parameters for item tower: dims, activation, dropout
 * @param temperature Temperature factor for similarity score
 * @param device Device for computation
 */
class FaceBookDSSM(
  userFeatures: List[Feature],
  posItemFeatures: List[Feature],
  negItemFeatures: List[Feature],
  userParams: Map[String, Any] = Map(),
  itemParams: Map[String, Any] = Map(),
  temperature: Float = 1.0f,
  device: String = DeviceSupport.backend
) extends Module {

  require(userFeatures.nonEmpty, "userFeatures cannot be empty")
  require(posItemFeatures.nonEmpty, "posItemFeatures cannot be empty")
  require(negItemFeatures.nonEmpty, "negItemFeatures cannot be empty")

  private val targetDevice = new Device(device)

  // =========================================================================
  // Feature dimensions
  // =========================================================================
   private val userDims: Int = userFeatures.map(_.embedDim).sum
   private val itemDims: Int = posItemFeatures.map(_.embedDim).sum

   // =========================================================================
   // Separate Embedding layers for each tower
   // =========================================================================
   private val userEmbedding = new EmbeddingLayer(
     userFeatures,
     userFeatures.head.embedDim,
     device
   )
   register_module("user_embedding", userEmbedding)

   private val posItemEmbedding = new EmbeddingLayer(
     posItemFeatures,
     posItemFeatures.head.embedDim,
     device
   )
   register_module("pos_item_embedding", posItemEmbedding)

   private val negItemEmbedding = new EmbeddingLayer(
     negItemFeatures,
     negItemFeatures.head.embedDim,
     device
   )
   register_module("neg_item_embedding", negItemEmbedding)

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
    new MLP(itemDims, dims, posItemFeatures.head.embedDim, activation, dropout, outputLayer = false, device = device)
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
   // User tower: encodes user features into user embedding
   // =========================================================================
   def userTower(x: Map[String, Tensor]): Tensor = {
     if (mode.contains("item")) {
       return torch.zeros(Array(1L, userFeatures.head.embedDim.toLong),
         new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
     }

     // [batch_size, num_features*embed_dim]
     val userFeats = x.view.filterKeys(k => userFeatures.map(_.name).contains(k)).toMap
     val inputUser = userEmbedding.forward(
       sparseFeats = userFeats,
       sequenceFeats = Map(),
       squeeze = true
     )

     // [batch_size, user_params["dims"][-1]]
     var userEmb = userMlp.forward(inputUser)

     // L2 normalize: F.normalize(user_embedding, p=2, dim=1)
     val normOpt = new NormalizeFuncOptions()
     normOpt.p().put(2)
     normOpt.dim().put(1)
     normOpt.eps().put(1e-8f)
     userEmb = torch.normalize(userEmb, normOpt)

     userEmb
   }

   // =========================================================================
   // Item tower: encodes item features into positive and negative embeddings
   // =========================================================================
   def itemTower(x: Map[String, Tensor]): (Tensor, Tensor) = {
     if (mode.contains("user")) {
       val zero = torch.zeros(Array(1L, posItemFeatures.head.embedDim.toLong),
         new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
       val zeroNeg = torch.zeros(Array(1L, posItemFeatures.head.embedDim.toLong),
         new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
       return (zero, zeroNeg)
     }

     // Extract positive and negative item features
     val posFeats = x.view.filterKeys(k => posItemFeatures.map(_.name).contains(k)).toMap
     val negFeats = x.view.filterKeys(k => negItemFeatures.map(_.name).contains(k)).toMap

     // [batch_size, num_features*embed_dim]
     val inputItemPos = if (posFeats.nonEmpty) {
       posItemEmbedding.forward(
         sparseFeats = posFeats,
         sequenceFeats = Map(),
         squeeze = true
       )
     } else {
       torch.zeros(Array(1L, posItemFeatures.head.embedDim.toLong),
         new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
     }

     if (mode.contains("item")) {
       // Inference embedding mode
       val posEmbedding = itemMlp.forward(inputItemPos)
       val zeroNeg = torch.zeros(Array(1L, posItemFeatures.head.embedDim.toLong),
         new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
       return (posEmbedding, zeroNeg)
     }

     // [batch_size, num_features*embed_dim]
     val inputItemNeg = if (negFeats.nonEmpty) {
       negItemEmbedding.forward(
         sparseFeats = negFeats,
         sequenceFeats = Map(),
         squeeze = true
       )
     } else {
       torch.zeros(Array(1L, negItemFeatures.head.embedDim.toLong),
         new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
     }

     // [batch_size, item_params["dims"][-1]]
     var posEmbedding = itemMlp.forward(inputItemPos)
     var negEmbedding = itemMlp.forward(inputItemNeg)

     // L2 normalize: F.normalize(embedding, p=2, dim=1)
     val normOpt = new NormalizeFuncOptions()
     normOpt.p().put(2)
     normOpt.dim().put(1)
     normOpt.eps().put(1e-8f)
     posEmbedding = torch.normalize(posEmbedding, normOpt)
     negEmbedding = torch.normalize(negEmbedding, normOpt)

     (posEmbedding, negEmbedding)
   }

  // =========================================================================
  // Main forward pass
  // =========================================================================
  def forward(x: Map[String, Tensor]): (Tensor, Tensor) = {
    val userEmbedding = userTower(x)
    val (posItemEmbedding, negItemEmbedding) = itemTower(x)

    if (mode.contains("user")) {
      val zeroNeg = torch.zeros(Array(1L, posItemFeatures.head.embedDim.toLong),
        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      return (userEmbedding, zeroNeg)
    }
    if (mode.contains("item")) {
      val zeroNeg = torch.zeros(Array(1L, posItemFeatures.head.embedDim.toLong),
        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      return (posItemEmbedding, zeroNeg)
    }

    // calculate cosine score
    // pos_score = torch.mul(user_embedding, pos_item_embedding).sum(dim=1)
    // neg_score = torch.mul(user_embedding, neg_item_embedding).sum(dim=1)
    val posScore = torch.mul(userEmbedding, posItemEmbedding).sum(1L)
    val negScore = torch.mul(userEmbedding, negItemEmbedding).sum(1L)

    (posScore, negScore)
  }
}

/**
 * FaceBookDSSM companion object with factory methods.
 */
object FaceBookDSSM {
  def apply(
    userFeatures: List[Feature],
    posItemFeatures: List[Feature],
    negItemFeatures: List[Feature],
    userParams: Map[String, Any] = Map("dims" -> List(128L)),
    itemParams: Map[String, Any] = Map("dims" -> List(128L)),
    temperature: Float = 1.0f,
    device: String = DeviceSupport.backend
  ): FaceBookDSSM = {
    new FaceBookDSSM(userFeatures, posItemFeatures, negItemFeatures, userParams, itemParams, temperature, device)
  }
}