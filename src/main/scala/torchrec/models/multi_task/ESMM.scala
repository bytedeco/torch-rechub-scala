package torchrec.models.multi_task

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.basic.features._
import torchrec.basic.layers.{EmbeddingLayer, MLP}
import torchrec.utils.DeviceSupport

/**
 * Entire Space Multi-Task Model (ESMM)
 *
 * Full implementation matching the Python torch_rechub ESMM model.
 * Uses separate embedding layers for user and item features with CVR and CTR towers.
 *
 * Reference:
 *   "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate"
 *   SIGIR'2018 - https://arxiv.org/abs/1804.07931
 *
 * Architecture:
 *   - User Embedding Layer + Item Embedding Layer -> Concatenated
 *   - CVR Tower: sigmoid output for conversion prediction
 *   - CTR Tower: sigmoid output for click prediction
 *   - CTCVR = CTR * CVR (post-click conversion rate)
 *   - Returns [cvr_pred, ctr_pred, ctcvr_pred] concatenated
 *
 * @param userFeatures List of user features
 * @param itemFeatures List of item features
 * @param cvrParams CVR tower params: dims, activation, dropout
 * @param ctrParams CTR tower params: dims, activation, dropout
 * @param device Device for computation
 */
class ESMM(
  userFeatures: List[Feature],
  itemFeatures: List[Feature],
  cvrParams: Map[String, Any] = Map(),
  ctrParams: Map[String, Any] = Map(),
  device: String = DeviceSupport.backend
) extends Module {

  require(userFeatures.nonEmpty || itemFeatures.nonEmpty, "userFeatures or itemFeatures cannot be empty")

  private val targetDevice = new Device(device)

  // Embedding dimensions
  private val userEmbedDim = userFeatures.map(_.embedDim).sum
  private val itemEmbedDim = itemFeatures.map(_.embedDim).sum
  private val towerInputDim = userEmbedDim + itemEmbedDim

  // Separate embedding layers for user and item
  private val userEmbedding = new EmbeddingLayer(userFeatures, userFeatures.head.embedDim, device)
  private val itemEmbedding = new EmbeddingLayer(itemFeatures, itemFeatures.head.embedDim, device)
  register_module("userEmbedding", userEmbedding)
  register_module("itemEmbedding", itemEmbedding)

  // CVR tower
  private val cvrTowerDims = cvrParams.getOrElse("dims", List(128L, 64L)).asInstanceOf[List[Long]]
  private val cvrActivation = cvrParams.getOrElse("activation", "relu").asInstanceOf[String]
  private val cvrDropout = cvrParams.getOrElse("dropout", 0.0f).asInstanceOf[Float]
  private val towerCvr: MLP = new MLP(towerInputDim, cvrTowerDims, 1, cvrActivation, cvrDropout, outputLayer = true, device = device)
  register_module("tower_cvr", towerCvr)

  // CTR tower
  private val ctrTowerDims = ctrParams.getOrElse("dims", List(128L, 64L)).asInstanceOf[List[Long]]
  private val ctrActivation = ctrParams.getOrElse("activation", "relu").asInstanceOf[String]
  private val ctrDropout = ctrParams.getOrElse("dropout", 0.0f).asInstanceOf[Float]
  private val towerCtr: MLP = new MLP(towerInputDim, ctrTowerDims, 1, ctrActivation, ctrDropout, outputLayer = true, device = device)
  register_module("tower_ctr", towerCtr)

  def forward(x: Map[String, Tensor]): Tensor = {
    // Get embeddings for user features
    val userEmbed = userEmbedding.forward(x, Map.empty, squeeze = true)  // [batch, userEmbedDim]

    // Get embeddings for item features
    val itemEmbed = itemEmbedding.forward(x, Map.empty, squeeze = true)  // [batch, itemEmbedDim]

    // Concatenate user and item embeddings: [batch, userEmbedDim + itemEmbedDim]
    val inputTower = torch.cat(new TensorVector(userEmbed, itemEmbed), 1)

    // CVR and CTR towers
    val cvrLogit = towerCvr.forward(inputTower)
    val ctrLogit = towerCtr.forward(inputTower)

    // Predictions with sigmoid
    val cvrPred = torch.sigmoid(cvrLogit)
    val ctrPred = torch.sigmoid(ctrLogit)

    // CTCVR = CTR * CVR (post-click conversion rate)
    val ctcvrPred = torch.mul(ctrPred, cvrPred)

    // Return [cvr_pred, ctr_pred, ctcvr_pred] concatenated along dim=1
    torch.cat(new TensorVector(cvrPred, ctrPred, ctcvrPred), 1)
  }
}

/**
 * ESMM companion object with factory methods.
 */
object ESMM {
  // Backward compatible constructor with single features list
  def apply(
    features: List[Feature],
    taskNames: List[String] = List("cvr", "ctr"),
    embedDim: Int = 8,
    towerDims: List[Long] = List(128L, 64L),
    dropout: Float = 0.2f,
    device: String = DeviceSupport.backend
  ): ESMM = {
    // Split features into user and item (first half and second half)
    val numFeatures = features.size
    val half = numFeatures / 2
    val userFeatures = features.take(half)
    val itemFeatures = features.drop(half)

    val cvrParams = Map[String, Any](
      "dims" -> towerDims,
      "activation" -> "relu",
      "dropout" -> dropout
    )
    val ctrParams = Map[String, Any](
      "dims" -> towerDims,
      "activation" -> "relu",
      "dropout" -> dropout
    )

    new ESMM(userFeatures, itemFeatures, cvrParams, ctrParams, device)
  }
}