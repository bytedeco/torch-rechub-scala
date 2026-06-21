package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * DeepFM: Combine FM (2nd-order) with DNN
 * Reference: IJCAI 2017
 */
class DeepFM(
  deepFeatures: List[Feature],
  fmFeatures: List[Feature],
  embedDim: Int = 8,
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  // All features combined for embedding lookup
  private val allFeatures = deepFeatures ++ fmFeatures
  private val embeddingLayer = new EmbeddingLayer(allFeatures, embedDim, device)
  register_module("embedding", embeddingLayer)

  // First-order: LR on FM features (flattened embeddings)
  private val fmDims = fmFeatures.map(_.embedDim).sum.toLong
  private val linear = new LinearImpl(fmDims, 1)
  register_module("linear", linear)

  // FM layer for 2nd-order interactions
  private val fm = new FM(embedDim, device)
  register_module("fm", fm)

  // Deep part
  private val deepDims = deepFeatures.map(_.embedDim).sum.toLong
  private val mlp = new MLP(deepDims, mlpDims.map(_.toLong), 1, "relu", dropout, false, device = device)
  register_module("mlp", mlp)

  // Move all submodules to device
  if (device != "cpu") {
    linear.to(new org.bytedeco.pytorch.Device(device), false)
    fm.to(new org.bytedeco.pytorch.Device(device), false)
    mlp.to(new org.bytedeco.pytorch.Device(device), false)
    embeddingLayer.to(device)
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Fetch embeddings separately for deep and FM feature sets to avoid
    // ordering/duplication issues when deepFeatures is a subset of fmFeatures.
    // Build per-group sparse maps preserving the feature names requested.
    val deepSparse = deepFeatures.map(f => f.name -> sparseFeats(f.name)).toMap
    val fmSparse = fmFeatures.map(f => f.name -> sparseFeats(f.name)).toMap

    // Get 3D embeddings per group: (batch, num_fields_in_group, embed_dim)
    val deepEmbeddings = embeddingLayer.forward3D(deepSparse, sequenceFeats = Map.empty)
    val fmEmbeddings = embeddingLayer.forward3D(fmSparse, sequenceFeats = Map.empty)
    val batchSize = deepEmbeddings.size(0).toInt

    // First-order: linear on flattened FM embeddings
    val fmFlattened = fmEmbeddings.view(batchSize, -1)  // (batch, num_fm_features * embed_dim)
    val linearOut = linear.forward(fmFlattened)  // (batch, 1)

    // Second-order: FM interactions
    val fmOut = fm.forward(fmEmbeddings)  // (batch, 1)

    // Deep part: flatten deep embeddings and pass through MLP
    val deepFlattened = deepEmbeddings.view(batchSize, -1)  // (batch, num_deep_features * embed_dim)
    val mlpOut = mlp.forward(deepFlattened)  // (batch, 1)

    // Combine all parts
    val logits = linearOut.add(fmOut).add(mlpOut)
    logits
  }
}
