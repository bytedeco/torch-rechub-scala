package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.nn.modules._
import org.bytedeco.pytorch.nn.modules.container._
import org.bytedeco.pytorch.nn.options._
import org.bytedeco.pytorch.optim._
import org.bytedeco.pytorch.data.datasets._
import org.bytedeco.pytorch.data.options._
import org.bytedeco.pytorch.data.sampler._
import org.bytedeco.pytorch.distributed._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * AutoInt — Automatic Feature Interaction via Attentive Multi-Head Self-Attention.
 *
 * Mirrors the Python `torch-rechub` reference:
 *   - `interacting_layers = nn.ModuleList([InteractingLayer(...) for _ in range(num_layers)])`
 *   - `dense_embeddings = nn.ModuleDict()` keyed by dense feature name
 *
 * Reference: https://arxiv.org/abs/1810.11921 (CIKM'2019).
 */
class AutoInt(
  sparseFeatures: List[Feature],
  denseFeatures: List[Feature] = List.empty,
  numAttnHeads: Int = 2,
  numLayers: Int = 3,
  mlpDims: List[Long] = List(128L, 64L),
  dropout: Float = 0.0f,
  useMlp: Boolean = true,
  device: String = DeviceSupport.backend
) extends Module {

  /** Backward-compatible secondary constructor: pre-rewrite signature
   *  `new AutoInt(sparseFeatures, embedDim, numAttnHeads, numLayers, mlpDims,
   *   dropout, useMlp, device)`. `embedDim` is taken from the first feature. */
  def this(
    sparseFeatures: List[Feature],
    embedDim: Int,
    numAttnHeads: Int,
    numLayers: Int,
    mlpDims: List[Long],
    dropout: Float,
    useMlp: Boolean,
    device: String
  ) = this(sparseFeatures, List.empty, numAttnHeads, numLayers, mlpDims, dropout, useMlp, device)

  require(sparseFeatures.nonEmpty, "AutoInt: sparseFeatures cannot be empty")

  private val numSparse: Int = sparseFeatures.size
  private val numDense: Int = denseFeatures.size
  private val numFields: Int = numSparse + numDense
  private val embedDim: Int = sparseFeatures.head.embedDim
  private val dims: Int = numFields * embedDim

  // Sparse embedding table.
  private val sparseEmbedding = new EmbeddingLayer(sparseFeatures, embedDim, device)
  register_module("sparse_embedding", sparseEmbedding)

  // Dense embeddings — `nn.ModuleDict` in Python, `ModuleDictImpl` here.
  private val denseEmbeddings: ModuleDictImpl = new ModuleDictImpl()
  for (fea <- denseFeatures) {
    val proj = new LinearImpl(LinearOptions(1L, embedDim.toLong).bias(false))
    register_module(s"dense_${fea.name}", proj)
    denseEmbeddings.insert(fea.name, proj)
  }

  // `nn.ModuleList` of interacting layers — direct mirror of the Python.
  private val interactingLayers: ModuleListImpl = new ModuleListImpl()
  for (i <- 0 until numLayers) {
    val layer = new InteractingLayer(embedDim, numAttnHeads, dropout, residual = true, device)
    register_module(s"interacting_$i", layer)
    interactingLayers.push_back(layer)
  }

  private val linear = new LR(dims.toLong, sigmoid = false, device)
  register_module("linear", linear)

  private val attnLinear = new LinearImpl(dims.toLong, 1L)
  register_module("attn_linear", attnLinear)

  private val mlp = if (useMlp) {
    val m = new MLP(dims.toLong, mlpDims.map(_.toLong), 1L, "relu", dropout, device = device)
    register_module("mlp", m)
    Some(m)
  } else None

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    val sparseEmb = sparseEmbedding.forward3D(sparseFeats) // (batch, numSparse, embedDim)

    // Project each dense feature through its own Linear(1, embedDim).
    val denseEmbList = scala.collection.mutable.ListBuffer[Tensor]()
    for (fea <- denseFeatures) {
      val v = denseFeats(fea.name).toType(ScalarType.Float).view(-1L, 1L, 1L)
      val proj = denseEmbeddings.get(fea.name).asInstanceOf[LinearImpl]
      denseEmbList += proj.forward(v)
    }

    val embedX = if (denseEmbList.nonEmpty) {
      val denseEmb = torch.cat(new TensorVector(denseEmbList.toSeq: _*), 1L)
      torch.cat(new TensorVector(sparseEmb, denseEmb), 1L)
    } else sparseEmb

    val embedXFlat = embedX.view(embedX.size(0), -1L) // (batch, dims)

    var attnOut: Tensor = embedX
    var i = 0
    while (i < numLayers) {
      attnOut = interactingLayers.get(i).forward(attnOut)
      i += 1
    }

    val attnOutFlat = attnOut.view(attnOut.size(0), -1L)
    val yAttn = attnLinear.forward(attnOutFlat)        // (batch, 1)
    val yLinear = linear.forward(embedXFlat)           // (batch, 1)

    var y = yAttn.add(yLinear)
    if (mlp.isDefined) y = y.add(mlp.get.forward(embedXFlat))
    y
  }
}