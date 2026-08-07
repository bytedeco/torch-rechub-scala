package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._
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
 * Wide & Deep Learning
 * Reference: Google
 */
class WideDeep(
  features: List[Feature],
  embedDim: Int = 8,
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  // Deep part
  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  private val sparseDim = Features.calcSparseDim(features)
  private val mlp = new MLP(sparseDim, mlpDims.map(_.toLong), 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    val embeddings = embeddingLayer.forward(sparseFeats)
    val dev = embeddings.device()

    // Wide: sum of embeddings per field (FM first-order style)
    val sparseNames = features.collect { case f: SparseFeature => f.name }
    val wideList = sparseNames.flatMap { name =>
      sparseFeats.get(name).map { idx =>
        val idxOnDev = if (idx.device().equals(dev)) {
          idx.toType(ScalarType.Long)
        } else {
          idx.toType(ScalarType.Long).to(dev, ScalarType.Long)
        }
        val emb = embeddingLayer.getEmbedding(name, idxOnDev)
        emb.sum(1).unsqueeze(1)
      }
    }
    val wideOut = if (wideList.nonEmpty) {
      wideList.reduceOption((a, b) => a.add(b)).getOrElse {
        torch.zeros(Array(embeddings.size(0).toLong, 1L), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(dev, ScalarType.Float)
      }
    } else {
      torch.zeros(Array(embeddings.size(0).toLong, 1L), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))).to(dev, ScalarType.Float)
    }

    // Deep
    val deepOut = mlp.forward(embeddings)

    // Combine (raw logits — BCEWithLogitsLoss handles sigmoid internally)
    val logits = wideOut.add(deepOut)
    logits
  }
}
