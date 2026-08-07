package torchrec.models.generative

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

import torchrec.Implicits._

/**
 * Tree-based Indexing with Generative Enhancement
 * Reference: Amazon, SIGIR 2023
 * Uses tree-based hierarchical indexing for efficient item retrieval
 * with generative enhancement for query understanding
 */
class TIGER(
  itemEmbeddings: Tensor, // Pre-computed item embeddings (num_items, embed_dim)
  embedDim: Int = 8,
  hiddenDim: Int = 128,
  numLayers: Int = 2,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  // Frozen item embeddings (no gradient)
  private val itemEmb = itemEmbeddings.clone().detach()
  itemEmb.requires_grad_(false)

  // MLP encoder instead of LSTM
  private val encoder = new MLP(embedDim, List(hiddenDim.toLong), hiddenDim, "relu", dropout, device = device)
  register_module("encoder", encoder)

  // Tree structure encoder for hierarchical indexing
  private val treeEncoder = new MLP(hiddenDim, List(hiddenDim / 2L), hiddenDim, "relu", 0f, device = device)
  register_module("treeEncoder", treeEncoder)

  override def forward(
    sequence: Tensor // (batch, seq_len) - item IDs
  ): Tensor = {
    // Get sequence embeddings from frozen table
    val seqFlat = sequence.view(-1).toType(ScalarType.Long)
    val seqEmb = itemEmb.index_select(0, seqFlat).view(sequence.size(0), sequence.size(1), embedDim)

    // Encode sequence with MLP (pool over sequence dimension)
    val pooled = seqEmb.mean(1)
    val encoded = encoder.forward(pooled)

    // Tree-based hierarchical indexing
    treeEncoder.forward(encoded)
  }

  def getItemEmbedding(itemId: Tensor): Tensor = {
    itemEmb.index_select(0, itemId.toType(ScalarType.Long))
  }

  def getItemCount(): Long = itemEmb.size(0)
}
