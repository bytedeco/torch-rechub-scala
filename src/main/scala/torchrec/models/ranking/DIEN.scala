package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits._
import torchrec.Implicits.SeqTensorRichSeq

import scala.collection.mutable

/**
 * Deep Interest Evolution Network (DIEN)
 * Reference: DIEN paper, AAAI 2019
 *
 * Simplified implementation for benchmark compatibility.
 * Uses Interest Extractor GRU and Interest Evolving AUGRU.
 */
class DIEN(
  features: List[Feature],
  sequenceFeatures: List[SequenceFeature],
  embedDim: Int = 8,
  hiddenDim: Int = 8,
  mlpDims: List[Long] = List(64L, 32L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")

  // Total dimension for MLP
  private val sparseDim: Long = Features.calcSparseDim(features)
  private val seqDim: Long = sequenceFeatures.map(_.embedDim).sum
  private val totalDim: Long = sparseDim + seqDim

  // Embedding layer
  private val embedding = new EmbeddingLayer(
    features ++ sequenceFeatures,
    embedDim,
    device
  )
  register_module("embedding", embedding)

  // Interest Extractor GRUs - one per sequence feature
  private val interestGRUs = sequenceFeatures.map { fea =>
    val opts = new GRUOptions(fea.embedDim, fea.embedDim)
    opts.batch_first().put(true)
    val gru = new GRUImpl(opts)
    if (device != "cpu") {
      gru.to(new org.bytedeco.pytorch.Device(device), false)
    }
    gru
  }
  interestGRUs.zipWithIndex.foreach { case (gru, i) =>
    register_module(s"interest_gru_$i", gru)
  }

  // Interest Evolving AUGRUs - one per sequence feature
  private val augrus = sequenceFeatures.map { fea =>
    new AUGRU(fea.embedDim, device)
  }
  augrus.zipWithIndex.foreach { case (augru, i) =>
    register_module(s"augru_$i", augru)
  }

  // MLP for final prediction with dice activation
  private val mlp = new MLP(
    inputDim = totalDim,
    hiddenDims = mlpDims,
    outputDim = 1,
    activation = "dice",
    dropout = dropout,
    device = device
  )
  register_module("mlp", mlp)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    this.to(dev, false)
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    seqFeats: Map[String, Tensor]
  ): Tensor = {
    // Feature embeddings
    val featEmb = embedding.forward(sparseFeats, squeeze = true)

    // Sequence embeddings - use forwardSeqRaw to get 3D embeddings without pooling for GRU
    val seqEmb = embedding.forwardSeqRaw(seqFeats)

    // Interest Extractor + Evolving
    val interestOut = mutable.ListBuffer[Tensor]()

    // sequenceFeatures embeddings are concatenated along the last dim in forwardSeqRaw
    val feaDims = sequenceFeatures.map(_.embedDim)
    var offset = 0
    for (i <- sequenceFeatures.indices) {
      val dim = feaDims(i)
      // slice out this feature's embedding from the last dim: seqEmb is (B, T, totalDim)
      val seq = seqEmb.narrow(2, offset.toLong, dim.toLong)
      offset += dim
      val mask = getMask(seqFeats, sequenceFeatures(i))  // (B, T)

      // Interest Extractor: GRU
      val gruOut = interestGRUs(i).forward(seq).get1()  // (B, T, D)

      // Use last valid hidden state as target - simplified without gather
      val seqLen = seq.size(1)
      // Take last time step directly: (B, D)
      val targetEmb = if (gruOut.dim() == 3L) gruOut.select(1, seqLen - 1) else gruOut

      // Interest Evolving: AUGRU
      // Avoid batch index_select/index_copy, which can fail on synthetic batches
      // when masks/indices are shaped unexpectedly. The AUGRU layer already
      // handles masking internally.
      val (evolvingSeq, _) = augrus(i).forward(seq, targetEmb, mask)
      val h = if (evolvingSeq.dim() == 3L) evolvingSeq.select(1, evolvingSeq.size(1) - 1) else evolvingSeq

      interestOut += h
    }

    // Concatenate all interest outputs
    val interestCat = interestOut.toSeq.cat(1)

    // Combine with features
    val combined = torch.cat(new TensorVector(interestCat, featEmb), 1)

    // Final MLP
    val y = mlp.forward(combined)

    torch.sigmoid(y.squeeze(1))
  }

  private def getMask(seqFeats: Map[String, Tensor], feature: SequenceFeature): Tensor = {
    val indices = seqFeats(feature.name)
    val paddingIdx = feature.paddingIdx
    if (paddingIdx >= 0) {
      indices.ne(new Scalar(paddingIdx)).toType(ScalarType.Bool)
    } else {
      indices.ne(new Scalar(-1L)).toType(ScalarType.Bool)
    }
  }
}