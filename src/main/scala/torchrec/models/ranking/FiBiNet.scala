package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._
import torchrec.TensorImplicits.RichTensor
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * FiBiNet: Feature Importance and Bilinear feature Interaction
 * Reference: RecSys 2019
 */
class FiBiNet(
  features: List[Feature],
  embedDim: Int = 8,
  mlpDims: List[Long] = List(256L, 128L),
  reduction: Int = 3,
  bilinearType: String = "field_all",  // "field_all", "field_each", "field_interaction"
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  private val senet = new SENETLayer(reduction, device)
  register_module("senet", senet)

  private val numFields = features.collect { case f: SparseFeature => 1 }.size

  // Bilinear interaction
  private val bilinear = bilinearType match {
    case "field_all" => new BilinearInteraction(embedDim, numFields, "field_all", device)
    case "field_each" => new BilinearInteraction(embedDim, numFields, "field_each", device)
    case _ => new BilinearInteraction(embedDim, numFields, "field_interaction", device)
  }
  register_module("bilinear", bilinear)

  // MLP
  private val mlp = new MLP(embedDim * numFields, mlpDims.map(_.toLong), 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    val embeddings = embeddingLayer.forward(sparseFeats)
    val batchSize = embeddings.size(0)
    val reshaped = embeddings.view(batchSize, numFields, embedDim)

    // SENET-enhanced features
    val senetFeatures = senet.forward(reshaped)

    // Bilinear interactions
    val biOut = bilinear.forward(senetFeatures, reshaped)

    // Concat bilinear and original, then MLP
    val tensorList = biOut.toList
    val combined = torch.cat(new TensorVector(tensorList.map(_.contiguous()): _*), 1L)
    val logits = mlp.forward(combined)
    logits
  }
}

/**
 * Bilinear Feature Interaction
 */
class BilinearInteraction(
  embedDim: Int,
  numFields: Int,
  bilinearType: String,
  device: String = DeviceSupport.backend
) extends Module {

  private val weight = bilinearType match {
    case "field_all" =>
      val w = new LinearImpl(embedDim, embedDim)
      register_module("bilinear_weight", w)
      w
    case _ =>
      val w = Array.fill(numFields)(new LinearImpl(embedDim, embedDim))
      w.zipWithIndex.foreach { case (lin, i) =>
        register_module(s"bilinear_weight_$i", lin)
      }
      w
  }

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    bilinearType match {
      case "field_all" =>
        weight.asInstanceOf[LinearImpl].to(dev, false)
      case _ =>
        weight.asInstanceOf[Array[LinearImpl]].foreach(_.to(dev, false))
    }
  }

  def forward(f1: Tensor, f2: Tensor): Seq[Tensor] = {
    // f1, f2: (batch, num_fields, embed_dim)
    bilinearType match {
      case "field_all" =>
        val w = weight.asInstanceOf[LinearImpl]
        (0 until numFields).flatMap { i =>
          (0 until numFields).map { j =>
            val vi = f1.select(1, i)
            val vj = w.forward(f2.select(1, j))
            vi.mul(vj).sum(1).unsqueeze(1)
          }
        }

      case _ =>
        val w = weight.asInstanceOf[Array[LinearImpl]]
        (0 until numFields).flatMap { i =>
          (0 until numFields).map { j =>
            val vi = f1.select(1, i)
            val vj = w(i).forward(f2.select(1, j))
            vi.mul(vj).sum(1).unsqueeze(1)
          }
        }
    }
  }
}