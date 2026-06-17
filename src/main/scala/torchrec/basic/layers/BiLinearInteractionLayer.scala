package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import torchrec.utils.DeviceSupport
import torchrec.TensorImplicits.RichTensor

import scala.collection.mutable

/**
 * Bilinear feature interaction (FFM-style).
 *
 * Parameters
 * ----------
 * inputDim : int
 *   Input dimension.
 * numFields : int
 *   Number of feature fields.
 * bilinearType : {'field_all', 'field_each', 'field_interaction'}, default 'field_interaction'
 *   Bilinear interaction variant.
 *
 * Shape
 * -----
 * Input: ``(B, num_fields, embed_dim)``
 * Output: ``(B, num_interactions, embed_dim)``
 */
class BiLinearInteractionLayer(
  inputDim: Long,
  numFields: Int,
  bilinearType: String = "field_interaction",
  device: String = DeviceSupport.backend
) extends Module {

  private val bilinearLayers = bilinearType match {
    case "field_all" =>
      val layer = new LinearImpl(inputDim, inputDim)
      register_module("bilinear_layer", layer)
      List(layer)
    case "field_each" =>
      List.tabulate(numFields) { i =>
        val layer = new LinearImpl(inputDim, inputDim)
        register_module(s"bilinear_layer_$i", layer)
        layer
      }
    case "field_interaction" =>
      val numInteractions = (0 until numFields).flatMap { i =>
        (i + 1 until numFields).map { j => (i, j) }
      }.size
      List.tabulate(numInteractions) { idx =>
        val layer = new LinearImpl(inputDim, inputDim)
        register_module(s"bilinear_layer_$idx", layer)
        layer
      }
    case _ =>
      throw new NotImplementedError(s"bilinearType $bilinearType not implemented")
  }

  def forward(x: Tensor): Tensor = {
    val numFields = x.size(1).toInt
    // embeddings: list of (batch, embed_dim)
    val embeddings = (0 until numFields).map { i =>
      x.select(1, i)
    }.toSeq

    val interactions = mutable.ListBuffer[Tensor]()

    bilinearType match {
      case "field_all" =>
        for (i <- 0 until numFields; j <- i + 1 until numFields) {
          val vi = embeddings(i)
          val vj = embeddings(j)
          interactions += bilinearLayers(0).forward(vi) * vj
        }
      case "field_each" =>
        var idx = 0
        for (i <- 0 until numFields; j <- i + 1 until numFields) {
          val vi = embeddings(i)
          val vj = embeddings(j)
          interactions += bilinearLayers(idx).forward(vi) * vj
          idx += 1
        }
      case "field_interaction" =>
        var idx = 0
        for (i <- 0 until numFields; j <- i + 1 until numFields) {
          val vi = embeddings(i)
          val vj = embeddings(j)
          interactions += bilinearLayers(idx).forward(vi) * vj
          idx += 1
        }
    }

    val numInteractions = interactions.size
    if (numInteractions == 0) {
      return torch.empty(0L)
    }

    // Stack tensors along dim 0: (num_interactions, batch, embed_dim) -> (batch, num_interactions, embed_dim)
    torch.stack(new TensorVector(interactions.toSeq*), 0L).transpose(0L, 1L)
  }
}

/**
 * BiLinearInteractionLayer companion object with factory methods.
 */
object BiLinearInteractionLayer {
  def apply(inputDim: Long, numFields: Int, bilinearType: String = "field_interaction", device: String = DeviceSupport.backend): BiLinearInteractionLayer = {
    new BiLinearInteractionLayer(inputDim, numFields, bilinearType, device)
  }
}