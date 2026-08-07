package torchrec.basic.layers

import org.bytedeco.pytorch.*
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
import torchrec.utils.DeviceSupport
import torchrec.Implicits.*
import torchrec.TensorImplicits.RichTensor

import scala.collection.mutable

class FFM(
  numFields: Int,
  reduceSum: Boolean = true,
  device: String = DeviceSupport.backend
) extends Module {

  override def forward(x: Tensor): Tensor = {
    val crossedEmbeddings = mutable.ListBuffer[Tensor]()

    for (i <- 0 until numFields - 1; j <- i + 1 until numFields) {
      val vi = x.select(1, i)
      val vj = x.select(1, j)
      crossedEmbeddings += vi * vj
    }

    // stacked: (num_interactions, batch, embed_dim)
    val stacked = torch.stack(new TensorVector(crossedEmbeddings.toSeq*))
    // transpose to (batch, num_interactions, embed_dim)
    val transposed = stacked.transpose(0L, 1L)

    if (reduceSum) {
      transposed.sum(-1)
    } else {
      transposed
    }
  }
}

object FFM {
  def apply(numFields: Int, reduceSum: Boolean = true, device: String = DeviceSupport.backend): FFM = {
    new FFM(numFields, reduceSum, device)
  }
}
