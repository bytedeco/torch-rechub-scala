package torchrec.basic.layers

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.basic.features.*

import scala.collection.mutable
import scala.collection.JavaConverters.seqAsJavaListConverter

/**
 * Return input masks from features.
 *
 * Shape
 * -----
 * Input
 *   x : dict
 *     ``{feature_name: feature_value}``; sequence ``(B, L)``, sparse/dense ``(B,)``.
 *   features : list or SparseFeature or SequenceFeature
 *     All elements must be sparse or sequence features.
 * Output
 *   - Sparse: ``(B, num_features)``
 *   - Sequence: ``(B, num_seq, seq_length)``
 */
class InputMask() extends Module {

  def forward(x: Map[String, Tensor], features: List[Feature]): Tensor = {
    val mask = mutable.ListBuffer[Tensor]()

    features.foreach {
      case sf: SparseFeature =>
        val paddingIdx = sf.paddingIdx
        val feaMask = if (paddingIdx.isDefined && paddingIdx.get >= 0) {
          x(sf.name).ne(new Scalar(paddingIdx.get))
        } else {
          x(sf.name).ne(new Scalar(-1L))
        }
        mask += feaMask.unsqueeze(1).toType(ScalarType.Float)
      case seqf: SequenceFeature =>
        val paddingIdx = seqf.paddingIdx
        val feaMask = if (paddingIdx >= 0) {
          x(seqf.name).ne(new Scalar(paddingIdx))
        } else {
          x(seqf.name).ne(new Scalar(-1L))
        }
        mask += feaMask.unsqueeze(1).toType(ScalarType.Float)
      case _ =>
        throw new IllegalArgumentException("Only SparseFeature or SequenceFeature support to get mask.")
    }

    torch.cat(new TensorVector(mask.toSeq*), 1L)
  }
}