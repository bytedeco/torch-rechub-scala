package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.utils.DeviceSupport

/**
 * Self-attention multi-interest module (Comirec).
 *
 * Parameters
 * ----------
 * embeddingDim : int
 *   Item embedding dimension.
 * interestNum : int
 *   Number of interests.
 * hiddenDim : int, optional
 *   Hidden dimension; defaults to ``4 * embedding_dim`` if None.
 *
 * Shape
 * -----
 * Input
 *   seqEmb : ``(B, L, D)``
 *   mask : ``(B, L, 1)``
 * Output
 *   ``(B, interest_num, D)``
 */
class MultiInterestSA(
  embeddingDim: Int,
  interestNum: Int,
  hiddenDim: Option[Int] = None,
  device: String = DeviceSupport.backend
) extends Module {

  private val actualHiddenDim = hiddenDim.getOrElse(embeddingDim * 4)

  // Initialize parameters with Xavier
  private val w1 = torch.rand(Array[Long](embeddingDim.toLong, actualHiddenDim.toLong),
    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  private val w2 = torch.rand(Array[Long](actualHiddenDim.toLong, interestNum.toLong),
    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  private val w3 = torch.rand(Array[Long](embeddingDim.toLong, embeddingDim.toLong),
    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))

  register_parameter("w1", w1)
  register_parameter("w2", w2)
  register_parameter("w3", w3)

  def forward(seqEmb: Tensor, mask: Option[Tensor]  = None): Tensor = {
    // H = seq_emb @ W1
    val h = torch.matmul(seqEmb, w1).tanh()

    // A = H @ W2
    val batchSize = seqEmb.size(0)
    val seqLen = seqEmb.size(1)
    var a = torch.matmul(h, w2)

    if (mask != null && mask.isDefined) {
      // Apply mask with large negative value
      val maskedA = a.add(mask.get.mul(new Scalar(-1e9f)).add(new Scalar(1e9f)))
      a = torch.softmax(maskedA, 1)
    } else {
      a = torch.softmax(a, 1)
    }

    // A: (batch, seq, interest) after transpose
    val aTransposed = a.transpose(1, 2)

    // multi_interest_emb: (batch, interest, D)
    val multiInterestEmb = torch.matmul(aTransposed, seqEmb)
    multiInterestEmb
  }
}

/**
 * MultiInterestSA companion object with factory methods.
 */
object MultiInterestSA {
  def apply(embeddingDim: Int, interestNum: Int, hiddenDim: Option[Int] = None, device: String = DeviceSupport.backend): MultiInterestSA = {
    new MultiInterestSA(embeddingDim, interestNum, hiddenDim, device)
  }
}