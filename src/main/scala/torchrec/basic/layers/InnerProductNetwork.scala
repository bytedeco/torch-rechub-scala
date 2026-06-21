package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.utils.DeviceSupport

/**
 * Inner Product Network — computes pairwise inner products of field embeddings
 * to capture 2nd-order feature interactions explicitly.
 *
 * Reference: "Product-based Neural Networks for User Response Prediction" (SJTU, 2016)
 *
 * For each pair of fields (i, j) where i < j:
 *   output += <V_i, V_j>  (inner product)
 *
 * Input:  (batch, num_fields, embed_dim)
 * Output: (batch, num_pairs) — one scalar per field pair
 */
class InnerProductNetwork(
  device: String = DeviceSupport.backend
) extends Module {

  def forward(embeddings: Tensor): Tensor = {
    // embeddings: (batch, num_fields, embed_dim)
    val numFields = embeddings.size(1)

    val outputs = collection.mutable.ListBuffer[Tensor]()

    var i = 0
    while (i < numFields) {
      var j = i + 1
      while (j < numFields) {
        val vi = embeddings.narrow(1, i, 1).squeeze(1)  // (batch, embed_dim)
        val vj = embeddings.narrow(1, j, 1).squeeze(1)  // (batch, embed_dim)
        val ip = vi.mul(vj).sum(1).unsqueeze(1)  // (batch, 1)
        outputs += ip
        j += 1
      }
      i += 1
    }

    if (outputs.isEmpty) {
      val opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
      torch.zeros(Array(embeddings.size(0), 1), opts).to(embeddings.device(), ScalarType.Float)
    } else if (outputs.size == 1) {
      outputs.head
    } else {
      // Avoid torch.cat over a TensorVector which may fail if any intermediate tensor
      // lacks device metadata. Allocate result on embeddings.device() and copy columns.
      val numOut = outputs.size
      val opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
      val result = torch.zeros(Array(embeddings.size(0), numOut.toLong), opts).to(embeddings.device(), ScalarType.Float)
      var k = 0
      while (k < numOut) {
        result.narrow(1, k, 1).copy_(outputs(k))
        k += 1
      }
      result
    }
  }
}
