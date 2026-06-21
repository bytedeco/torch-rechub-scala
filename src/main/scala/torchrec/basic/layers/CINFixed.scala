package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import torchrec.utils.DeviceSupport
import torchrec.Implicits._

/**
 * Fixed Compressed Interaction Network from xDeepFM.
 * Creates all layers in the constructor so they are properly registered
 * with the module and visible to the optimizer.
 *
 * splitHalf: if true, halves the output size of intermediate layers (standard xDeepFM behavior).
 * Reference: "xDeepFM: Combining Explicit and Implicit Feature Interactions in Recommender Systems"
 * (KDD 2018)
 */
class CINFixed(
  numFields: Int,
  embedDim: Int,
  crossLayerSizes: List[Int],
  splitHalf: Boolean = true,
  device: String = DeviceSupport.backend
) extends Module {

  require(crossLayerSizes.nonEmpty, "crossLayerSizes cannot be empty")
  require(embedDim > 0, "embedDim must be positive")
  require(numFields > 0, "numFields must be positive")

  // Build layers eagerly in constructor so they are registered properly
  private val convLayers = collection.mutable.ListBuffer[org.bytedeco.pytorch.LinearImpl]()
  // Projection layers: maps split-half output dim to (num_fields * embed_dim) for next iteration
  private val projLayers = collection.mutable.ListBuffer[org.bytedeco.pytorch.LinearImpl]()

  for (i <- crossLayerSizes.indices) {
    val outDim = crossLayerSizes(i)
    // xh is (batch, F, F), flatten to F*F for conv
    val inChannels = numFields * numFields
    val conv = new org.bytedeco.pytorch.LinearImpl(inChannels, outDim)
    register_module(s"conv_$i", conv)
    conv.to(new org.bytedeco.pytorch.Device(device), false)
    convLayers += conv

    // Projection from split-half output dim to (num_fields * embed_dim) for next iteration
    // This is needed when splitHalf=true and this isn't the last layer
    if (splitHalf && i < crossLayerSizes.length - 1) {
      val projInDim = math.floor(outDim / 2.0).toInt max 1
      val proj = new org.bytedeco.pytorch.LinearImpl(projInDim, numFields * embedDim)
      register_module(s"proj_$i", proj)
      proj.to(new org.bytedeco.pytorch.Device(device), false)
      projLayers += proj
    }
  }

  def forward(embeddings: Tensor): Tensor = {
    // embeddings: (batch, num_fields, embed_dim)
    val batchSize = embeddings.size(0)

    // x0: original embeddings (stays constant throughout all layers)
    val x0 = embeddings
    // hk: output of previous layer, transposed to (batch, embed_dim, num_fields)
    var hk = x0.transpose(1, 2)  // (batch, embed_dim, num_fields) for first iteration

    val outputs = collection.mutable.ListBuffer[Tensor]()
    var projLayerIdx = 0

    for (i <- crossLayerSizes.indices) {
      // Outer product: x0 (batch, F, D) and hk (batch, D, F)
      // bmm gives (batch, F, F) - the field interaction map
      val xh = torch.bmm(x0, hk)
      // Flatten: (batch, F * F)
      val flat = xh.view(batchSize, numFields * numFields)

      // Convolution (1x1): compress from F*F to cross_layer_size
      var convOut = convLayers(i).forward(flat)
      convOut = convOut.relu()

      // Pool this layer's output and store (sum over field dimension)
      val pooled = convOut.sum(1).unsqueeze(1)  // (batch, 1)
      outputs += pooled

      // Prepare hk for next layer
      // hk must be (batch, embed_dim, num_fields) for bmm with x0
      if (splitHalf && i < crossLayerSizes.length - 1) {
        // After split-half, take first half of convOut channels
        val actualNextDim = math.floor(convOut.size(1) / 2.0).toInt max 1
        val half = convOut.narrow(1, 0, actualNextDim)  // (batch, actualNextDim)

        // Project from actualNextDim to (num_fields * embed_dim)
        val projOut = projLayers(projLayerIdx).forward(half)
        projLayerIdx += 1

        // Reshape: (batch, num_fields * embed_dim) -> (batch, num_fields, embed_dim)
        // Then transpose to (batch, embed_dim, num_fields)
        hk = projOut.view(batchSize, numFields, embedDim).transpose(1, 2)
      }
    }

    // Concatenate all pooled layer outputs: (batch, num_layers)
    import torchrec.Implicits.SeqTensorRichSeq
    val tensorSeq: Seq[Tensor] = outputs.toSeq
    val cinOut = torch.cat(new TensorVector(tensorSeq*), 1L)  // (batch, num_layers)

    // Sum over all layers to get final CIN output
    cinOut.sum(1).unsqueeze(1)  // (batch, 1)
  }
}
