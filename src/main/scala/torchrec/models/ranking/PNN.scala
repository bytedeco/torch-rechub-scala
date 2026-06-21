package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Product-based Neural Network (PNN)
 *
 * Captures both first-order (linear) and second-order (product) feature interactions.
 * Uses an embedding layer + product layer (inner or outer product) + MLP.
 *
 * Two product types:
 *   - "inner": Inner product between field embeddings (learns scalar interactions)
 *   - "outer": Outer product via learnable kernel (learns vector interactions)
 *
 * Architecture:
 *   Sparse Input → EmbeddingLayer → Flat Embeddings + Product Features → MLP → Output
 *
 * Reference: "Product-based Neural Networks for User Response Prediction" (Song et al., 2016)
 *
 * @param features     List of sparse features
 * @param embedDim    Embedding dimension
 * @param mlpDims     Hidden layer dimensions for the MLP
 * @param productType One of "inner" or "outer"
 * @param dropout     Dropout rate
 * @param device      Device to run on
 */
class PNN(
  features: List[Feature],
  embedDim: Int = 8,
  mlpDims: List[Long] = List(256L, 128L, 64L),
  productType: String = "inner",
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(productType == "inner" || productType == "outer",
    s"productType must be 'inner' or 'outer', got $productType")

  private val numFields = features.collect { case f: SparseFeature => 1 }.size
  require(numFields >= 2, "PNN requires at least 2 sparse features")

  // Embedding layer
  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  // Number of field pairs
  private val numPairs = (numFields * (numFields - 1)) / 2

  // Product layer
  productType match {
    case "inner" =>
      // InnerProductNetwork is stateless, can be created here
    case "outer" =>
      // OuterProductNetwork has learnable parameters, needs registration
  }

  // MLP input dimension
  // - Flat embeddings: num_fields * embed_dim
  // - Product: num_pairs (inner) or num_pairs * embed_dim (outer)
  private val embedFlatDim = numFields * embedDim
  private val productDim = productType match {
    case "inner" => numPairs
    case "outer" => numPairs * embedDim
  }
  private val mlpInputDim = embedFlatDim + productDim

  // Outer product network (registered here in constructor)
  private val outerNet = productType match {
    case "outer" =>
      val opn = new OuterProductNetwork(numFields, embedDim, "vec", device)
      register_module("opn", opn)
      Some(opn)
    case _ => None
  }

  // MLP
  private val mlp = new MLP(mlpInputDim, mlpDims, 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  if (device != "cpu") {
    mlp.to(new org.bytedeco.pytorch.Device(device), false)
    ()
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Get embeddings: 3D and flat. Use EmbeddingLayer.forward for the flattened
    // version to ensure device alignment and avoid manual view on possibly
    // non-contiguous tensors.
    val embeddings = embeddingLayer.forward3D(sparseFeats)
    val batchSize = embeddings.size(0).toInt
    val flatEmbed = embeddingLayer.forward(sparseFeats) // (batch, num_fields * embed_dim)

    // Compute product interactions
    val productFeatures: Tensor = productType match {
      case "inner" =>
        // Inner product between all field pairs: scalar per pair
        val ip = new InnerProductNetwork()
        ip.forward(embeddings)  // (batch, num_pairs)

      case "outer" =>
        // Outer product via learnable kernel: vector per pair
        outerNet match {
          case Some(opn) => opn.forward(embeddings)  // (batch, num_pairs)
          case None => throw new IllegalStateException("outerNet not initialized")
        }
    }

    // Concatenate flat embeddings and product features
    // Build mlpInput by allocating on the target device and copying slices to avoid
    // any torch.cat device-metadata issues.
    val targetDev = flatEmbed.device()
    val batch = batchSize.toLong
    val opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
    val mlpInputTmp = torch.zeros(Array(batch, mlpInputDim.toLong), opts).to(targetDev, ScalarType.Float)

    // Copy flat embeddings into the left slice
    val left = flatEmbed.view(batch, embedFlatDim.toLong)
    mlpInputTmp.narrow(1L, 0L, embedFlatDim.toLong).copy_(left)

    // Ensure product features live on target device, then copy into the right slice
    var prodOnDev: Tensor = null.asInstanceOf[Tensor]
    try {
      prodOnDev = productFeatures.to(targetDev, productFeatures.dtype())
    } catch {
      case _: Throwable => println("[PNN] productFeatures.to(...) failed, falling back to allocation+copy_")
    }
    if (prodOnDev == null) {
      val d0 = productFeatures.size(0)
      val d1 = if (productFeatures.dim() > 1L) productFeatures.size(1) else 1L
      val tmp = torch.zeros(Array(d0, d1), opts).to(targetDev, ScalarType.Float)
      tmp.copy_(productFeatures)
      prodOnDev = tmp
    }

    val right = prodOnDev.view(batch, productDim.toLong)
    mlpInputTmp.narrow(1L, embedFlatDim.toLong, productDim.toLong).copy_(right)

    val mlpInput = mlpInputTmp

    // MLP
    mlp.forward(mlpInput).squeeze(1)
  }
}
