package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.utils.DeviceSupport

/**
 * SKVMN: Student-friendly Key-Value Memory Network
 */
class SKVMN(
  numConcepts: Long,
  embedDim: Int = 64,
  memSize: Int = 20,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private val kEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim))
  register_module("k_emb", kEmb)

  private val xEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2 + 2, embedDim))
  register_module("x_emb", xEmb)

  private val Mk = {
    val init = torch.randn(Array(memSize.toLong, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .mul(new Scalar(scala.math.sqrt(1.0 / embedDim.toDouble).toFloat))
    register_buffer("Mk", init)
    init
  }

  private val Mv0 = {
    val init = torch.randn(Array(memSize.toLong, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    register_buffer("Mv0", init)
    init
  }

  private val aEmbed = new LinearImpl(embedDim * 2, embedDim)
  register_module("a_embed", aEmbed)

  private val fLayer = new LinearImpl(embedDim * 2, embedDim)
  register_module("f_layer", fLayer)

  private val lstm = new LSTMImpl(embedDim, embedDim)
  register_module("lstm", lstm)

  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropout", dropoutLayer)

  private val pLayer = new LinearImpl(embedDim, 1)
  register_module("p_layer", pLayer)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    kEmb.to(dev, false)
    xEmb.to(dev, false)
    aEmbed.to(dev, false)
    fLayer.to(dev, false)
    lstm.to(dev, false)
    pLayer.to(dev, false)
  }

  def forward(
    conceptIds: Tensor,
    responses: Tensor
  ): Tensor = {
    val batchSize = conceptIds.size(0).toInt
    val seqLen = conceptIds.size(1).toInt

    // Convert to Long and clamp
    val cLong = conceptIds.toType(ScalarType.Long)
    val rLong = responses.toType(ScalarType.Long)

    val conceptIdx = cLong.clamp(
      new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(0)),
      new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(numConcepts.toDouble))
    ).toType(ScalarType.Long)
    val responseIdx = rLong.clamp(
      new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(0)),
      new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(1))
    ).toType(ScalarType.Long)

    val kEmbed = kEmb.forward(conceptIdx)

    // interaction_id = concept * 2 + response
    val interactionIds = conceptIdx.mul(new Scalar(2)).add(responseIdx)
    val interactionEmbeds = xEmb.forward(interactionIds)

    // Initialize memory - use repeat instead of expand for reliability
    val mv0Tensor = Mv0.clone()  // (memSize, embedDim)
    val memInit = mv0Tensor.unsqueeze(0)  // (1, memSize, embedDim)
    var currentMem = memInit.repeat(batchSize, 1, 1)  // (batchSize, memSize, embedDim)

    val fTList = scala.collection.mutable.ListBuffer[Tensor]()

    for (i <- 0 until seqLen) {
      val q = kEmbed.select(1, i)  // (batchSize, embedDim)

      // Compute attention scores using einsum-style computation
      // q: (batchSize, embedDim), Mk: (memSize, embedDim)
      // scores[b, m] = sum_e q[b, e] * Mk[m, e]
      val mkT = Mk.t()  // (embedDim, memSize)
      val scores = torch.mm(q, mkT)  // (batchSize, memSize)
      val attention = scores.softmax(1)  // (batchSize, memSize)

      // Read from memory using attention-weighted sum
      val attnExp = attention.unsqueeze(2)  // (batchSize, memSize, 1)
      val readContent = currentMem.mul(attnExp).sum(1)  // (batchSize, embedDim)

      // Concatenate and fuse
      val concat = torch.cat(new TensorVector(readContent, q), 1)  // (batchSize, 2*embedDim)
      val f = torch.tanh(fLayer.forward(concat))  // (batchSize, embedDim)
      fTList.append(f)

      // Memory update
      val y = interactionEmbeds.select(1, i)  // (batchSize, embedDim)
      val writeInput = torch.cat(new TensorVector(f, y), 1)  // (batchSize, 2*embedDim)
      val writeEmbed = aEmbed.forward(writeInput)  // (batchSize, embedDim)

      // Compute erase and add signals
      val eraseSignal = torch.sigmoid(writeEmbed)  // (batchSize, embedDim)
      val addSignal = torch.tanh(writeEmbed)  // (batchSize, embedDim)

      // Erase and add gates
      val eraseGate = eraseSignal.unsqueeze(1)  // (batchSize, 1, embedDim)
      val addGate = addSignal.unsqueeze(1)  // (batchSize, 1, embedDim)

      // Update memory
      val eraseFactor = attnExp.mul(eraseGate).neg().add(new Scalar(1.0))
      currentMem = currentMem.mul(eraseFactor).add(attnExp.mul(addGate))
    }

    // Stack fTList: each f has shape (batchSize, embedDim), result is (seqLen, batchSize, embedDim)
    val ft = torch.stack(new TensorVector(fTList.toSeq*), 0)

    val lstmOut = lstm.forward(ft).get0()

    // lstmOut shape: (seqLen, batchSize, embedDim) because LSTM was created
    // with batch_first = false. The final time-step is at dim 0, index seqLen-1.
    val lastHidden = lstmOut.select(0, seqLen - 1)

    val pred = pLayer.forward(dropoutLayer.forward(lastHidden))
    pred.sigmoid()
  }

  def predict(conceptIds: Tensor, responses: Tensor): Tensor = forward(conceptIds, responses)
}
