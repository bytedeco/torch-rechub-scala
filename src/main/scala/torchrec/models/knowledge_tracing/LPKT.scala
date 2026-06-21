package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.utils.DeviceSupport

import scala.collection.mutable

/**
 * LPKT: Learning Persistence Knowledge Tracing
 */
class LPKT(
  numExercises: Long,
  numConcepts: Long,
  numActionTypes: Int = 1,
  embedDim: Int = 64,
  exerciseDim: Int = 64,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private val exerciseEmb = new EmbeddingImpl(new EmbeddingOptions(numExercises + 1, exerciseDim))
  register_module("exercise_emb", exerciseEmb)

  private val actionEmb = new EmbeddingImpl(new EmbeddingOptions(numActionTypes + 10, embedDim))
  register_module("action_emb", actionEmb)

  private val itemEmb = new EmbeddingImpl(new EmbeddingOptions(numExercises + 10, embedDim))
  register_module("item_emb", itemEmb)

  private val fc1 = new LinearImpl(embedDim * 3, embedDim)
  private val fc2 = new LinearImpl(embedDim, embedDim)
  private val fc3 = new LinearImpl(embedDim * 3, embedDim)
  private val fc4 = new LinearImpl(embedDim * 3, embedDim)
  private val predictor = new LinearImpl(exerciseDim + embedDim, 1)

  register_module("fc1", fc1)
  register_module("fc2", fc2)
  register_module("fc3", fc3)
  register_module("fc4", fc4)
  register_module("predictor", predictor)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    this.to(dev, false)
  }

  def forward(
    exerciseIds: Tensor,
    actionTypes: Tensor,
    knowledgeStates: Tensor
  ): Tensor = {
    val batchSize = exerciseIds.size(0).toInt
    val seqLen = exerciseIds.size(1).toInt

    val eEmbed = exerciseEmb.forward(exerciseIds).contiguous()
    val atEmbed = actionEmb.forward(actionTypes).contiguous()
    val itEmbed = itemEmb.forward(exerciseIds).contiguous()

    var hPre = torch.zeros(Array(batchSize.toLong, numConcepts, embedDim),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))

    val preds = mutable.ArrayBuffer[Tensor]()

    for (t <- 0 until seqLen) {
      val e_t = eEmbed.select(1, t).contiguous()
      val at_t = atEmbed.select(1, t).contiguous()
      val it_t = itEmbed.select(1, t).contiguous()

      val hPreAvg = hPre.mean(1).contiguous()

      val combined1 = torch.cat(new TensorVector(hPreAvg, e_t, it_t), 1).contiguous()
      val lc0 = torch.tanh(fc1.forward(combined1)).contiguous()

      val combined2 = torch.cat(new TensorVector(lc0, e_t, it_t), 1).contiguous()
      val gammaL = torch.sigmoid(fc3.forward(combined2)).contiguous()
      val LG = gammaL.mul(torch.tanh(fc2.forward(lc0)).add(new Scalar(1.0)).div(new Scalar(2.0))).contiguous()

      // Forgetting gate with detach to prevent gradient issues
      val itRepeat = it_t.unsqueeze(1).expand(batchSize.toLong, numConcepts, embedDim).clone().contiguous()
      val hPreDet = hPre.detach()
      val hPreFlat = hPreDet.mul(itRepeat).contiguous()
      val lgRepeat = LG.unsqueeze(1).expand(batchSize.toLong, numConcepts, embedDim).clone().contiguous()
      // Flatten 3D to 2D before concat (batch*numConcepts, embedDim*3)
      val hPreFlat2D = hPreFlat.view(-1L, embedDim)
      val lgRepeat2D = lgRepeat.view(-1L, embedDim)
      val itRepeat2D = itRepeat.view(-1L, embedDim)
      val combined3 = torch.cat(new TensorVector(hPreFlat2D, lgRepeat2D, itRepeat2D), 1).contiguous()
      val gammaF = torch.sigmoid(fc4.forward(combined3)).contiguous()
      val gammaF3D = gammaF.view(batchSize, numConcepts, embedDim)
      val hfTerm = gammaF3D.mul(hPre).contiguous()

      val hTildeNew = LG.unsqueeze(1).mul(itRepeat).contiguous()
      hPre = hfTerm.add(hTildeNew)

      val hFinal = hPre.mean(1).contiguous()
      val predCombined = torch.cat(new TensorVector(e_t, hFinal), 1).contiguous()
      val pred = torch.sigmoid(predictor.forward(predCombined))
      preds += pred.squeeze(1)
    }

    if (preds.isEmpty) {
      torch.zeros(Array(batchSize.toLong, seqLen.toLong),
        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    } else {
      torch.stack(new TensorVector(preds.toSeq: _*), 1)
    }
  }

  def predict(exerciseIds: Tensor, actionTypes: Tensor, knowledgeStates: Tensor): Tensor =
    forward(exerciseIds, actionTypes, knowledgeStates)
}