package torchrec.models.knowledge_tracing.datasets

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.data.{Batch, Dataset}

/**
 * Knowledge Tracing Dataset.
 *
 * Stores sequences of:
 * - concept_seqs: concept IDs for each interaction (batch, seq_len)
 * - question_seqs: question IDs for each interaction (batch, seq_len)
 * - response_seqs: 0/1 responses for each interaction (batch, seq_len)
 * - mask_seqs: valid position mask (batch, seq_len)
 * - time_seqs: optional timestamps (batch, seq_len)
 * - target_concepts: next concept to predict (batch, seq_len) -- shifted
 * - target_responses: next response to predict (batch, seq_len) -- shifted
 *
 * For prediction at position t, input is 0..t-1, target is t.
 */
class KTDataset(
  val conceptSeqs: Tensor,           // (num_samples, seq_len)
  val questionSeqs: Tensor,         // (num_samples, seq_len)
  val responseSeqs: Tensor,         // (num_samples, seq_len)
  val maskSeqs: Tensor,             // (num_samples, seq_len) - 1 for valid, 0 for padding
  val labelSeqs: Tensor,            // (num_samples, seq_len) - shifted responses (target)
  val timeSeqs: Option[Tensor] = None,  // (num_samples, seq_len)
  val targetConcepts: Option[Tensor] = None  // (num_samples, seq_len) - shifted concepts
) extends Dataset {

  require(conceptSeqs.size(0) == responseSeqs.size(0))
  require(conceptSeqs.size(1) == responseSeqs.size(1))
  require(maskSeqs.size(0) == responseSeqs.size(0))
  require(maskSeqs.size(1) == responseSeqs.size(1))

  override def size: Long = conceptSeqs.size(0)

  override def get(index: Long): Batch = {
    val safeIndex = math.min(index.toInt, (conceptSeqs.size(0) - 1).toInt).max(0)
    Batch(
      sparseFeatures = Map(
        "concepts" -> conceptSeqs.narrow(0, safeIndex, 1).squeeze(0),
        "questions" -> questionSeqs.narrow(0, safeIndex, 1).squeeze(0),
        "responses" -> responseSeqs.narrow(0, safeIndex, 1).squeeze(0)
      ),
      sequenceFeatures = Map(
        "concept_seq" -> conceptSeqs.narrow(0, safeIndex, 1).squeeze(0),
        "question_seq" -> questionSeqs.narrow(0, safeIndex, 1).squeeze(0),
        "response_seq" -> responseSeqs.narrow(0, safeIndex, 1).squeeze(0)
      ),
      labels = Some(labelSeqs.narrow(0, safeIndex, 1).squeeze(0)),
      timeDiffs = timeSeqs.map(t => t.narrow(0, safeIndex, 1).squeeze(0)),
      targets = targetConcepts.map(t => t.narrow(0, safeIndex, 1).squeeze(0))
    )
  }

  /** Create a JavaDataset adapter for use with JavaCPP DataLoader. */
  def asJavaDataset(): JavaDataset = {
    new KTDatasetAdapter(this)
  }
}

/**
 * Java adapter so KTDataset works with JavaCPP DataLoader.
 */
class KTDatasetAdapter(private val dataset: KTDataset) extends JavaDataset {

  override def size(): SizeTOptional = new SizeTOptional(dataset.size)

  override def get(index: Long): Example = {
    val batch = dataset.get(index)
    val example = new Example()
    val keys = batch.sparseFeatures.keys.toArray
    val dataArr = keys.flatMap { k =>
      val t = batch.sparseFeatures(k)
      val arr = try t.toFloatArray catch { case _ => Array(0.0f) }
      arr
    }
    if (dataArr.nonEmpty) {
      val data = torchrec.Implicits.tensor(dataArr, Array(dataArr.length.toLong))
        .toType(ScalarType.Long)
      example.data(data)
    }
    example
  }
}
