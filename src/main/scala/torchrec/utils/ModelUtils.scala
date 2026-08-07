package torchrec.utils

import torchrec.basic.features._
import torchrec.Implicits._

import torchrec.utils.DeviceSupport

/**
 * Model utilities for introspection and dummy input generation.
 */
object ModelUtils {

  /**
   * Generate dummy input tensors based on feature definitions.
   * @param features List of Feature objects
   * @param batchSize Batch size for dummy input
   * @param seqLength Sequence length for SequenceFeature
   * @param device Device string ("cpu", "cuda:0", etc.)
   * @return Map of feature name -> Tensor
   */
  def generateDummyInput(
    features: List[Feature],
    batchSize: Int = 2,
    seqLength: Int = 10,
    device: String = DeviceSupport.backend
  ): Map[String, org.bytedeco.pytorch.Tensor] = {
    import org.bytedeco.pytorch._
    import org.bytedeco.pytorch.global.torch
    import org.bytedeco.pytorch.global.torch.ScalarType

    val result = scala.collection.mutable.Map[String, Tensor]()

    features.foreach {
      case f: SequenceFeature =>
        // Shape: (batchSize, seqLength)
        val data = Array.fill(batchSize * seqLength)(0L)
        val tensor = longTensor(data).reshape(batchSize.toLong, seqLength.toLong)
        if (device != "cpu") {
          result(f.name) = tensor.to(new Device(device), tensor.dtype())
        } else {
          result(f.name) = tensor
        }

      case f: SparseFeature =>
        // Shape: (batchSize,)
        val data = Array.fill(batchSize)(0L)
        val tensor = longTensor(data)
        if (device != "cpu") {
          result(f.name) = tensor.to(new Device(device), tensor.dtype())
        } else {
          result(f.name) = tensor
        }

      case f: DenseFeature =>
        // Shape: (batchSize, embedDim)
        val data = Array.fill(batchSize * f.embedDim)(0.0f)
        val tensor = floatTensor(data).reshape(batchSize.toLong, f.embedDim.toLong)
        if (device != "cpu") {
          result(f.name) = tensor.to(new Device(device), tensor.dtype())
        } else {
          result(f.name) = tensor
        }

      case _ =>
      // Unknown feature type, skip
    }

    result.toMap
  }

  /**
   * Extract unique feature names from a list.
   */
  def extractFeatureNames(features: List[Feature]): List[String] = {
    val seen = scala.collection.mutable.Set[String]()
    features.flatMap {
      case f: Feature if !seen.contains(f.name) =>
        seen += f.name
        Some(f.name)
      case _ => None
    }
  }

  /**
   * Categorize features by type.
   */
  def categorizeFeatures(
    features: List[Feature]
  ): (List[SparseFeature], List[DenseFeature], List[SequenceFeature]) = {
    val sparse = scala.collection.mutable.ListBuffer[SparseFeature]()
    val dense = scala.collection.mutable.ListBuffer[DenseFeature]()
    val seq = scala.collection.mutable.ListBuffer[SequenceFeature]()

    features.foreach {
      case f: SparseFeature => sparse += f
      case f: DenseFeature => dense += f
      case f: SequenceFeature => seq += f
      case _ =>
    }

    (sparse.toList, dense.toList, seq.toList)
  }

  /**
   * Count total embedding parameters for a feature list.
   */
  def countEmbeddingParams(
    features: List[Feature],
    embedDim: Int
  ): Long = {
    features.foldLeft(0L) { (acc, f) =>
      f match {
        case sf: SparseFeature => acc + sf.vocabSize * embedDim
        case sq: SequenceFeature => acc + sq.vocabSize * embedDim
        case _ => acc
      }
    }
  }

  /**
   * Generate dynamic axes configuration for ONNX export.
   */
  def generateDynamicAxes(
    inputNames: List[String],
    outputNames: List[String] = List("output"),
    batchDim: Int = 0,
    seqFeatures: List[String] = List.empty
  ): Map[String, Map[Int, String]] = {
    val axes = scala.collection.mutable.Map[String, Map[Int, String]]()

    // Input axes
    inputNames.foreach { name =>
      val dims = scala.collection.mutable.Map[Int, String]()
      dims(batchDim) = "batch_size"
      if (seqFeatures.contains(name)) {
        dims(1) = "seq_length"
      }
      axes(name) = dims.toMap
    }

    // Output axes
    outputNames.foreach { name =>
      axes(name) = Map(batchDim -> "batch_size")
    }

    axes.toMap
  }

  /**
   * Get model architecture summary as a string.
   */
  def modelSummary(
    model: org.bytedeco.pytorch.nn.Module,
    inputFeatures: List[Feature],
    embedDim: Int
  ): String = {
    val sb = new StringBuilder()
    sb.append("=" * 60 + "\n")
    sb.append("Model Summary\n")
    sb.append("=" * 60 + "\n\n")

    sb.append("Features:\n")
    inputFeatures.foreach { f =>
      sb.append(f"  ${f.name}: ${f.getClass.getSimpleName} (vocab=${f.vocabSize}, embed=$embedDim)\n")
    }

    sb.append(f"\nEmbedding Parameters: ${countEmbeddingParams(inputFeatures, embedDim)}%,d\n")

    sb.append("\nParameters:\n")
    var totalParams = 0L
    try {
      // Try to access state_dict - this may not be available in all JavaCPP builds
      throw new UnsupportedOperationException("state_dict not available in JavaCPP")
    } catch {
      case _: UnsupportedOperationException => sb.append("Parameters unavailable (state_dict not supported)\n")
      case _: Throwable => sb.append("Parameters unavailable\n")
    }
    sb.append(f"Model Size: ${totalParams * 4 / (1024 * 1024)}%.2f MB\n")
    sb.append("=" * 60)

    sb.toString
  }
}
