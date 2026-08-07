package torchrec.utils

import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.*
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

import java.io.File

/** Configuration for the LLM inference engine */
class Config(
  val model: String,
  var maxNumBatchedTokens: Int = 16384,
  var maxNumSeqs: Int = 512,
  var maxModelLen: Int = 4096,
  var gpuMemoryUtilization: Float = 0.9f,
  var computeDtype: ScalarType = ScalarType.Half,  // Default to FP16 to save memory
  var tensorParallelSize: Int = 1,
  var enforceEager: Boolean = false,
  var kvcacheBlockSize: Int = 256,
  var numKvcacheBlocks: Int = -1,
) {
  require(model != null, "model path cannot be null")

  // HuggingFace config fields
  var modelType: String = ""
  var architectures: List[String] = Nil
  var hiddenSize: Int = 0
  var intermediateSize: Int = 0
  var numHiddenLayers: Int = 0
  var numAttentionHeads: Int = 0
  var numKeyValueHeads: Int = 0
  var vocabSize: Int = 0
  var headDim: Int = 0
  var maxPositionEmbeddings: Int = 0
  var rmsNormEps: Float = 1e-6f
  var ropeTheta: Float = 1000000f
  var ropeScaling: String = ""
  var ropeFactor: Float = 1.0f
  var hiddenAct: String = "silu"
  var mropeSection: List[Int] = Nil
  var tieWordEmbeddings: Boolean = false
  var attentionBias: Boolean = false
  var mlpBias: Boolean = false
  var slidingWindow: Int = -1
  var eos: Int = -1
  var bos: Int = -1

  /** Normalised model family derived from `model_type`. */
  def modelFamily: String = {
    val mt = Option(modelType).map(_.trim.toLowerCase).getOrElse("")
    if (mt == "qwen3_asr") "qwen3_asr"
    else if (mt.startsWith("qwen")) "qwen"
    else if (mt == "mistral") "mistral"
    else if (mt == "llama" || mt == "deepseek") "llama"
    else if (mt == "glm_ocr") "glm_ocr"
    else "unknown"
  }

  /** Llama, Mistral, GLM-OCR, and Qwen3-ASR text decoders share similar paths. */
  def isLlamaLike: Boolean = { val f = modelFamily; f == "llama" || f == "mistral" }
  def isGLMOcr: Boolean = modelFamily == "glm_ocr"
  def isQwenASR: Boolean = modelFamily == "qwen3_asr"

  require(kvcacheBlockSize % 256 == 0, "kvcacheBlockSize must be multiple of 256")
  require(tensorParallelSize >= 1 && tensorParallelSize <= 8, "tensorParallelSize must be 1-8")

  def loadFromHuggingFace(modelPath: String): Unit = {
    val configFile = new File(modelPath, "config.json")
    if (!configFile.exists()) {
      throw new IllegalArgumentException(s"Config file not found: ${configFile.getAbsolutePath}")
    }

    val content = scala.io.Source.fromFile(configFile).mkString
    val json = parseJson(content)

    // Model identity
    modelType = getString(json, "model_type", "")
    architectures = json.get("architectures") match {
      case Some(xs: List[_] @unchecked) => xs.map(_.toString)
      case Some(x) => List(x.toString)
      case None => Nil
    }

    // GLM-OCR has nested text_config and vision_config
    // Qwen3-ASR has text_config nested under thinker_config.text_config
    // For both, the text decoder params come from the text_config map.
    val textConfig = modelType match {
      case "qwen3_asr" =>
        // Text config is under thinker_config.text_config
        json.get("thinker_config") match {
          case Some(tc: Map[String, Any] @unchecked) =>
            tc.get("text_config") match {
              case Some(t: Map[String, Any] @unchecked) => t
              case _ => json // fallback
            }
          case _ => json // fallback
        }
      case _ =>
        // Standard: text_config at root level (GLM-OCR, standard models)
        json.get("text_config") match {
          case Some(m: Map[String, Any] @unchecked) => m
          case _ => json
        }
    }

    hiddenSize = getInt(textConfig, "hidden_size", 1024)
    intermediateSize = getInt(textConfig, "intermediate_size", 3072)
    numHiddenLayers = getInt(textConfig, "num_hidden_layers", 28)
    // Read numAttentionHeads directly from config (Q heads for GQA)
    numAttentionHeads = getInt(textConfig, "num_attention_heads", hiddenSize / getInt(textConfig, "head_dim", 128))
    // headDim: use explicit head_dim if present, else compute from hidden/numHeads
    headDim = getInt(textConfig, "head_dim", hiddenSize / numAttentionHeads)
    // numKeyValueHeads: use explicit kv_heads if present
    numKeyValueHeads = getInt(textConfig, "num_key_value_heads", getInt(textConfig, "num_kv_heads", numAttentionHeads))
    vocabSize = getInt(json, "vocab_size", getInt(textConfig, "vocab_size", 151936))
    maxPositionEmbeddings = getInt(textConfig, "max_position_embeddings", 32768)
    rmsNormEps = getFloat(textConfig, "rms_norm_eps", 1e-6f)
    ropeTheta = getFloat(textConfig, "rope_theta", 1000000f)
    hiddenAct = getString(textConfig, "hidden_act", "silu")
    tieWordEmbeddings = getBool(textConfig, "tie_word_embeddings", false)
    attentionBias = getBool(textConfig, "attention_bias", false)
    mlpBias = getBool(textConfig, "mlp_bias", false)
    slidingWindow = textConfig.get("sliding_window") match {
      case Some(n: Number) => n.intValue()
      case _ => -1
    }
    bos = getInt(json, "bos_token_id", getInt(textConfig, "bos_token_id", -1))
    eos = json.get("eos_token_id") match {
      case Some(n: Number) => n.intValue()
      case Some(xs: List[_] @unchecked) if xs.nonEmpty => xs.head.asInstanceOf[Number].intValue()
      case _ => getInt(textConfig, "eos_token_id", -1)
    }

    textConfig.get("rope_scaling") match {
      case Some(rsMap: Map[String, Any] @unchecked) =>
        ropeScaling = rsMap.get("rope_type") match {
          case Some(s: String) => s
          case _ => ""
        }
        ropeFactor = rsMap.get("factor") match {
          case Some(n: Number) => n.floatValue()
          case _ => 1.0f
        }
        mropeSection = rsMap.get("mrope_section") match {
          case Some(xs: List[_] @unchecked) => xs.map(_.asInstanceOf[Number].intValue())
          case Some(n: Number) => List(n.intValue())
          case _ => Nil
        }
      case _ =>
    }

    maxModelLen = math.min(maxModelLen, maxPositionEmbeddings)

    println(s"  Loaded config: model_type=$modelType, hidden=$hiddenSize, layers=$numHiddenLayers, heads=$numAttentionHeads, kv_heads=$numKeyValueHeads, vocab=$vocabSize, head_dim=$headDim, rope_scaling=$ropeScaling, mrope=$mropeSection")
  }

  private def parseJson(content: String): Map[String, Any] = {
    SimpleJson.parse(content).asInstanceOf[Map[String, Any]]
  }

  private def getInt(json: Map[String, Any], key: String, default: Int): Int =
    json.get(key) match {
      case Some(n: Number) => n.intValue
      case Some(xs: List[_]) if xs.nonEmpty => xs.head match {
        case n: Number => n.intValue
        case _ => default
      }
      case _ => default
    }

  private def getFloat(json: Map[String, Any], key: String, default: Float): Float =
    json.get(key).map(_.asInstanceOf[Number].floatValue).getOrElse(default)

  private def getString(json: Map[String, Any], key: String, default: String): String =
    json.get(key).map(_.toString).getOrElse(default)

  private def getBool(json: Map[String, Any], key: String, default: Boolean): Boolean =
    json.get(key).map(_.asInstanceOf[Boolean]).getOrElse(default)

  val deviceType: String = DeviceSupport.backend
  val device: Device = DeviceSupport.deviceOf(deviceType)

  val CUDA_OPTS: TensorOptions = new TensorOptions()
    .dtype(new ScalarTypeOptional(computeDtype))
    .device(new DeviceOptional(device))

  val LONG_OPTS: TensorOptions = new TensorOptions()
    .dtype(new ScalarTypeOptional(ScalarType.Long))
    .device(new DeviceOptional(device))

  val CPU_OPTS: TensorOptions = new TensorOptions()
    .dtype(new ScalarTypeOptional(ScalarType.Float))
    .device(new DeviceOptional(new Device("cpu")))
}
