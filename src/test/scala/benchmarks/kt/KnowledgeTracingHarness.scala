package benchmarks.kt

import org.bytedeco.pytorch.{Adam, AdamOptions, Device, Module, Scalar, Tensor, TensorOptions, TensorVector}
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.{RichTensor, tensor, longTensor}
import torchrec.basic.losses.{BCELoss, BCEWithLogitsLoss}
import torchrec.basic.metrics.{AUC, Accuracy, LogLoss}
import torchrec.data.{DataLoader, Dataset, Batch}
import torchrec.models.knowledge_tracing.*
import torchrec.utils.DeviceSupport

import scala.collection.mutable
import scala.io.Source
import scala.util.Random

object KnowledgeTracingHarness {

  case class KTSequence(
    questionIds: Vector[Int],
    conceptIds: Vector[Int],
    responses: Vector[Int]
  )

  case class KTMetadata(
    datasetName: String,
    trainPath: String,
    testPath: String,
    trainSequences: Int,
    testSequences: Int,
    numQuestions: Int,
    numConcepts: Int,
    minTrainLen: Int,
    maxTrainLen: Int,
    avgTrainLen: Double,
    maxSeqLen: Int,
    windowStride: Int,
    paddingId: Int = 0,
    startResponseId: Int = 2
  )

  case class KTWindow(
    questionIds: Array[Int],
    conceptIds: Array[Int],
    responses: Array[Int],
    targetConcepts: Array[Int],
    labels: Array[Float],
    mask: Array[Float]
  )

  case class KTDatasets(
    train: KTHarnessDataset,
    test: KTHarnessDataset,
    metadata: KTMetadata
  )

  case class KTRunConfig(
    datasetRoot: String = "/home/muller/IdeaProjects/Knowledge-Tracing-Datasets",
    datasetName: String,
    modelName: String,
    epochs: Int = 1,
    batchSize: Int = 32,
    maxSeqLen: Int = 200,
    windowStride: Int = 200,
    learningRate: Float = 1e-3f,
    embedDim: Int = 64,
    device: String = "cpu",
    seed: Int = 2026,
    limitTrainSequences: Int = 0,
    limitTestSequences: Int = 0
  )

  case class KTRunResult(
    datasetName: String,
    modelName: String,
    metrics: Map[String, Float],
    metadata: KTMetadata,
    numTrainWindows: Int,
    numTestWindows: Int
  )

  class KTHarnessDataset(val windows: Vector[KTWindow]) extends Dataset {
    override def size: Long = windows.size.toLong

    override def get(index: Long): Batch = {
      val safe = math.max(0, math.min(index.toInt, windows.size - 1))
      val sample = windows(safe)
        // Ensure index tensors are created as explicit Long tensors to avoid
        // accidental float dtypes being passed into embedding/index_select
        // operations in models.
        Batch(
          sparseFeatures = Map(
            "questions" -> longTensor(sample.questionIds.map(_.toLong).toArray),
            "concepts" -> longTensor(sample.conceptIds.map(_.toLong).toArray),
            "responses" -> longTensor(sample.responses.map(_.toLong).toArray)
          ),
          denseFeatures = Map(
            // mask is a float tensor
            "mask" -> tensor(sample.mask, Array(sample.mask.length.toLong))
          ),
          sequenceFeatures = Map(
            "target_concepts" -> longTensor(sample.targetConcepts.map(_.toLong).toArray)
          ),
          labels = Some(tensor(sample.labels, Array(sample.labels.length.toLong)).toType(ScalarType.Float))
        )
    }
  }

  object Args {
    def parse(args: Array[String]): Map[String, String] = {
      val out = mutable.Map.empty[String, String]
      var i = 0
      while (i < args.length) {
        val cur = args(i)
        if (cur.startsWith("--")) {
          val key = cur.stripPrefix("--")
          val value = if (i + 1 < args.length && !args(i + 1).startsWith("--")) {
            i += 1
            args(i)
          } else "true"
          out += key -> value
        }
        i += 1
      }
      out.toMap
    }
  }

  def run(config: KTRunConfig): KTRunResult = {
    val datasets = loadDatasets(config)
    val model = buildModel(config.modelName, datasets.metadata, config)
    val trainer = new KTTrainer(model, config.modelName, datasets.metadata, learningRate = config.learningRate, device = config.device, epochs = config.epochs)
    val trainLoader = new DataLoader(datasets.train, batchSize = config.batchSize, shuffle = true, device = config.device)
    val testLoader = new DataLoader(datasets.test, batchSize = config.batchSize, shuffle = false, device = config.device)
    trainer.fit(trainLoader)
    val metrics = trainer.evaluate(testLoader)
    KTRunResult(config.datasetName, config.modelName, metrics, datasets.metadata, datasets.train.windows.size, datasets.test.windows.size)
  }

  def loadDatasets(config: KTRunConfig): KTDatasets = {
    val trainPath = s"${config.datasetRoot}/${config.datasetName}/train.csv"
    val testPath = s"${config.datasetRoot}/${config.datasetName}/test.csv"
    val rawTrain = readRawDataset(trainPath)
    val rawTest = readRawDataset(testPath)
    val trainSeqs0 = buildSequences(rawTrain._1, rawTrain._2, config.datasetName)
    val testSeqs0 = buildSequences(rawTest._1, rawTest._2, config.datasetName)
    val trainSeqsLimited = if (config.limitTrainSequences > 0) trainSeqs0.take(config.limitTrainSequences) else trainSeqs0
    val testSeqsLimited = if (config.limitTestSequences > 0) testSeqs0.take(config.limitTestSequences) else testSeqs0

    val allQuestions = (trainSeqsLimited ++ testSeqsLimited).flatMap(_.questionIds).distinct.sorted
    val allConcepts = (trainSeqsLimited ++ testSeqsLimited).flatMap(_.conceptIds).distinct.sorted
    val questionMap = allQuestions.zipWithIndex.toMap
    val conceptMap = allConcepts.zipWithIndex.toMap

    def remap(seq: KTSequence): KTSequence =
      KTSequence(
        questionIds = seq.questionIds.map(questionMap),
        conceptIds = seq.conceptIds.map(conceptMap),
        responses = seq.responses
      )

    val trainSeqs = trainSeqsLimited.map(remap)
    val testSeqs = testSeqsLimited.map(remap)
    val lengths = trainSeqs.map(_.responses.length)
    val effectiveSeqLen = math.max(16, config.maxSeqLen)
    val stride = math.max(1, math.min(config.windowStride, effectiveSeqLen))

    val metadata = KTMetadata(
      datasetName = config.datasetName,
      trainPath = trainPath,
      testPath = testPath,
      trainSequences = trainSeqs.size,
      testSequences = testSeqs.size,
      numQuestions = allQuestions.size,
      numConcepts = allConcepts.size,
      minTrainLen = if (lengths.nonEmpty) lengths.min else 0,
      maxTrainLen = if (lengths.nonEmpty) lengths.max else 0,
      avgTrainLen = if (lengths.nonEmpty) lengths.sum.toDouble / lengths.size.toDouble else 0.0,
      maxSeqLen = effectiveSeqLen,
      windowStride = stride
    )

    val trainWindows = createWindows(trainSeqs, metadata)
    val testWindows = createWindows(testSeqs, metadata)

    KTDatasets(new KTHarnessDataset(trainWindows), new KTHarnessDataset(testWindows), metadata)
  }

  private def readRawDataset(path: String): (Vector[Vector[Int]], Vector[Vector[Int]]) = {
    val src = Source.fromFile(path)
    val lines = try src.getLines().map(_.trim).filter(_.nonEmpty).toVector finally src.close()
    require(lines.size % 3 == 0, s"Dataset $path has malformed 3-line records: ${lines.size}")
    val ids = mutable.ArrayBuffer.empty[Vector[Int]]
    val responses = mutable.ArrayBuffer.empty[Vector[Int]]
    var i = 0
    while (i < lines.size) {
      val len = lines(i).toInt
      val seq = lines(i + 1).split(",", -1).filter(_.nonEmpty).map(_.trim.toInt).toVector
      val resp = lines(i + 2).split(",", -1).filter(_.nonEmpty).map(_.trim.toInt).toVector
      if (seq.length == len && resp.length == len && len >= 2 && resp.forall(r => r == 0 || r == 1)) {
        ids += seq
        responses += resp
      }
      i += 3
    }
    (ids.toVector, responses.toVector)
  }

  private def buildSequences(rawIds: Vector[Vector[Int]], rawResponses: Vector[Vector[Int]], datasetName: String): Vector[KTSequence] = {
    rawIds.zip(rawResponses).map { case (ids, resps) =>
      datasetName.toLowerCase match {
        case "statics11" =>
          KTSequence(questionIds = ids, conceptIds = ids, responses = resps)
        case _ =>
          KTSequence(questionIds = ids, conceptIds = ids, responses = resps)
      }
    }.filter(_.responses.length >= 2)
  }

  private def createWindows(seqs: Vector[KTSequence], metadata: KTMetadata): Vector[KTWindow] = {
    val windows = mutable.ArrayBuffer.empty[KTWindow]
    val maxLen = metadata.maxSeqLen
    val stride = metadata.windowStride

    seqs.foreach { seq =>
      val total = seq.responses.length
      val starts = if (total <= maxLen) Vector(0) else (0 to math.max(0, total - 2) by stride).toVector
      starts.foreach { start =>
        val endExclusive = math.min(total, start + maxLen + 1)
        val qSlice = seq.questionIds.slice(start, endExclusive)
        val cSlice = seq.conceptIds.slice(start, endExclusive)
        val rSlice = seq.responses.slice(start, endExclusive)
        if (qSlice.length >= 2 && cSlice.length == qSlice.length && rSlice.length == qSlice.length) {
          val inputLen = math.min(maxLen, qSlice.length - 1)
          val qIn = qSlice.take(inputLen)
          val cIn = cSlice.take(inputLen)
          val rIn = rSlice.take(inputLen)
          val cTarget = cSlice.slice(1, inputLen + 1)
          val yTarget = rSlice.slice(1, inputLen + 1).map(_.toFloat)
          if (qIn.nonEmpty && cTarget.length == qIn.length && yTarget.length == qIn.length) {
            val qPad = Array.fill(maxLen)(metadata.paddingId)
            val cPad = Array.fill(maxLen)(metadata.paddingId)
            val rPad = Array.fill(maxLen)(metadata.paddingId)
            val tPad = Array.fill(maxLen)(metadata.paddingId)
            val yPad = Array.fill(maxLen)(0.0f)
            val mPad = Array.fill(maxLen)(0.0f)
            var i = 0
            while (i < inputLen) {
              qPad(i) = qIn(i)
              cPad(i) = cIn(i)
              rPad(i) = rIn(i)
              tPad(i) = cTarget(i)
              yPad(i) = yTarget(i)
              mPad(i) = 1.0f
              i += 1
            }
            windows += KTWindow(qPad, cPad, rPad, tPad, yPad, mPad)
          }
        }
      }
    }
    windows.toVector
  }

  private def buildModel(modelName: String, meta: KTMetadata, config: KTRunConfig): Module = {
    val numConcepts = math.max(meta.numConcepts, 2).toLong
    val numQuestions = math.max(meta.numQuestions, numConcepts.toInt).toLong
    val d = config.embedDim
    val device = config.device

    modelName match {
      case "DKT" => new DKT(numConcepts, embedDim = d, device = device)
      case "DKTForget" => new DKTForget(numConcepts, embedDim = d, device = device)
      case "DKVMN" => new DKVMN(numConcepts, numQuestions = numQuestions, memDim = d, memSize = 20, device = device)
      case "DeepIRT" => new DeepIRT(numConcepts, numQuestions = numQuestions, memDim = d, memSize = 20, device = device)
      case "GKT" => new GKT(numConcepts, embedDim = d, hiddenDim = d, device = device)
      case "AKT" => new AKT(numConcepts, embedDim = d, numHeads = 4, numBlocks = 2, ffnDim = d * 4, device = device)
      case "ATDKT" => new ATDKT(numConcepts, embedDim = d, numLayers = 1, numHeads = 4, device = device)
      case "ATKT" => new ATKT(numConcepts, skillDim = d / 2, answerDim = d / 2, hiddenDim = d, attentionDim = math.max(32, d), device = device)
      case "CSKT" => new CSKT(numConcepts, embedDim = d, numHeads = 4, numBlocks = 2, device = device)
      case "DIMKT" => new DIMKT(numConcepts, embedDim = d, numHeads = 4, numBlocks = 2, hiddenDim = d, device = device)
      case "IEKT" => new IEKT(numConcepts, embedDim = d, numCogLevels = 8, numAcqLevels = 8, device = device)
      case "LPKT" => new LPKT(numExercises = numQuestions, numConcepts = numConcepts, numActionTypes = 2, embedDim = d, exerciseDim = d, device = device)
      case "MTKT" => new MTKT(numConcepts, embedDim = d, numHeads = 4, numBlocks = 2, device = device)
      case "PromptKT" => new PromptKT(numConcepts, embedDim = d, numHeads = 4, numBlocks = 2, device = device)
      case "QDKT" => new QDKT(numQuestions = numQuestions, numConcepts = numConcepts, embedDim = d, device = device)
      case "RKT" => new RKT(numConcepts, embedDim = d, numHeads = 4, numLayers = 1, device = device)
      case "RobustKT" => new RobustKT(numConcepts, embedDim = d, numHeads = 4, numBlocks = 2, device = device)
      case "SAINT" => new SAINT(numExercises = numQuestions, numCategories = numConcepts, embedDim = d, numHeads = 4, numEncoderBlocks = 2, numDecoderBlocks = 2, ffnDim = d * 4, device = device)
      case "SAINTPlusPlus" => new SAINTPlusPlus(numExercises = numQuestions, numCategories = numConcepts, embedDim = d, numHeads = 4, numEncoderBlocks = 2, numDecoderBlocks = 2, ffnDim = d * 4, device = device)
      case "SAKT" => new SAKT(numConcepts, embedDim = d, numHeads = 4, numBlocks = 2, device = device)
      case "SKVMN" => new SKVMN(numConcepts, embedDim = d, memSize = 20, device = device)
      case "StableKT" => new StableKT(numConcepts, embedDim = d, numHeads = 4, numBlocks = 2, device = device)
      case "UKT" => new UKT(numConcepts, embedDim = d, numHeads = 4, numBlocks = 2, device = device)
      case other => throw new IllegalArgumentException(s"Unsupported KT model: $other")
    }
  }

  class KTTrainer(
    model: Module,
    modelName: String,
    metadata: KTMetadata,
    learningRate: Float,
    device: String,
    epochs: Int
  ) {
    private val optimizer = new Adam(model.parameters(), new AdamOptions(learningRate.toDouble))
    private val bceLogits = new BCEWithLogitsLoss()
    private val bce = new BCELoss()

    def fit(trainLoader: DataLoader): Unit = {
      torchrec.utils.DataLoaderDevice.set(device)
      model.train(true)
      var epoch = 0
      while (epoch < epochs) {
        val iter = trainLoader.iterator
        while (iter.hasNext) {
          val batch = iter.next()
          optimizer.zero_grad()
          val loss = computeLoss(batch)
          loss.backward()
          optimizer.step()
        }
        epoch += 1
      }
    }

    def evaluate(loader: DataLoader): Map[String, Float] = {
      torchrec.utils.DataLoaderDevice.set(device)
      model.eval()
      val preds = mutable.ArrayBuffer.empty[Float]
      val labels = mutable.ArrayBuffer.empty[Float]
      val iter = loader.iterator
      while (iter.hasNext) {
        val batch = iter.next()
        val (p, y) = extractPredAndLabels(batch)
        preds ++= p
        labels ++= y
      }
      if (preds.isEmpty || labels.isEmpty) {
        Map("AUC" -> 0.0f, "LogLoss" -> 0.0f, "Accuracy" -> 0.0f)
      } else {
        val auc = new AUC(); auc.update(preds.toArray, labels.toArray)
        val logLoss = new LogLoss(); logLoss.update(preds.toArray, labels.toArray)
        val acc = new Accuracy(); acc.update(preds.toArray, labels.toArray)
        Map("AUC" -> auc.compute(), "LogLoss" -> logLoss.compute(), "Accuracy" -> acc.compute())
      }
    }

    private def computeLoss(batch: Batch): Tensor = {
      val concepts = batch.sparseFeatures("concepts")
      val questions = batch.sparseFeatures("questions")
      val responses = batch.sparseFeatures("responses")
      val targets = batch.sequenceFeatures("target_concepts")
      val labels = batch.labels.get.toType(ScalarType.Float)
      val mask = batch.denseFeatures("mask").toType(ScalarType.Float)
      val logitsOrProb = forwardByModel(concepts, questions, responses, targets)
      val (selected, isLogits) = selectPredictions(logitsOrProb, targets)
      val validPred = selected.masked_select(mask.toType(ScalarType.Bool))
      val validLabel = labels.masked_select(mask.toType(ScalarType.Bool))
      if (isLogits) bceLogits(validPred, validLabel) else bce(validPred, validLabel)
    }

    private def extractPredAndLabels(batch: Batch): (Array[Float], Array[Float]) = {
      val concepts = batch.sparseFeatures("concepts")
      val questions = batch.sparseFeatures("questions")
      val responses = batch.sparseFeatures("responses")
      val targets = batch.sequenceFeatures("target_concepts")
      val labels = batch.labels.get.toType(ScalarType.Float)
      val mask = batch.denseFeatures("mask").toType(ScalarType.Float)
      val logitsOrProb = forwardByModel(concepts, questions, responses, targets)
      val (selected, isLogits) = selectPredictions(logitsOrProb, targets)
      val probs = if (isLogits) selected.sigmoid() else selected
      val validPred = probs.masked_select(mask.toType(ScalarType.Bool)).to(ScalarType.Float).contiguous().cpu().toFloatArray
      val validLabel = labels.masked_select(mask.toType(ScalarType.Bool)).to(ScalarType.Float).contiguous().cpu().toFloatArray
      (validPred, validLabel)
    }

    private def forwardByModel(concepts: Tensor, questions: Tensor, responses: Tensor, targets: Tensor): Tensor = {
      modelName match {
        case "DKT" => model.asInstanceOf[DKT].forward(concepts, responses)
        case "DKTForget" => model.asInstanceOf[DKTForget].forward(concepts, responses)
        case "DKVMN" => model.asInstanceOf[DKVMN].forward(concepts, responses)
        case "DeepIRT" => model.asInstanceOf[DeepIRT].forward(concepts, responses)
        case "GKT" => model.asInstanceOf[GKT].forward(concepts, responses)
        case "AKT" => model.asInstanceOf[AKT].forward(concepts, responses)
        case "ATDKT" => model.asInstanceOf[ATDKT].forward(concepts, responses)
        case "ATKT" => model.asInstanceOf[ATKT].forward(concepts, responses)
        case "CSKT" => model.asInstanceOf[CSKT].forward(concepts, responses)
        case "DIMKT" => model.asInstanceOf[DIMKT].forward(concepts, concepts, concepts, responses)
        case "IEKT" => model.asInstanceOf[IEKT].forward(concepts, responses)
        case "LPKT" => model.asInstanceOf[LPKT].forward(questions, responses, concepts.toType(ScalarType.Float))
        case "MTKT" => model.asInstanceOf[MTKT].forward(concepts, responses)
        case "PromptKT" => model.asInstanceOf[PromptKT].forward(concepts, responses)
        case "QDKT" => model.asInstanceOf[QDKT].forward(questions, concepts, responses)
        case "RKT" => model.asInstanceOf[RKT].forward(concepts, responses)
        case "RobustKT" => model.asInstanceOf[RobustKT].forward(concepts, responses)
        case "SAINT" => model.asInstanceOf[SAINT].forward(questions, concepts, responses)
        case "SAINTPlusPlus" => model.asInstanceOf[SAINTPlusPlus].forward(questions, concepts, responses)
        case "SAKT" => model.asInstanceOf[SAKT].forward(concepts, responses, targets)
        case "SKVMN" => model.asInstanceOf[SKVMN].forward(concepts, responses)
        case "StableKT" => model.asInstanceOf[StableKT].forward(concepts, responses)
        case "UKT" => model.asInstanceOf[UKT].forward(concepts, responses)
        case other => throw new IllegalArgumentException(s"Unsupported KT model for forward: $other")
      }
    }

    private def selectPredictions(output: Tensor, targets: Tensor): (Tensor, Boolean) = {
      if (output.dim() == 3L && modelName != "GKT" && modelName != "DeepIRT") {
        val gathered = output.gather(2, targets.unsqueeze(2).toType(ScalarType.Long)).squeeze(2)
        (gathered, modelName match {
          case "DKT" | "DKTForget" => true
          case _ => false
        })
      } else {
        val squeezed = if (output.dim() == 3L) output.squeeze(2) else output
        val isLogits = modelName match {
          case "LPKT" => false
          case _ => false
        }
        (squeezed, isLogits)
      }
    }
  }
}

