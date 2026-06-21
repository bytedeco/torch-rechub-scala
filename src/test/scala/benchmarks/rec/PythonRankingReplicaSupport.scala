package benchmarks.rec

import org.bytedeco.pytorch.global.torch.ScalarType
import org.bytedeco.pytorch.{Adam, AdamOptions, Module, Tensor}
import torchrec.Implicits.{RichTensor, tensor}
import torchrec.basic.features.{Feature, SequenceFeature, SparseFeature}
import torchrec.basic.losses.{BCELoss, BCEWithLogitsLoss}
import torchrec.basic.metrics.{AUC, Accuracy, LogLoss}
import torchrec.data.{DataLoader, MultiTaskDataset, SequenceDataset, TensorDataset}
import torchrec.models.ranking.{DIEN, DIN}

import scala.collection.mutable
import scala.io.Source
import scala.util.Random

object PythonRankingReplicaSupport {

  case class CTRData(
    features: List[SparseFeature],
    train: TensorDataset,
    valid: TensorDataset,
    test: TensorDataset
  )

  case class SequenceCtrData(
    features: List[SparseFeature],
    historyFeatures: List[SequenceFeature],
    train: SequenceDataset,
    valid: SequenceDataset,
    test: SequenceDataset
  )

  case class CensusMtlData(
    allFeatures: List[SparseFeature],
    userFeatures: List[SparseFeature],
    itemFeatures: List[SparseFeature],
    train: MultiTaskDataset,
    valid: MultiTaskDataset,
    test: MultiTaskDataset
  )

  case class AliExpressMtlData(
    features: List[SparseFeature],
    train: MultiTaskDataset,
    valid: MultiTaskDataset,
    test: MultiTaskDataset
  )

  case class EvalResult(auc: Float, logLoss: Float, accuracy: Float)

  private def parseCsv(path: String, maxRows: Int = Int.MaxValue): (Array[String], Vector[Array[String]]) = {
    val src = Source.fromFile(path)
    try {
      val lines = src.getLines()
      require(lines.hasNext, s"CSV is empty: $path")
      val header = lines.next().split(",", -1).map(_.trim)
      val rows = lines.take(maxRows).map(_.split(",", -1).map(_.trim)).filter(_.length == header.length).toVector
      (header, rows)
    } finally {
      src.close()
    }
  }

  private def buildSparseTensorDataset(
    encodedRows: Vector[Array[Int]],
    labels: Vector[Float],
    featureNames: List[String]
  ): TensorDataset = {
    val n = encodedRows.size
    val sparseMap = featureNames.zipWithIndex.map { case (name, idx) =>
      val arr = encodedRows.map(_(idx).toFloat).toArray
      name -> tensor(arr, Array(n.toLong)).toType(ScalarType.Long)
    }.toMap
    val labelTensor = tensor(labels.toArray, Array(n.toLong))
    new TensorDataset(sparseMap, Map.empty, Some(labelTensor))
  }

  private def buildMultiTaskDataset(
    encodedRows: Vector[Array[Int]],
    taskLabels: Map[String, Vector[Float]],
    featureNames: List[String]
  ): MultiTaskDataset = {
    val n = encodedRows.size
    val featMap = featureNames.zipWithIndex.map { case (name, idx) =>
      val arr = encodedRows.map(_(idx).toFloat).toArray
      name -> tensor(arr, Array(n.toLong)).toType(ScalarType.Long)
    }.toMap
    val taskMap = taskLabels.map { case (task, values) =>
      task -> tensor(values.toArray, Array(n.toLong))
    }
    new MultiTaskDataset(featMap, taskMap)
  }

  private def splitRandom[T](data: Vector[T], seed: Int, trainRatio: Double, validRatio: Double): (Vector[T], Vector[T], Vector[T]) = {
    val shuffled = Random(seed).shuffle(data)
    val n = shuffled.size
    val trainN = (n * trainRatio).toInt
    val validN = (n * validRatio).toInt
    val train = shuffled.take(trainN)
    val valid = shuffled.slice(trainN, trainN + validN)
    val test = shuffled.drop(trainN + validN)
    (train, valid, test)
  }

  def loadAvazu(path: String, maxRowsPerSplit: Int = 200000): CTRData = {
    val (trainHeader, trainRows) = parseCsv(s"$path/train_sample.csv", maxRowsPerSplit)
    val (_, validRows) = parseCsv(s"$path/valid_sample.csv", maxRowsPerSplit)
    val (_, testRows) = parseCsv(s"$path/test_sample.csv", maxRowsPerSplit)

    val featureNames = trainHeader.filter(_.startsWith("f")).toList
    val labelIdx = trainHeader.indexOf("label")
    require(labelIdx >= 0, "avazu label column not found")

    val featIndices = featureNames.map(name => trainHeader.indexOf(name))
    val allRows = trainRows ++ validRows ++ testRows

    val vocab = featureNames.zip(featIndices).map { pair =>
      val name: String = pair._1
      val idx: Int = pair._2
      val table = mutable.HashMap.empty[String, Int]
      var nextId = 1
      allRows.foreach { row =>
        val key = if (idx >= row.length || row(idx).isEmpty) "<NA>" else row(idx)
        if (!table.contains(key)) {
          table.update(key, nextId)
          nextId += 1
        }
      }
      name -> table.toMap
    }.toMap

    def encode(rows: Vector[Array[String]]): (Vector[Array[Int]], Vector[Float]) = {
      val enc = rows.map { row =>
        featIndices.zip(featureNames).map { pair =>
          val idx: Int = pair._1
          val name: String = pair._2
          val key = if (idx >= row.length || row(idx).isEmpty) "<NA>" else row(idx)
          vocab(name).getOrElse(key, 0)
        }.toArray
      }
      val y = rows.map(r => r(labelIdx).toFloat)
      (enc, y)
    }

    val (trainX, trainY) = encode(trainRows)
    val (validX, validY) = encode(validRows)
    val (testX, testY) = encode(testRows)

    val features = featureNames.map(name => SparseFeature(name, vocab(name).size + 2, 16))

    CTRData(
      features = features,
      train = buildSparseTensorDataset(trainX, trainY, featureNames),
      valid = buildSparseTensorDataset(validX, validY, featureNames),
      test = buildSparseTensorDataset(testX, testY, featureNames)
    )
  }

  private def bucketizeDense(raw: String): String = {
    val v = try raw.toDouble catch { case _: Throwable => 0.0 }
    if (v > 2.0) math.pow(math.log(v), 2.0).toInt.toString else (v.toInt - 2).toString
  }

  def loadCriteo(csvFile: String, maxRows: Int = 250000, seed: Int = 2022): CTRData = {
    val (header, rowsRaw) = parseCsv(csvFile, maxRows)
    val labelIdx = header.indexWhere(h => h.equalsIgnoreCase("label"))
    require(labelIdx >= 0, "criteo label column not found")

    val denseCols = header.filter(_.startsWith("I")).toList
    val sparseCols = header.filter(_.startsWith("C")).toList
    val denseIdx = denseCols.map(name => header.indexOf(name))
    val sparseIdx = sparseCols.map(name => header.indexOf(name))

    val featureNames = sparseCols ++ denseCols.map(_ + "_cat")

    val allAsMap = rowsRaw.map { row =>
      val m = mutable.Map.empty[String, String]
      sparseCols.zip(sparseIdx).foreach { pair =>
        val name: String = pair._1
        val idx: Int = pair._2
        m.update(name, if (idx >= row.length || row(idx).isEmpty) "0" else row(idx))
      }
      denseCols.zip(denseIdx).foreach { pair =>
        val name: String = pair._1
        val idx: Int = pair._2
        val value = if (idx >= row.length || row(idx).isEmpty) "0" else row(idx)
        m.update(name + "_cat", bucketizeDense(value))
      }
      (m.toMap, row(labelIdx).toFloat)
    }

    val vocab = featureNames.map { name =>
      val table = mutable.HashMap.empty[String, Int]
      var nextId = 1
      allAsMap.foreach { case (featMap, _) =>
        val key = featMap.getOrElse(name, "0")
        if (!table.contains(key)) {
          table.update(key, nextId)
          nextId += 1
        }
      }
      name -> table.toMap
    }.toMap

    val encoded = allAsMap.map { case (featMap, y) =>
      val arr = featureNames.map(name => vocab(name).getOrElse(featMap.getOrElse(name, "0"), 0)).toArray
      (arr, y)
    }

    val (trainPart, validPart, testPart) = splitRandom(encoded, seed, 0.7, 0.1)

    def unpack(part: Vector[(Array[Int], Float)]): (Vector[Array[Int]], Vector[Float]) =
      (part.map(_._1), part.map(_._2))

    val (trainX, trainY) = unpack(trainPart)
    val (validX, validY) = unpack(validPart)
    val (testX, testY) = unpack(testPart)

    val features = featureNames.map(name => SparseFeature(name, vocab(name).size + 2, 16))

    CTRData(
      features = features,
      train = buildSparseTensorDataset(trainX, trainY, featureNames),
      valid = buildSparseTensorDataset(validX, validY, featureNames),
      test = buildSparseTensorDataset(testX, testY, featureNames)
    )
  }

  def loadAmazonElectronics(csvFile: String, maxSeqLen: Int = 20, seed: Int = 2022): SequenceCtrData = {
    val (header, rows) = parseCsv(csvFile)
    val userIdx = header.indexOf("user_id")
    val itemIdx = header.indexOf("item_id")
    val cateIdx = header.indexOf("cate_id")
    val timeIdx = header.indexOf("time")
    require(userIdx >= 0 && itemIdx >= 0 && cateIdx >= 0 && timeIdx >= 0, "amazon-electronics columns missing")

    val userEnc = mutable.HashMap.empty[String, Int]
    val itemEnc = mutable.HashMap.empty[String, Int]
    val cateEnc = mutable.HashMap.empty[String, Int]

    def encode(map: mutable.HashMap[String, Int], key: String): Int = {
      map.getOrElseUpdate(key, map.size + 1)
    }

    val encodedRows = rows.map { row =>
      val user = encode(userEnc, row(userIdx))
      val item = encode(itemEnc, row(itemIdx))
      val cate = encode(cateEnc, row(cateIdx))
      val time = try row(timeIdx).toLong catch { case _: Throwable => 0L }
      (user, item, cate, time)
    }

    val itemToCate = encodedRows.groupBy(_._2).view.mapValues(_.head._3).toMap
    val byUser = encodedRows.groupBy(_._1).view.mapValues(_.sortBy(_._4)).toMap
    val rng = Random(seed)

    case class SeqSample(user: Int, targetItem: Int, targetCate: Int, histItem: Array[Int], histCate: Array[Int], label: Float)
    val samples = mutable.ArrayBuffer.empty[SeqSample]

    byUser.foreach { case (u, interactions) =>
      if (interactions.size > 1) {
        for (i <- 1 until interactions.size) {
          val hist = interactions.slice(math.max(0, i - maxSeqLen), i)
          val hItems = Array.fill(maxSeqLen)(0)
          val hCates = Array.fill(maxSeqLen)(0)
          hist.zipWithIndex.foreach { case (r, j) =>
            hItems(j) = r._2
            hCates(j) = r._3
          }

          val pos = interactions(i)
          samples += SeqSample(u, pos._2, pos._3, hItems.clone(), hCates.clone(), 1.0f)

          var negItem = pos._2
          while (negItem == pos._2) {
            negItem = rng.between(1, itemEnc.size + 1)
          }
          val negCate = itemToCate.getOrElse(negItem, 1)
          samples += SeqSample(u, negItem, negCate, hItems.clone(), hCates.clone(), 0.0f)
        }
      }
    }

    val (trainS, validS, testS) = splitRandom(samples.toVector, seed, 0.8, 0.1)

    def toSeqDataset(part: Vector[SeqSample]): SequenceDataset = {
      val n = part.size
      val user = tensor(part.map(_.user.toFloat).toArray, Array(n.toLong)).toType(ScalarType.Long)
      val targetItem = tensor(part.map(_.targetItem.toFloat).toArray, Array(n.toLong)).toType(ScalarType.Long)
      val targetCate = tensor(part.map(_.targetCate.toFloat).toArray, Array(n.toLong)).toType(ScalarType.Long)

      val histItemFlat = part.flatMap(_.histItem.map(_.toFloat)).toArray
      val histCateFlat = part.flatMap(_.histCate.map(_.toFloat)).toArray
      val histItem = tensor(histItemFlat, Array(n.toLong, maxSeqLen.toLong)).toType(ScalarType.Long)
      val histCate = tensor(histCateFlat, Array(n.toLong, maxSeqLen.toLong)).toType(ScalarType.Long)

      val labels = tensor(part.map(_.label).toArray, Array(n.toLong))

      new SequenceDataset(
        features = Map("user_id" -> user, "target_item_id" -> targetItem, "target_cate_id" -> targetCate),
        sequenceFeatures = Map("hist_item_id" -> histItem, "hist_cate_id" -> histCate),
        labels = Some(labels)
      )
    }

    val features = List(
      SparseFeature("user_id", userEnc.size + 2, 8),
      SparseFeature("target_item_id", itemEnc.size + 2, 8),
      SparseFeature("target_cate_id", cateEnc.size + 2, 8)
    )
    val historyFeatures = List(
      SequenceFeature("hist_item_id", itemEnc.size + 2, 8, pooling = "concat", sharedWith = Some("target_item_id"), maxLen = maxSeqLen, paddingIdx = 0),
      SequenceFeature("hist_cate_id", cateEnc.size + 2, 8, pooling = "concat", sharedWith = Some("target_cate_id"), maxLen = maxSeqLen, paddingIdx = 0)
    )

    SequenceCtrData(features, historyFeatures, toSeqDataset(trainS), toSeqDataset(validS), toSeqDataset(testS))
  }

  def loadCensus(path: String, maxRowsPerSplit: Int = 100000): CensusMtlData = {
    val (header, trainRows) = parseCsv(s"$path/census_income_train_sample.csv", maxRowsPerSplit)
    val (_, validRows) = parseCsv(s"$path/census_income_val_sample.csv", maxRowsPerSplit)
    val (_, testRows) = parseCsv(s"$path/census_income_test_sample.csv", maxRowsPerSplit)

    val cvrCol = "income"
    val ctrCol = "marital status"
    val cvrIdx = header.indexOf(cvrCol)
    val ctrIdx = header.indexOf(ctrCol)
    require(cvrIdx >= 0 && ctrIdx >= 0, "census label columns missing")

    val featureNames = header.filterNot(h => h == cvrCol || h == ctrCol).toList
    val featIndices = featureNames.map(name => header.indexOf(name))
    val all = trainRows ++ validRows ++ testRows

    val vocab = featureNames.zip(featIndices).map { pair =>
      val name: String = pair._1
      val idx: Int = pair._2
      val table = mutable.HashMap.empty[String, Int]
      var nextId = 1
      all.foreach { row =>
        val key = if (idx >= row.length || row(idx).isEmpty) "0" else row(idx)
        if (!table.contains(key)) {
          table.update(key, nextId)
          nextId += 1
        }
      }
      name -> table.toMap
    }.toMap

    def encode(rows: Vector[Array[String]]): (Vector[Array[Int]], Vector[Float], Vector[Float], Vector[Float]) = {
      val feats = rows.map { row =>
        featIndices.zip(featureNames).map { case (idx, name) =>
          vocab(name).getOrElse(if (idx >= row.length || row(idx).isEmpty) "0" else row(idx), 0)
        }.toArray
      }
      val cvr = rows.map(r => r(cvrIdx).toFloat)
      val ctr = rows.map(r => r(ctrIdx).toFloat)
      val ctcvr = cvr.zip(ctr).map { case (a, b) => a * b }
      (feats, cvr, ctr, ctcvr)
    }

    val userCols = Set("industry code", "occupation code", "race", "education", "sex")

    def toDataset(rows: Vector[Array[String]]): MultiTaskDataset = {
      val (x, cvr, ctr, ctcvr) = encode(rows)
      buildMultiTaskDataset(x, Map("cvr_label" -> cvr, "ctr_label" -> ctr, "ctcvr_label" -> ctcvr), featureNames)
    }

    val allFeatures = featureNames.map(name => SparseFeature(name, vocab(name).size + 2, 4))
    val userFeatures = allFeatures.filter(f => userCols.contains(f.name))
    val itemFeatures = allFeatures.filterNot(f => userCols.contains(f.name))

    CensusMtlData(allFeatures, userFeatures, itemFeatures, toDataset(trainRows), toDataset(validRows), toDataset(testRows))
  }

  def loadAliExpress(path: String, maxTrain: Int = 200000, maxTest: Int = 100000): AliExpressMtlData = {
    val (header, trainRows) = parseCsv(s"$path/aliexpress_train_sample.csv", maxTrain)
    val (_, testRows) = parseCsv(s"$path/aliexpress_test_sample.csv", maxTest)

    val featureNames = header.filter(h => h.startsWith("categorical") || h.startsWith("numerical")).toList
    val featIndices = featureNames.map(name => header.indexOf(name))
    val clickIdx = header.indexOf("click")
    val convIdx = header.indexOf("conversion")
    require(clickIdx >= 0 && convIdx >= 0, "aliexpress label columns missing")

    val all = trainRows ++ testRows
    val vocab = featureNames.zip(featIndices).map { pair =>
      val name: String = pair._1
      val idx: Int = pair._2
      val table = mutable.HashMap.empty[String, Int]
      var nextId = 1
      all.foreach { row =>
        val key = if (idx >= row.length || row(idx).isEmpty) "0" else row(idx)
        if (!table.contains(key)) {
          table.update(key, nextId)
          nextId += 1
        }
      }
      name -> table.toMap
    }.toMap

    def encode(rows: Vector[Array[String]]): (Vector[Array[Int]], Vector[Float], Vector[Float]) = {
      val x = rows.map { row =>
        featIndices.zip(featureNames).map { case (idx, name) =>
          vocab(name).getOrElse(if (idx >= row.length || row(idx).isEmpty) "0" else row(idx), 0)
        }.toArray
      }
      val conversion = rows.map(r => r(convIdx).toFloat)
      val click = rows.map(r => r(clickIdx).toFloat)
      (x, conversion, click)
    }

    val (trainX, trainConv, trainClick) = encode(trainRows)
    val (testX, testConv, testClick) = encode(testRows)

    val features = featureNames.map(name => SparseFeature(name, vocab(name).size + 2, 5))
    val trainDs = buildMultiTaskDataset(trainX, Map("conversion" -> trainConv, "click" -> trainClick), featureNames)
    val testDs = buildMultiTaskDataset(testX, Map("conversion" -> testConv, "click" -> testClick), featureNames)

    AliExpressMtlData(features, trainDs, testDs, testDs)
  }

  def trainAndEvalDIN(
    model: DIN,
    trainLoader: DataLoader,
    validLoader: DataLoader,
    testLoader: DataLoader,
    epochs: Int,
    learningRate: Float
  ): EvalResult = {
    val lossFn = new BCEWithLogitsLoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(learningRate.toDouble))

    for (_ <- 0 until epochs) {
      model.train(true)
      val it = trainLoader.iterator
      while (it.hasNext) {
        val batch = it.next()
        batch.labels.foreach { labels =>
          optimizer.zero_grad()
          val feats = batch.sparseFeatures
          val seq = batch.sequenceFeatures
          val targetIdx = feats("target_item_id").view(labels.size(0), 1).toType(ScalarType.Long)
          val logits = model.forward(feats, seq, targetIdx)
          val y = labels.view(labels.size(0), 1).toType(ScalarType.Float)
          val loss = lossFn.apply(logits, y)
          loss.backward()
          optimizer.step()
        }
      }
      evaluateSequenceCtr(model, validLoader, isDien = false)
    }

    evaluateSequenceCtr(model, testLoader, isDien = false)
  }

  def trainAndEvalDIEN(
    model: DIEN,
    trainLoader: DataLoader,
    validLoader: DataLoader,
    testLoader: DataLoader,
    epochs: Int,
    learningRate: Float
  ): EvalResult = {
    val lossFn = new BCELoss()
    val optimizer = new Adam(model.parameters(), new AdamOptions(learningRate.toDouble))

    for (_ <- 0 until epochs) {
      model.train(true)
      val it = trainLoader.iterator
      while (it.hasNext) {
        val batch = it.next()
        batch.labels.foreach { labels =>
          optimizer.zero_grad()
          val probs = model.forward(batch.sparseFeatures, batch.sequenceFeatures)
          val y = labels.view(labels.size(0)).toType(ScalarType.Float)
          val loss = lossFn.apply(probs, y)
          loss.backward()
          optimizer.step()
        }
      }
      evaluateSequenceCtr(model, validLoader, isDien = true)
    }

    evaluateSequenceCtr(model, testLoader, isDien = true)
  }

  private def evaluateSequenceCtr(model: Module, loader: DataLoader, isDien: Boolean): EvalResult = {
    val preds = mutable.ArrayBuffer.empty[Float]
    val labels = mutable.ArrayBuffer.empty[Float]
    model.train(false)

    val it = loader.iterator
    while (it.hasNext) {
      val batch = it.next()
      batch.labels.foreach { y =>
        val p = if (isDien) {
          model.asInstanceOf[DIEN].forward(batch.sparseFeatures, batch.sequenceFeatures)
        } else {
          val t = batch.sparseFeatures("target_item_id").view(y.size(0), 1).toType(ScalarType.Long)
          model.asInstanceOf[DIN].forward(batch.sparseFeatures, batch.sequenceFeatures, t).sigmoid().squeeze(1)
        }
        val pHost = p.contiguous().cpu().toType(ScalarType.Float)
        val yHost = y.contiguous().cpu().toType(ScalarType.Float)
        preds.appendAll(pHost.nn.toFloatArray)
        labels.appendAll(yHost.nn.toFloatArray)
      }
    }

    val auc = new AUC()
    val ll = new LogLoss()
    val acc = new Accuracy()
    auc.update(preds.toArray, labels.toArray)
    ll.update(preds.toArray, labels.toArray)
    acc.update(preds.toArray, labels.toArray)
    EvalResult(auc.compute(), ll.compute(), acc.compute())
  }

  def parseArgs(args: Array[String]): Map[String, String] = {
    val out = mutable.Map.empty[String, String]
    var i = 0
    while (i < args.length) {
      val key = args(i)
      if (key.startsWith("--") && i + 1 < args.length) {
        out.update(key.stripPrefix("--"), args(i + 1))
        i += 2
      } else {
        i += 1
      }
    }
    out.toMap
  }
}

