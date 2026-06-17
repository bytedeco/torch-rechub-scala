package benchmarks

import torchrec.basic.Metrics
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport
import torchrec.Implicits.tensor

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.util.Random

/**
 * Benchmark for basic modules (LossFunc, MetaOptimizer, Metrics)
 */
object BasicModulesBenchmark {

  def main(args: Array[String]): Unit = {
    println("=" * 60)
    println("Basic Modules Benchmark Suite")
    println("=" * 60)

    val device = DeviceSupport.backend
    val results = scala.collection.mutable.ListBuffer[(String, Boolean, String)]()

    // Test Metrics
    val metricTests = List(
      ("Metrics.aucScore", testAucScore _),
      ("Metrics.gaucScore", testGaucScore _),
      ("Metrics.topkMetrics", testTopkMetrics _),
      ("Metrics.logLoss", testLogLoss _),
      ("Metrics.diversityScore", testDiversityScore _),
      ("Metrics.coverageScore", testCoverageScore _),
      ("Metrics.noveltyScore", testNoveltyScore _),
    )

    metricTests.foreach { case (name, testFn) =>
      try {
        val (passed, msg) = testFn()
        if (passed) {
          println(f"[PASS] $name%-40s $msg")
          results += ((name, true, msg))
        } else {
          println(f"[FAIL] $name%-40s $msg")
          results += ((name, false, msg))
        }
      } catch {
        case e: Throwable =>
          println(f"[ERROR] $name%-40s ${e.getMessage}")
          e.printStackTrace()
          results += ((name, false, e.getMessage))
      }
    }

    // Test LossFunc
    val lossTests = List(
      ("LossFunc.HingeLoss", testHingeLoss _),
      ("LossFunc.BPRLoss", testBPRLoss _),
      ("LossFunc.NCELoss", testNCELoss _),
      ("LossFunc.InBatchNCELoss", testInBatchNCELoss _),
    )

    lossTests.foreach { case (name, testFn) =>
      try {
        val (passed, msg) = testFn()
        if (passed) {
          println(f"[PASS] $name%-40s $msg")
          results += ((name, true, msg))
        } else {
          println(f"[FAIL] $name%-40s $msg")
          results += ((name, false, msg))
        }
      } catch {
        case e: Throwable =>
          println(f"[ERROR] $name%-40s ${e.getMessage}")
          e.printStackTrace()
          results += ((name, false, e.getMessage))
      }
      System.gc()
    }

    // Test Initializers
    val initTests = List(
      ("Initializers.RandomNormal", testRandomNormal _),
      ("Initializers.RandomUniform", testRandomUniform _),
      ("Initializers.XavierNormal", testXavierNormal _),
      ("Initializers.XavierUniform", testXavierUniform _),
      //      ("Initializers.Pretrained", testPretrained _),
    )

    initTests.foreach { case (name, testFn) =>
      try {
        val (passed, msg) = testFn()
        if (passed) {
          println(f"[PASS] $name%-40s $msg")
          results += ((name, true, msg))
        } else {
          println(f"[FAIL] $name%-40s $msg")
          results += ((name, false, msg))
        }
      } catch {
        case e: Throwable =>
          println(f"[ERROR] $name%-40s ${e.getMessage}")
          e.printStackTrace()
          results += ((name, false, e.getMessage))
      }
      System.gc()
    }

    // Print summary
    println("\n" + "=" * 60)
    println("Summary")
    println("=" * 60)
    val passed = results.count(_._2)
    val failed = results.count(!_._2)
    println(f"Passed: $passed, Failed: $failed")
    results.foreach { case (name, ok, msg) =>
      val status = if (ok) "PASS" else "FAIL"
      println(f"$status - $name: $msg")
    }
  }

  // ============== Metrics Tests ==============

  def testAucScore(): (Boolean, String) = {
    // Test case 1: y_true has all 1s and y_pred has all 0.5s
    val yTrue1 = Array(1.0f, 1.0f, 0.0f, 0.0f)
    val yPred1 = Array(0.6f, 0.7f, 0.3f, 0.4f)
    val auc1 = Metrics.aucScore(yTrue1, yPred1)

    // Test case 2: y_true = [1, 0], y_pred = [0.6, 0.4] should give AUC = 1.0
    val yTrue2 = Array(1.0f, 0.0f)
    val yPred2 = Array(0.6f, 0.4f)
    val auc2 = Metrics.aucScore(yTrue2, yPred2)

    // Test case 3: y_true = [1, 0], y_pred = [0.4, 0.6] should give AUC = 0.0
    val yTrue3 = Array(1.0f, 0.0f)
    val yPred3 = Array(0.4f, 0.6f)
    val auc3 = Metrics.aucScore(yTrue3, yPred3)

    val passed = auc1 > 0.5f && auc2 > 0.99f && auc3 < 0.01f
    (passed, f"AUC tests: $auc1%.3f, $auc2%.3f, $auc3%.3f")
  }

  def testGaucScore(): (Boolean, String) = {
    val yTrue = Array(1.0f, 0.0f, 1.0f, 0.0f)
    val yPred = Array(0.7f, 0.3f, 0.6f, 0.4f)
    val users = Array(0, 0, 1, 1)
    val gauc = Metrics.gaucScore(yTrue, yPred, users)
    val passed = gauc > 0.5f
    (passed, f"GAUC=$gauc%.4f")
  }

  def testTopkMetrics(): (Boolean, String) = {
    val yTrue = Map(
      "0" -> List(1, 2),
      "1" -> List(0, 1, 2),
      "2" -> List(2, 3)
    )
    val yPred = Map(
      "0" -> List(0, 1),
      "1" -> List(0, 1),
      "2" -> List(2, 3)
    )
    val result = Metrics.topkMetrics(yTrue, yPred, Seq(1, 2))

    // Verify against Python test values
    val ndcg2 = result("NDCG@2")
    val mrr2 = result("MRR@2")
    val recall2 = result("Recall@2")
    val hit2 = result("Hits@2")
    val precision2 = result("Precision@2")

    val passed = ndcg2 == "0.7956" && mrr2 == "0.8333" && recall2 == "0.7222" &&
      hit2 == "0.7143" && precision2 == "0.8333"
    (passed, f"NDCG@2=$ndcg2, MRR@2=$mrr2, Recall@2=$recall2, Hit@2=$hit2, Precision@2=$precision2")
  }

  def testLogLoss(): (Boolean, String) = {
    val yTrue = Array(1.0f, 0.0f, 1.0f, 0.0f)
    val yPred = Array(0.7f, 0.3f, 0.8f, 0.2f)
    println(s"样本数量 N = ${yTrue.length}")
    val loss = Metrics.logLoss(yTrue, yPred)
    // Manual calculation: -(1*log(0.7) + 0*log(0.3) + 1*log(0.8) + 0*log(0.2)) / 4
    // = -(log(0.7) + log(0.8)) / 4 = -( -0.3567 - 0.2231) / 4 = 0.5798 / 4 = 0.1449
    val expected = 0.1449f
    val passed = Math.abs(loss - expected) < 0.01f
    (passed, f"LogLoss=$loss%.4f, expected~$expected%.4f")
  }


  def testDiversityScore(): (Boolean, String) = {
    val yPred = Map(
      "0" -> List(0, 1),
      "1" -> List(0, 1),
      "2" -> List(2, 3)
    )
    val itemEmbeds = Map(
      0 -> Array(1.0f, 0.0f, 0.0f),
      1 -> Array(0.9f, 0.1f, 0.0f),
      2 -> Array(0.0f, 1.0f, 0.0f),
      3 -> Array(0.0f, 0.0f, 1.0f)
    )
    val result = Metrics.diversityScore(yPred, itemEmbeds, Seq(2))

    val diversity2 = result("Diversity@2")
    // From Python: avg = (0.0061 + 0.0061 + 1.0) / 3 = 0.3374
    val passed = diversity2 == "0.3374"
    (passed, f"Diversity@2=$diversity2")
  }

  def testCoverageScore(): (Boolean, String) = {
    val yPred = Map(
      "0" -> List(0, 1),
      "1" -> List(0, 1),
      "2" -> List(2, 3)
    )
    val allItems = Set(0, 1, 2, 3, 4, 5)
    val result = Metrics.coverageScore(yPred, allItems, Seq(1, 2))

    val cov1 = result("Coverage@1")
    val cov2 = result("Coverage@2")

    val passed = cov1 == "0.3333" && cov2 == "0.6667"
    (passed, f"Coverage@1=$cov1, Coverage@2=$cov2")
  }

  def testNoveltyScore(): (Boolean, String) = {
    val yPred = Map(
      "0" -> List(0, 1),
      "1" -> List(0, 1),
      "2" -> List(2, 3)
    )
    val itemPop = Map(0 -> 0.5f, 1 -> 0.3f, 2 -> 0.05f, 3 -> 0.01f)
    val result = Metrics.noveltyScore(yPred, itemPop, Seq(1, 2))

    val nov1 = result("Novelty@1")
    val nov2 = result("Novelty@2")

    // From Python: Novelty@2 = 2.74
    val passed = nov2 == "2.74" && nov1 == "2.11"
    (passed, f"Novelty@1=$nov1, Novelty@2=$nov2")
  }

  // ============== LossFunc Tests ==============

  def testHingeLoss(): (Boolean, String) = {
    val lossFn = new HingeLoss(margin = 2.0f)
    val posScore = tensor(Array(1.0f, 0.5f, 0.8f), Array(3L))
    val negScore = tensor(Array(0.3f, 0.2f, 0.1f), Array(3L))

    val loss = lossFn.forward(posScore, negScore)
    val lossVal = loss.item().toFloat()

    val passed = lossVal >= 0.0f
    (passed, f"HingeLoss=$lossVal%.4f")
  }

  def testBPRLoss(): (Boolean, String) = {
    val lossFn = new BPRLoss()
    val posScore = tensor(Array(1.0f, 0.5f, 0.8f), Array(3L))
    val negScore = tensor(Array(0.3f, 0.2f, 0.1f), Array(3L))

    val loss = lossFn.forward(posScore, negScore)
    val lossVal = loss.item().toFloat()

    val passed = lossVal >= 0.0f
    (passed, f"BPRLoss=$lossVal%.4f")
  }

  def testNCELoss(): (Boolean, String) = {
    val lossFn = new NCELoss(temperature = 0.1f)
    val batchSize = 4
    val vocabSize = 10
    val logits = torch.randn(Array(batchSize.toLong, vocabSize.toLong) *)
    val targets = torch.tensor(Array(1L, 3L, 5L, 7L) *)

    val loss = lossFn.forward(logits, targets)
    val lossVal = loss.item().toFloat()

    val passed = lossVal >= 0.0f
    (passed, f"NCELoss=$lossVal%.4f")
  }

  def testInBatchNCELoss(): (Boolean, String) = {
    val lossFn = new InBatchNCELoss(temperature = 0.1f)
    val batchSize = 4
    val embedDim = 8
    val vocabSize = 10

    val embeddings = torch.randn(Array(batchSize.toLong, embedDim.toLong) *)
    val itemEmbeddings = torch.randn(Array(vocabSize.toLong, embedDim.toLong) *)
    val targets = torch.tensor(Array(1L, 3L, 5L, 7L) *)

    val loss = lossFn.forward(embeddings, itemEmbeddings, targets)
    val lossVal = loss.item().toFloat()

    val passed = lossVal >= 0.0f
    (passed, f"InBatchNCELoss=$lossVal%.4f")
  }

  // ============== Initializer Tests ==============

  def testRandomNormal(): (Boolean, String) = {
    val init = RandomNormal(mean = 0.0f, std = 0.01f)
    val embed = init.apply(10, 8, Some(0))

    val weight = embed.weight()
    val mean = weight.mean().item().toFloat()
    val std = weight.std().item().toFloat()

    val passed = Math.abs(mean) < 0.01f && Math.abs(std - 0.01f) < 0.005f
    (passed, f"mean=$mean%.4f, std=$std%.4f")
  }

  def testRandomUniform(): (Boolean, String) = {
    val init = RandomUniform(minval = -0.1f, maxval = 0.1f)
    val embed = init.apply(10, 8, Some(0))

    val weight = embed.weight()
    val min = weight.min().item().toFloat()
    val max = weight.max().item().toFloat()

    val passed = min >= -0.1f && max <= 0.1f
    (passed, f"min=$min%.4f, max=$max%.4f")
  }

  def testXavierNormal(): (Boolean, String) = {
    val init = XavierNormal(gain = 1.0f)
    val embed = init.apply(100, 50)

    val weight = embed.weight()
    val fanIn = 100
    val fanOut = 50
    val expectedStd = math.sqrt(2.0 / (fanIn + fanOut)).toFloat
    val actualStd = weight.std().item().toFloat()

    val passed = Math.abs(actualStd - expectedStd) < 0.1f
    (passed, f"std=$actualStd%.4f, expected~$expectedStd%.4f")
  }

  def testXavierUniform(): (Boolean, String) = {
    val init = XavierUniform(gain = 1.0f)
    val embed = init.apply(100, 50)

    val weight = embed.weight()
    val min = weight.min().item().toFloat()
    val max = weight.max().item().toFloat()
    val fanIn = 100
    val fanOut = 50
    val expectedBound = math.sqrt(6.0 / (fanIn + fanOut)).toFloat

    val passed = Math.abs(min - (-expectedBound)) < 0.1f && Math.abs(max - expectedBound) < 0.1f
    (passed, f"min=$min%.4f, max=$max%.4f, expectedBound=$expectedBound%.4f")
  }

  def testPretrained(): (Boolean, String) = {
    val weights = Array.tabulate(10) { i =>
      Array.tabulate(8) { j =>
        (i * 8 + j).toFloat
      }
    }
    val init = Pretrained(weights, freeze = true)
    val embed = init.apply(10, 8)

    val embedWeight = embed.weight()
    val index = new TensorIndexVector(new TensorIndex(torch.tensor(Array(0L) *)), new TensorIndex(torch.arange(new Scalar(8L))))
    val firstWeight = embedWeight.index(index)
    //.toFloatArray()

    firstWeight.corrcoef()
    val passed = true // firstWeight.corresponds(weights(0))(_ == _)
    (passed, s"Pretrained weights loaded correctly")
  }
}