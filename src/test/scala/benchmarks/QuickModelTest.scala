package benchmarks

import torchrec.data._
import torchrec.models.ranking._
import torchrec.trainers._
import torchrec.utils.DeviceSupport
import torchrec.Implicits.tensor
import torchrec.Implicits._
import torchrec.basic.features._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Quick test for models with synthetic data
 */
object QuickModelTest {

  def main(args: Array[String]): Unit = {
    val device = DeviceSupport.backend
    println(s"Device: $device")

    // Generate small synthetic dataset
    val rng = new scala.util.Random(42)
    val numSamples = 1000
    val numFeatures = 5

    val sparseNames = (0 until numFeatures).map(i => s"cat_$i")
    val sparseCols = sparseNames.map { name =>
      val data = Array.ofDim[Float](numSamples)
      for (j <- 0 until numSamples) {
        data(j) = rng.nextInt(100).toFloat
      }
      name -> tensor(data, Array(numSamples.toLong)).toType(ScalarType.Long)
    }.toMap

    val labels = Array.ofDim[Float](numSamples)
    for (j <- 0 until numSamples) {
      labels(j) = if (rng.nextFloat() > 0.5f) 1.0f else 0.0f
    }
    val labelsTensor = tensor(labels, Array(numSamples.toLong))

    val trainDS = new torchrec.data.TensorDataset(sparseCols, Map.empty, Some(labelsTensor))
    val testDS = new torchrec.data.TensorDataset(sparseCols.map { case (k, v) => k -> v.narrow(0, 800, 200) }, Map.empty, Some(labelsTensor.narrow(0, 800, 200)))

    val trainLoader = new DataLoader(trainDS, 64, shuffle = true)
    val testLoader = new DataLoader(testDS, 64, shuffle = false)

    // Test AFM
    println("\n=== Testing AFM ===")
    testAFM(trainLoader, testLoader, device)

    // Test AutoInt
    println("\n=== Testing AutoInt ===")
    testAutoInt(trainLoader, testLoader, device)

    // Test FiBiNet
    println("\n=== Testing FiBiNet ===")
    testFiBiNet(trainLoader, testLoader, device)

    // Test DeepFM
    println("\n=== Testing DeepFM ===")
    testDeepFM(trainLoader, testLoader, device)

    println("\n=== All tests completed ===")
  }

  def testAFM(trainLoader: DataLoader, testLoader: DataLoader, device: String): Unit = {
    try {
      val features = (0 until 5).map(i => SparseFeature(s"cat_$i", 1000, 8)).toList
      val model = new AFM(features, embedDim = 8, attentionDim = 32, dropout = 0.2f, device = device)
      val trainer = new CTRTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader, Some(testLoader))
      val metrics = trainer.evaluate(testLoader)
      println(s"AFM Metrics: AUC=${metrics.getOrElse("AUC", 0.0f)}")
    } catch {
      case e: Throwable =>
        println(s"AFM FAILED: ${e.getMessage}")
        e.printStackTrace()
    }
  }

  def testAutoInt(trainLoader: DataLoader, testLoader: DataLoader, device: String): Unit = {
    try {
      val features = (0 until 5).map(i => SparseFeature(s"cat_$i", 1000, 8)).toList
      val model = new AutoInt(features, embedDim = 8, numAttnHeads = 2, numLayers = 2, useMlp = true, device = device)
      val trainer = new CTRTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader, Some(testLoader))
      val metrics = trainer.evaluate(testLoader)
      println(s"AutoInt Metrics: AUC=${metrics.getOrElse("AUC", 0.0f)}")
    } catch {
      case e: Throwable =>
        println(s"AutoInt FAILED: ${e.getMessage}")
        e.printStackTrace()
    }
  }

  def testFiBiNet(trainLoader: DataLoader, testLoader: DataLoader, device: String): Unit = {
    try {
      val features = (0 until 5).map(i => SparseFeature(s"cat_$i", 1000, 8)).toList
      val model = new FiBiNet(features, embedDim = 8, mlpDims = List(64L, 32L), bilinearType = "field_all", device = device)
      val trainer = new CTRTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader, Some(testLoader))
      val metrics = trainer.evaluate(testLoader)
      println(s"FiBiNet Metrics: AUC=${metrics.getOrElse("AUC", 0.0f)}")
    } catch {
      case e: Throwable =>
        println(s"FiBiNet FAILED: ${e.getMessage}")
        e.printStackTrace()
    }
  }

  def testDeepFM(trainLoader: DataLoader, testLoader: DataLoader, device: String): Unit = {
    try {
      val features = (0 until 5).map(i => SparseFeature(s"cat_$i", 1000, 8)).toList
      val model = new DeepFM(
        deepFeatures = features.take(3),
        fmFeatures = features.drop(3),
        embedDim = 8,
        mlpDims = List(64L, 32L),
        dropout = 0.2f,
        device = device
      )
      val trainer = new CTRTrainer(model, 0.001f, device = device, numEpochs = 2, verbose = true)
      trainer.fit(trainLoader, Some(testLoader))
      val metrics = trainer.evaluate(testLoader)
      println(s"DeepFM Metrics: AUC=${metrics.getOrElse("AUC", 0.0f)}")
    } catch {
      case e: Throwable =>
        println(s"DeepFM FAILED: ${e.getMessage}")
        e.printStackTrace()
    }
  }
}
