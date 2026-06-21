package benchmarks

import torchrec.basic.layers._
import torchrec.utils.DeviceSupport
import torchrec.Implicits.tensor

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.util.Random

/**
 * Benchmark for individual layers to verify correctness
 */
object LayersBenchmark {

  def main(args: Array[String]): Unit = {
    println("=" * 60)
    println("Layers Benchmark Suite")
    println("=" * 60)

    val device = DeviceSupport.backend
    val results = scala.collection.mutable.ListBuffer[(String, Boolean, String)]()

    // Test each layer
    val layerTests = List(
      ("PredictionLayer", testPredictionLayer _),
      ("LR", testLR _),
      ("ConcatPooling", testConcatPooling _),
      ("AveragePooling", testAveragePooling _),
      ("SumPooling", testSumPooling _),
      ("CrossLayer", testCrossLayer _),

      ("MultiInterestSA", testMultiInterestSA _),
      ("InteractingLayer", testInteractingLayer _),
      ("MLP", testMLP _),
      ("BiLinearInteractionLayer", testBiLinearInteractionLayer _),
      ("RelativeBucketedTimeAndPositionBias", testRelativeBucketedTimeAndPositionBias _),

      ("FFM", testFFM _),

      ("HSTULayer", testHSTULayer _),
      ("HSTUBlock", testHSTUBlock _),
      ("CEN", testCEN _),
      ("CapsuleNetwork", testCapsuleNetwork _),

    )

    layerTests.foreach { case (name, testFn) =>
      try {
        val (passed, msg) = testFn(device)
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

  // Helper to create 2D tensor
  def tensor2d(arr: Array[Array[Float]], device: String = DeviceSupport.backend): Tensor = {
    val flat = arr.flatten
    tensor(flat, Array(arr.length.toLong, arr(0).length.toLong)).to(new Device(device), ScalarType.Float)
  }

  // Helper to create 3D tensor from Array[Array[Array[Float]]]
  def tensor3d(arr: Array[Array[Array[Float]]], device: String = DeviceSupport.backend): Tensor = {
    val b = arr.length
    val s = arr(0).length
    val d = arr(0)(0).length
    val flat = arr.flatten.flatten
    tensor(flat, Array(b.toLong, s.toLong, d.toLong)).to(new Device(device), ScalarType.Float)
  }


  // Helper to create 4D tensor from Array[Array[Array[Array[Float]]]]
  def tensor4d(arr: Array[Array[Array[Array[Float]]]], device: String = DeviceSupport.backend): Tensor = {
    val b = arr.length
    val f1 = arr(0).length
    val f2 = arr(0)(0).length
    val d = arr(0)(0)(0).length
    val flat = arr.flatten.flatten.flatten
    tensor(flat, Array(b.toLong, f1.toLong, f2.toLong, d.toLong)).to(new Device(device), ScalarType.Float)
  }

  def testPredictionLayer(device: String = DeviceSupport.backend): (Boolean, String) = {
    val layer = new PredictionLayer("classification")
    val input = tensor2d(Array.fill(4, 8)(1.0f))
    val out = layer.forward(input)
    val passed = out.dim() == 2L && out.size(0) == 4L && out.size(1) == 8L
    (passed, f"output shape: ${out.size(0)}x${out.size(1)}")
  }

  def testLR(device: String = DeviceSupport.backend): (Boolean, String) = {
    val layer = new LR(8, sigmoid = false, device)
    val input = tensor2d(Array.fill(4, 8)(1.0f))
    val out = layer.forward(input)
    val passed = out.dim() == 2L && out.size(0) == 4L && out.size(1) == 1L
    (passed, f"output shape: ${out.size(0)}x${out.size(1)}")
  }

  def testConcatPooling(device: String = DeviceSupport.backend): (Boolean, String) = {
    val layer = new ConcatPooling()
    val input = tensor3d(Array.fill(4, 10, 8)(1.0f))
    val out = layer.forward(input, None)
    val passed = out.dim() == 3L && out.size(0) == 4L && out.size(1) == 10L && out.size(2) == 8L
    (passed, f"output shape: ${out.size(0)}x${out.size(1)}x${out.size(2)}")
  }

  def testAveragePooling(device: String = DeviceSupport.backend): (Boolean, String) = {
    val layer = new AveragePooling()
    val input = tensor3d(Array.fill(4, 10, 8)(1.0f))
    val out = layer.forward(input, None)
    val passed = out.dim() == 2L && out.size(0) == 4L && out.size(1) == 8L
    (passed, f"output shape: ${out.size(0)}x${out.size(1)}")
  }

  def testSumPooling(device: String = DeviceSupport.backend): (Boolean, String) = {
    val layer = new SumPooling()
    val input = tensor3d(Array.fill(4, 10, 8)(1.0f))
    val out = layer.forward(input, None)
    val passed = out.dim() == 2L && out.size(0) == 4L && out.size(1) == 8L
    (passed, f"output shape: ${out.size(0)}x${out.size(1)}")
  }

  def testCrossLayer(device: String = DeviceSupport.backend): (Boolean, String) = {
    val layer = new CrossLayer(8, device)
    val x0 = tensor2d(Array.fill(4, 8)(1.0f))
    val xi = tensor2d(Array.fill(4, 8)(0.5f))
    val out = layer.forward(x0, xi)
    val passed = out.dim() == 2L && out.size(0) == 4L && out.size(1) == 8L
    (passed, f"output shape: ${out.size(0)}x${out.size(1)}")
  }

  def testBiLinearInteractionLayer(device: String = DeviceSupport.backend): (Boolean, String) = {
    val layer = new BiLinearInteractionLayer(8, 4, "field_interaction", device)
    val input = tensor3d(Array.fill(4, 4, 8)(1.0f))
    val out = layer.forward(input)
    // Output: (batch, num_interactions=6, embed_dim=8)
    val passed = out.dim() == 3L && out.size(0) == 4L && out.size(1) == 6L && out.size(2) == 8L
    (passed, f"output shape: ${out.size(0)}x${out.size(1)}x${out.size(2)}")
  }

  def testMultiInterestSA(device: String = DeviceSupport.backend): (Boolean, String) = {
    val layer = new MultiInterestSA(8, 4, None, device)
    val seqEmb = tensor3d(Array.fill(4, 10, 8)(1.0f))
    val mask = tensor3d(Array.fill(4, 10, 1)(1.0f))
    val out = layer.forward(seqEmb, Some(mask))
    val passed = out.dim() == 3L && out.size(0) == 4L && out.size(1) == 4L && out.size(2) == 8L
    (passed, f"output shape: ${out.size(0)}x${out.size(1)}x${out.size(2)}")
  }

  def testCapsuleNetwork(device: String = DeviceSupport.backend): (Boolean, String) = {
    val layer = new CapsuleNetwork(8, 10, 2, 4, 3, false, device)
    val itemEb = tensor3d(Array.fill(4, 10, 8)(1.0f))
    val mask = tensor3d(Array.fill(4, 10, 1)(1.0f))
    val out = layer.forward(itemEb, mask)
    val passed = out.dim() == 3L && out.size(0) == 4L && out.size(1) == 4L && out.size(2) == 8L
    (passed, f"output shape: ${out.size(0)}x${out.size(1)}x${out.size(2)}")
  }

  def testFFM(device: String = DeviceSupport.backend): (Boolean, String) = {
    val layer = new FFM(4, reduceSum = true, device)
    // Input: (batch=4, num_fields=4, embed_dim=8)
    val input = tensor3d(Array.fill(4, 4, 8)(1.0f))
    val out = layer.forward(input)
    // Output with reduceSum=true: (batch, num_interactions=6, embed_dim=8) -> sum along last dim -> (batch, 6)
    val passed = out.dim() == 2L && out.size(0) == 4L && out.size(1) == 6L
    (passed, f"output shape: ${out.size(0)}x${out.size(1)}")
  }

  def testCENs(device: String = DeviceSupport.backend): (Boolean, String) = {
    val layer = new CEN(8, 6, 2, device)
    // em shape [B=4, numFieldCrosses=6, embedDim=8] 完全匹配Python输入要求
    val input = tensor3d(Array.fill(4, 6, 8)(1.0f))
    val out = layer.forward(input)
    // output [4, 6*8=48]
    println(out.sizes().vec().get().mkString(","))
    val passed = out.dim() == 2L && out.size(0) == 4L && out.size(1) == 48L
    (passed, f"output shape: ${out.size(0)}x${out.size(1)} * ${out.size(2)}")
  }

  def testCEN(device: String = DeviceSupport.backend): (Boolean, String) = {
    val layer = new CEN(8, 6, 2, device)
    // Input: (batch=4, num_field_crosses=6, embed_dim=8)
    val input = tensor3d(Array.fill(4, 6, 8)(1.0f))
    val out = layer.forward(input)
    // Output: (batch, num_field_crosses * embed_dim) = (4, 48)
    val passed = out.dim() == 2L && out.size(0) == 4L && out.size(1) == 48L
    (passed, f"output shape: ${out.size(0)}x${out.size(1)}")
  }

  def testHSTULayer(device: String = DeviceSupport.backend): (Boolean, String) = {
    val layer = new HSTULayer(64, 4, 16, 16, 0.1f, 20, 128, "sqrt", 1.0f, "minutes", device)
    val input = tensor3d(Array.fill(4, 20, 64)(1.0f))
    val out = layer.forward(input, None, None)
    val passed = out.dim() == 3L && out.size(0) == 4L && out.size(1) == 20L && out.size(2) == 64L
    (passed, f"output shape: ${out.size(0)}x${out.size(1)}x${out.size(2)}")
  }

  def testHSTUBlock(device: String = DeviceSupport.backend): (Boolean, String) = {
    val layer = new HSTUBlock(64, 4, 2, 16, 16, 0.1f, 20, 128, "sqrt", 1.0f, "minutes", device)
    val input = tensor3d(Array.fill(4, 20, 64)(1.0f))
    val out = layer.forward(input, None, None)
    val passed = out.dim() == 3L && out.size(0) == 4L && out.size(1) == 20L && out.size(2) == 64L
    (passed, f"output shape: ${out.size(0)}x${out.size(1)}x${out.size(2)}")
  }

  def testInteractingLayer(device: String = DeviceSupport.backend): (Boolean, String) = {
    val layer = new InteractingLayer(8, 2, 0.0f, true, device)
    val input = tensor3d(Array.fill(4, 4, 8)(1.0f))
    val out = layer.forward(input)
    val passed = out.dim() == 3L && out.size(0) == 4L && out.size(1) == 4L && out.size(2) == 8L
    (passed, f"output shape: ${out.size(0)}x${out.size(1)}x${out.size(2)}")
  }

  def testMLP(device: String = DeviceSupport.backend): (Boolean, String) = {
    val layer = new MLP(8, List(16L, 8L), 1, "relu", 0.0f, false, false, true, device)
    val input = tensor2d(Array.fill(4, 8)(1.0f))
    val out = layer.forward(input)
    val passed = out.dim() == 2L && out.size(0) == 4L && out.size(1) == 1L
    (passed, f"output shape: ${out.size(0)}x${out.size(1)}")
  }

  def testRelativeBucketedTimeAndPositionBias(device: String = DeviceSupport.backend): (Boolean, String) = {
    val layer = new RelativeBucketedTimeAndPositionBias(4, 20, 128, "sqrt", 1.0f, "minutes")
    val out = layer.forward(None, 10)
    val passed = out.dim() == 4L && out.size(0) == 1L && out.size(1) == 4L && out.size(2) == 10L && out.size(3) == 10L
    (passed, f"output shape: ${out.size(0)}x${out.size(1)}x${out.size(2)}x${out.size(3)}")
  }
}