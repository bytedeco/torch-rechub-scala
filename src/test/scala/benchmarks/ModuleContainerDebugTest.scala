package benchmarks

import org.bytedeco.javacpp.Pointer
import torchrec.basic.layers.*
import torchrec.distributed.*
import torchrec.utils.DeviceSupport
import torchrec.Implicits.tensor
import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import torchrec.Implicits.tensor
import org.bytedeco.javacpp.annotation.Cast
/**
 * Test to diagnose and demonstrate the type erasure problem with
 * SequentialImpl/ModuleListImpl/ModuleDictImpl.
 */
object ModuleContainerDebugTest {

  def safeCastLinear(mod: Module): LinearImpl = {
    // 核心：用Pointer重新构造子类包装，不依赖JVM类型转换
    new LinearImpl(mod)
  }

  def safeCastMlp(mod: Module): MLP = {
    // 核心：用Pointer重新构造子类包装，不依赖JVM类型转换
//    new LinearImpl(mod)
    mod.asInstanceOf[MLP]
  }

//  def safeCastLinear(mod: Module): LinearImpl = {
//    val rawPtr: Pointer = mod.poi
//    // 显式限定参数类型为Pointer，强制走 Pointer 单参构造，避开 LinearOptions 重载
//    new LinearImpl(rawPtr)
//  }
  def main(args: Array[String]): Unit = {

    val seq = new SequentialImpl()
    seq.push_back(new LinearImpl(8, 1116))
    seq.push_back(new ReLUImpl())



    val module: Module = seq.get(0)
    val linear: LinearImpl = safeCastLinear(module)

    println(linear.weight())
    println(s"转型成功，无ClassCastException ${linear.options().in_features().get} -> ${linear.options().out_features().get}")

    println("=" * 60)
    println("Module Container Type Erasure Debug Test")
    println("=" * 60)

    val moduleList = new ModuleListImpl()
    val mlp = new MLP(8, List(16L, 8L), 1, "relu", 0.0f, false, false, true, DeviceSupport.backend)
    moduleList.push_back(new ReLUImpl())
    moduleList.push_back(mlp)
    val mlpcast = safeCastMlp(moduleList.get(1))
    println(s"mlpcast 转型成功，无ClassCastException ${mlpcast} -> ${mlpcast.is_training()}")

    println("\n[Test 1] Direct MLP forward - Should work")
    testDirectMLP()

    println("\n[Test 2] MLP with SequentialImpl internal - Should work")
    testMLPWithSequential()

    println("\n[Test 3] Iterate SequentialImpl elements")
    testSequentialIteration()

    println("\n[Test 4] Cast attempt (will fail - demonstrating the problem)")
    testCastAttempt()

    println("\n[Test 5] Using ModuleForward reflection (the solution)")
    testModuleForwardReflection()

    println("\n[Test 6] Using ModuleContainerUtils")
    testModuleContainerUtils()
  }

  def testDirectMLP(): Unit = {
    try {
      val mlp = new MLP(8, List(16L, 8L), 1, "relu", 0.0f, false, false, true, DeviceSupport.backend)
      val input = tensor2d(Array.fill(4, 8)(1.0f))

      println(s"  MLP class: ${mlp.getClass.getName}")
      println(s"  MLP direct forward: ${mlp.forward(input).dim()}")
      println("  [PASS] Direct MLP works")
    } catch {
      case e: Exception =>
        println(s"  [FAIL] ${e.getMessage}")
        e.printStackTrace()
    }
  }

  def testMLPWithSequential(): Unit = {
    try {
      val mlp = new MLP(8, List(16L, 8L), 1, "relu", 0.0f, false, false, true, DeviceSupport.backend)
      val input = tensor2d(Array.fill(4, 8)(1.0f))

      // MLP.forward() internally calls sequential.forward(x)
      val output = mlp.forward(input)
      println(s"  MLP internal sequential forward: ${output.dim()}")
      println("  [PASS] MLP with internal SequentialImpl works")
    } catch {
      case e: Exception =>
        println(s"  [FAIL] ${e.getMessage}")
        e.printStackTrace()
    }
  }

  def testSequentialIteration(): Unit = {
    try {
      // Create a SequentialImpl with some modules
      val seq = new SequentialImpl()
      seq.push_back(new LinearImpl(8, 16))
      seq.push_back(new ReLUImpl())
      seq.push_back(new LinearImpl(16, 8))

      val size = seq.size().toInt
      println(s"  SequentialImpl size: $size")

      var i = 0
      while (i < size) {
        val module = seq.get(i)
        println(s"  [$i] module class: ${module.getClass.getName}")
        println(s"  [$i] module superclass: ${module.getClass.getSuperclass.getName}")

        // This is what ModuleContainerUtils does:
        val forward = ModuleForward.of(module.asInstanceOf[Module])
        println(s"  [$i] forward method found: ${forward != null}")
        i += 1
      }
      println("  [PASS] SequentialImpl iteration works")
    } catch {
      case e: Exception =>
        println(s"  [FAIL] ${e.getMessage}")
        e.printStackTrace()
    }
  }

  def testCastAttempt(): Unit = {
    try {
      val seq = new SequentialImpl()
      seq.push_back(new LinearImpl(8, 16))
      seq.push_back(new ReLUImpl())

      val module = seq.get(0)
      println(s"  Module class: ${module.getClass.getName}")

      // This WILL FAIL if you try to cast to a different class loader type
      // Uncomment to see the failure:
      // val linear = module.asInstanceOf[LinearImpl]

      // But JavaCPP types should work:
      val linear = module.asInstanceOf[LinearImpl]
      println(s"  Cast to LinearImpl: ${linear != null}")
      println("  [PASS] Cast to same JavaCPP type works")
    } catch {
      case e: Exception =>
        println(s"  [FAIL] ${e.getMessage}")
        e.printStackTrace()
    }
  }

  def testModuleForwardReflection(): Unit = {
    try {
      val seq = new SequentialImpl()
      seq.push_back(new LinearImpl(8, 16))
      seq.push_back(new ReLUImpl())
      seq.push_back(new LinearImpl(16, 8))

      val input = tensor2d(Array.fill(4, 8)(1.0f))//, Array(4L, 8L))

      // Use ModuleForward to invoke forward via reflection
      val results = ModuleContainerUtils.iterateSequential(seq, input)
      println(s"  Number of forward calls: ${results.length}")
      results.foreach { r => println(s"    output dim: ${r.dim()}") }
      println("  [PASS] ModuleForward reflection works")
    } catch {
      case e: Exception =>
        println(s"  [FAIL] ${e.getMessage}")
        e.printStackTrace()
    }
  }

  def testModuleContainerUtils(): Unit = {
    try {
      val seq = new SequentialImpl()
      seq.push_back(new LinearImpl(8, 16))
      seq.push_back(new ReLUImpl())
      seq.push_back(new LinearImpl(16, 1))

      val input = tensor2d(Array.fill(4, 8)(1.0f))

      // Use ModuleContainerUtils.forwardSequential for chained forward
      val output = ModuleContainerUtils.forwardSequential(seq, input)
      println(s"  Sequential forward output dim: ${output.dim()}")
      println("  [PASS] ModuleContainerUtils.forwardSequential works")
    } catch {
      case e: Exception =>
        println(s"  [FAIL] ${e.getMessage}")
        e.printStackTrace()
    }
  }

  // Helper to create 2D tensor
  def tensor2d(arr: Array[Array[Float]]): Tensor = {
    val flat = arr.flatten
    tensor(flat, Array(arr.length.toLong, arr(0).length.toLong))
  }
}
