package torchrec.distributed

import org.bytedeco.pytorch.{Module, Tensor}

import java.lang.reflect.Method

/**
 * Utility for invoking Module.forward(Tensor) via reflection.
 * Handles type erasure when Module instances are stored in ModuleListImpl
 * by allowing explicit class-based method lookup.
 *
 * IMPORTANT: This class solves the JavaCPP type erasure problem where
 * SequentialImpl/ModuleListImpl/ModuleDictImpl return Module references
 * that cannot be directly cast to subclass types (e.g., MLP).
 * Instead of casting, we use reflection to find and invoke the forward method.
 */
object ModuleForward {

  // Cache for forward methods to avoid repeated reflection lookup
  private val methodCache = scala.collection.mutable.Map[String, ModuleForward]()
  private val classCache = scala.collection.mutable.Map[Class[_], Class[_]]()

  /**
   * Find and cache the forward(Tensor) method on the module class hierarchy.
   * Uses reflection to search through the class hierarchy.
   */
  def of(module: Module): ModuleForward = {
    val className = module.getClass.getName
    methodCache.getOrElseUpdate(className, {
      var cls: Class[_] = module.getClass
      while (cls != null && cls != classOf[java.lang.Object]) try {
        val m = cls.getDeclaredMethod("forward", classOf[Tensor])
        m.setAccessible(true)
        System.out.printf("[ModuleForward] Found forward method on %s%n", cls.getName)
        return new ModuleForward(m)
      } catch {
        case e: NoSuchMethodException =>
          cls = cls.getSuperclass
      }
      throw new UnsupportedOperationException("Module " + module.getClass.getName + " has no forward(Tensor) method")
    })
  }

  /**
   * Find the forward(Tensor) method using an explicit class.
   * This avoids type erasure issues when Module instances are stored in
   * ModuleListImpl or other generic containers.
   *
   * @param module The module instance
   * @param cls The explicit class type to look up the forward method
   */
  def of(module: Module, cls: Class[_]): ModuleForward = {
    val key = s"${cls.getName}_${classOf[Tensor].getName}"
    methodCache.getOrElseUpdate(key, {
      try {
        val m = cls.getDeclaredMethod("forward", classOf[Tensor])
        m.setAccessible(true)
        System.out.printf("[ModuleForward] Found forward method on %s (explicit lookup)%n", cls.getName)
        new ModuleForward(m)
      } catch {
        case e: NoSuchMethodException =>
          throw new UnsupportedOperationException(s"Class ${cls.getName} has no forward(Tensor) method: ${e.getMessage}", e)
      }
    })
  }

  /**
   * Find the forward(Tensor) method using an explicit class name.
   * Useful when you have the class name but not the Class object.
   *
   * @param module The module instance
   * @param className The fully qualified class name to look up
   */
  def of(module: Module, className: String): ModuleForward = {
    val key = s"${className}_${classOf[Tensor].getName}"
    methodCache.getOrElseUpdate(key, {
      try {
        val cls = Class.forName(className)
        of(module, cls)
      } catch {
        case e: ClassNotFoundException =>
          throw new UnsupportedOperationException(s"Class not found: $className", e)
      }
    })
  }

  /**
   * Find the forward(Map[String, Tensor]) method using an explicit class.
   *
   * @param module The module instance
   * @param cls The explicit class type to look up the forward method
   */
  def ofMapInput(module: Module, cls: Class[_]): ModuleForward = {
    val key = s"${cls.getName}_MapStringTensor"
    methodCache.getOrElseUpdate(key, {
      try {
        val m = cls.getDeclaredMethod("forward", classOf[java.util.Map[String, Tensor]])
        m.setAccessible(true)
        System.out.printf("[ModuleForward] Found forward(Map) method on %s (explicit lookup)%n", cls.getName)
        new ModuleForward(m)
      } catch {
        case e: NoSuchMethodException =>
          throw new UnsupportedOperationException(s"Class ${cls.getName} has no forward(Map[String, Tensor]) method: ${e.getMessage}", e)
      }
    })
  }

  /**
   * Clear the method cache. Useful for testing or when modules are recreated.
   */
  def clearCache(): Unit = {
    methodCache.clear()
    classCache.clear()
  }
}

class ModuleForward private(private val forwardMethod: Method) {
  /**
   * Invoke forward via reflection.
   */
  def apply(module: Module, input: Tensor): Tensor = try {
    forwardMethod.invoke(module, input).asInstanceOf[Tensor]
  } catch {
    case e: Exception =>
      throw new RuntimeException("Failed to invoke forward: " + e.getMessage, e)
  }

  def apply(module: Module, inputMap: Map[String, Tensor]): Tensor = try {
    // Convert Scala Map to Java Map for reflection call
    val javaMap = new java.util.HashMap[String, Tensor]()
    inputMap.foreach { case (k, v) => javaMap.put(k, v) }
    forwardMethod.invoke(module, javaMap).asInstanceOf[Tensor]
  } catch {
    case e: Exception =>
      throw new RuntimeException("Failed to invoke forward: " + e.getMessage, e)
  }
}