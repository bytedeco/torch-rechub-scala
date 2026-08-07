package torchrec.serving

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.nn.Module
import org.bytedeco.pytorch.nn.modules._
import org.bytedeco.pytorch.nn.modules.container._
import org.bytedeco.pytorch.nn.options._
import org.bytedeco.pytorch.optim._
import org.bytedeco.pytorch.data.datasets._
import org.bytedeco.pytorch.data.options._
import org.bytedeco.pytorch.data.sampler._
import org.bytedeco.pytorch.distributed._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import torchrec.Implicits.SeqTensorRichSeq
import torchrec.Implicits.RichTensor
import torchrec.Implicits.tensor
import scala.collection.mutable

/** Base trait for vector indexers used in the retrieval stage. */
trait BaseIndexer {
  def query(embeddings: Tensor, topK: Int): (Tensor, Tensor)
}

/** Base trait for vector index builders. */
trait BaseBuilder {
  def from_embeddings(embeddings: Tensor): BaseIndexer
  def from_index_file(filePath: String): BaseIndexer
}

/** Metric type for similarity search. */
enum IndexMetric:
  case L2, IP, COSINE

// ============================================================================
// VectorIndexer — In-memory brute-force with batch matrix operations
// ============================================================================

/** In-memory brute-force vector indexer using batch matrix operations. */
class VectorIndexer(
    embeddings: Tensor,
    metric: IndexMetric = IndexMetric.L2
) extends BaseIndexer {

  private val index: Tensor = embeddings.contiguous()

  def query(embeddings: Tensor, topK: Int): (Tensor, Tensor) = {
    val q = embeddings.contiguous()
    val n = q.size(0).toInt
    val m = index.size(0).toInt

    // Batch distance computation: (n,d) vs (m,d) → (n,m)
    val distsMat: Tensor = metric match {
      case IndexMetric.L2 =>
        val x_sq = q.square.sum(1).unsqueeze(1)
        val y_sq = index.square.sum(1).unsqueeze(0)
        val xy = torch.matmul(q, index.t())
        x_sq.add(y_sq).add(xy.neg().mul(new Scalar(2.0f)))
      case IndexMetric.IP =>
        torch.neg(torch.matmul(q, index.t()))
      case IndexMetric.COSINE =>
        def normalize(t: Tensor): Tensor = {
          val sq = t.square.sum(1).unsqueeze(1)
          val norm = sq.sqrt().add(new Scalar(1e-8f))
          t.div(norm)
        }
        torch.neg(torch.matmul(normalize(q), normalize(index).t()))
    }

    // Extract topK per row
    val indicesArr = new Array[Long](n * topK)
    val distsArr = new Array[Float](n * topK)

    val distsHost = distsMat.to(ScalarType.Float).contiguous().cpu()
    val flatData = distsHost.toFloatArray

    var i = 0
    while (i < n) {
      val rowOffset = i * m
      val row = flatData.slice(rowOffset, rowOffset + m)
      val sorted = row.zipWithIndex.sortBy(_._1)
      var k = 0
      while (k < topK) {
        indicesArr(i * topK + k) = sorted(k)._2.toLong
        distsArr(i * topK + k) = sorted(k)._1
        k += 1
      }
      i += 1
    }
    distsHost.close()

    import torchrec.Implicits._
    val indicesFlatF = tensor(indicesArr.map(_.toFloat), Array((n.toLong * topK.toLong)))
    val distsFlatF = tensor(distsArr, Array((n.toLong * topK.toLong)))
    val indicesTensor = indicesFlatF.reshape(n.toLong, topK.toLong).toType(ScalarType.Long)
    val distsTensor = distsFlatF.reshape(n.toLong, topK.toLong)

    (indicesTensor, distsTensor)
  }

  def save(filePath: String): Unit = {
    throw new UnsupportedOperationException("torch.save is not available in JavaCPP bindings")
  }
}

class VectorIndexerBuilder(metric: IndexMetric = IndexMetric.L2) extends BaseBuilder {
  def from_embeddings(embeddings: Tensor): BaseIndexer = new VectorIndexer(embeddings, metric)
  def from_index_file(filePath: String): BaseIndexer = {
    val loaded = torch.load(filePath)
    new VectorIndexer(loaded.asInstanceOf[Tensor], metric)
  }
}

// ============================================================================
// FaissBuilder — IVF (Inverted File) with K-Means Clustering
// ============================================================================

enum IndexType:
  case FLAT, HNSW, IVF

/**
 * FaissIVFIndexer — inverted file index with k-means centroids.
 * Algorithm:
 *   1. K-means to cluster vectors into nlist clusters
 *   2. Build inverted lists: each cluster stores vector IDs
 *   3. Query: find nearest nprobe centroids, then brute-force within clusters
 */
class FaissIVFIndexer(
    private val vectors: Tensor,
    private val centroids: Tensor,
    private val invertedLists: Array[mutable.ArrayBuilder[Int]],
    private val nprobe: Int,
    private val metric: IndexMetric
) extends BaseIndexer {

  private val n = vectors.size(0).toInt
  private val nlist = centroids.size(0).toInt

  def query(embeddings: Tensor, topK: Int): (Tensor, Tensor) = {
    val q = embeddings.contiguous()
    val nq = q.size(0).toInt

    // Step 1: Distances to all centroids
    val centroidDists: Tensor = metric match {
      case IndexMetric.L2 =>
        val q_sq = q.square.sum(1).unsqueeze(1)
        val c_sq = centroids.square.sum(1).unsqueeze(0)
        val qc = torch.matmul(q, centroids.t())
        q_sq.add(c_sq).add(qc.neg().mul(new Scalar(2.0f)))
      case IndexMetric.IP =>
        torch.neg(torch.matmul(q, centroids.t()))
      case IndexMetric.COSINE =>
        def norm(t: Tensor): Tensor = {
          val s = t.square.sum(1).unsqueeze(1)
          t.div(s.sqrt().add(new Scalar(1e-8f)))
        }
        torch.neg(torch.matmul(norm(q), norm(centroids).t()))
    }

    // Step 2: Find nearest nprobe clusters per query
    val nprobeActual = Math.min(nprobe, nlist)
    val centroidDistsHost = centroidDists.to(ScalarType.Float).contiguous().cpu()
    val centroidFlat = centroidDistsHost.toFloatArray

    val selectedClusters = new Array[Array[Int]](nq)
    var qi = 0
    while (qi < nq) {
      val rowOffset = qi * nlist
      val row = centroidFlat.slice(rowOffset, rowOffset + nlist)
      val sorted = row.zipWithIndex.sortBy(_._1)
      selectedClusters(qi) = sorted.take(nprobeActual).map(_._2)
      qi += 1
    }
    centroidDistsHost.close()

    // Step 3: Collect candidate vector IDs
    val candidateSet = new java.util.HashSet[Int]()
    qi = 0
    while (qi < nq) {
      for (c <- selectedClusters(qi)) {
        for (idx <- invertedLists(c).result()) candidateSet.add(idx)
      }
      qi += 1
    }
    val nc = candidateSet.size()
    if (nc == 0) {
      // Fallback: empty
      val idxT = tensor(Array(0f), Array(1L, topK.toLong)).toType(ScalarType.Long)
      val distT = tensor(Array(Float.MaxValue), Array(1L, topK.toLong))
      return (idxT, distT)
    }

    val candArray = candidateSet.toArray.map(_.asInstanceOf[Int])

    // Step 4: Batch distances to candidates
    // Build candidate tensor: slice rows from vectors
    val candIndicesLong = candArray.map(_.toLong).toArray
    val candIndicesTensor = torchrec.Implicits.longTensor(candIndicesLong).to(vectors.device(), ScalarType.Long)
    val candTensor = vectors.index_select(0, candIndicesTensor)
    candIndicesTensor.close()

    val distsMat: Tensor = metric match {
      case IndexMetric.L2 =>
        val q_sq = q.square.sum(1).unsqueeze(1)
        val c_sq = candTensor.square.sum(1).unsqueeze(0)
        val qc = torch.matmul(q, candTensor.t())
        q_sq.add(c_sq).add(qc.neg().mul(new Scalar(2.0f)))
      case IndexMetric.IP =>
        torch.neg(torch.matmul(q, candTensor.t()))
      case IndexMetric.COSINE =>
        def norm(t: Tensor): Tensor = {
          val s = t.square.sum(1).unsqueeze(1)
          t.div(s.sqrt().add(new Scalar(1e-8f)))
        }
        torch.neg(torch.matmul(norm(q), norm(candTensor).t()))
    }

    // Step 5: Extract topK per query
    val indicesArr = new Array[Long](nq * topK)
    val distsArr = new Array[Float](nq * topK)

    val distsHost = distsMat.to(ScalarType.Float).contiguous().cpu()
    val flatData = distsHost.toFloatArray

    qi = 0
    while (qi < nq) {
      val rowOffset = qi * nc
      val row = flatData.slice(rowOffset, rowOffset + nc)
      val sorted = row.zipWithIndex.sortBy(_._1)
      var k = 0
      while (k < topK && k < sorted.length) {
        indicesArr(qi * topK + k) = candArray(sorted(k)._2).toLong
        distsArr(qi * topK + k) = sorted(k)._1
        k += 1
      }
      while (k < topK) {
        indicesArr(qi * topK + k) = 0L
        distsArr(qi * topK + k) = Float.MaxValue
        k += 1
      }
      qi += 1
    }
    distsHost.close()
    candTensor.close()

    val indicesFlatF = tensor(indicesArr.map(_.toFloat), Array((nq.toLong * topK.toLong)))
    val distsFlatF = tensor(distsArr, Array((nq.toLong * topK.toLong)))
    val indicesTensor = indicesFlatF.reshape(nq.toLong, topK.toLong).toType(ScalarType.Long)
    val distsTensor = distsFlatF.reshape(nq.toLong, topK.toLong)

    (indicesTensor, distsTensor)
  }
}

/** K-Means clustering using Lloyd's algorithm. */
class KMeansClustering(nClusters: Int, maxIter: Int = 25, seed: Int = 42) {
  private var centroids: Tensor = _

  def fit(vectors: Tensor): Tensor = {
    val n = vectors.size(0).toInt
    val d = vectors.size(1).toInt

    // Initialize centroids by random sampling
    val rng = new scala.util.Random(seed)
    val initIdx = rng.shuffle((0 until n).toSeq).take(nClusters).sorted
    val initIndices = torchrec.Implicits.longTensor(initIdx.map(_.toLong).toArray).to(vectors.device(), ScalarType.Long)
    centroids = vectors.index_select(0, initIndices).clone()
    initIndices.close()

    // Lloyd's algorithm
    var iter = 0
    while (iter < maxIter) {
      // Assign each point to nearest centroid using batch L2 distance
      val v_sq = vectors.square.sum(1).unsqueeze(1)
      val c_sq = centroids.square.sum(1).unsqueeze(0)
      val vc = torch.matmul(vectors, centroids.t())
      val distMat = v_sq.add(c_sq).add(vc.neg().mul(new Scalar(2.0f)))

      // Manual argmin along dim 1: find nearest cluster per vector
      val distHost = distMat.to(ScalarType.Float).contiguous().cpu()
      val flatData = distHost.toFloatArray
      val assignArr = new Array[Long](n)
      var r = 0
      while (r < n) {
        var minIdx = 0
        var minVal = Float.MaxValue
        var k = 0
        while (k < nClusters) {
          val v = flatData(r * nClusters + k)
          if (v < minVal) { minVal = v; minIdx = k }
          k += 1
        }
        assignArr(r) = minIdx.toLong
        r += 1
      }
      distHost.close()

      // Accumulate sums per cluster
      val newCentroids = torchrec.Implicits.zeros(Array(nClusters.toLong, d.toLong)).to(vectors.device(), ScalarType.Float)
      val counts = new Array[Int](nClusters)

      var i = 0
      while (i < n) {
        val c = assignArr(i).toInt
        val src = vectors.select(0, i)
        val dst = newCentroids.select(0, c)
        dst.add(src)
        src.close()
        counts(c) += 1
        i += 1
      }
      // Average
      var c = 0
      while (c < nClusters) {
        if (counts(c) > 0) {
          val target = newCentroids.select(0, c)
          target.div(new Scalar(counts(c).toFloat))
        }
        c += 1
      }

      centroids.close()
      centroids = newCentroids
      iter += 1
    }

    // Return final assignments (manual argmin along dim 1)
    val v_sq = vectors.square.sum(1).unsqueeze(1)
    val c_sq = centroids.square.sum(1).unsqueeze(0)
    val vc = torch.matmul(vectors, centroids.t())
    val distMat = v_sq.add(c_sq).add(vc.neg().mul(new Scalar(2.0f)))
    val distHost = distMat.to(ScalarType.Float).contiguous().cpu()
    val flatData = distHost.toFloatArray
    val resultArr = new Array[Long](n)
    var ri = 0
    while (ri < n) {
      var minIdx = 0
      var minVal = Float.MaxValue
      var kk = 0
      while (kk < nClusters) {
        val vv = flatData(ri * nClusters + kk)
        if (vv < minVal) { minVal = vv; minIdx = kk }
        kk += 1
      }
      resultArr(ri) = minIdx.toLong
      ri += 1
    }
    distHost.close()
    distMat.close()
    torchrec.Implicits.longTensor(resultArr)
  }

  def getCentroids: Tensor = centroids
}

/** FaissBuilder — builds IVF index with k-means clustering. */
class FaissBuilder(
    indexType: IndexType = IndexType.FLAT,
    metric: IndexMetric = IndexMetric.L2,
    m: Int = 32,
    nlist: Int = 100,
    efSearch: Option[Int] = None,
    nprobe: Option[Int] = None
) extends BaseBuilder {

  def from_embeddings(embeddings: Tensor): BaseIndexer = {
    val vecs = embeddings.contiguous()
    val n = vecs.size(0).toInt

    if (indexType == IndexType.FLAT || n < nlist) {
      return new VectorIndexer(vecs, metric)
    }

    val nprobeActual = nprobe.getOrElse(Math.max(1, nlist / 10))
    println(s"  [Faiss IVF] Clustering $n vectors into $nlist clusters...")

    val kmeans = new KMeansClustering(nClusters = nlist, maxIter = 20)
    val assignments = kmeans.fit(vecs)
    val centroids = kmeans.getCentroids

    val invertedLists = Array.fill(nlist)(mutable.ArrayBuilder.make[Int])
    val idxBuf = new Array[Long](1)
    var i = 0
    while (i < n) {
      idxBuf(0) = i.toLong
      val idxT = torchrec.Implicits.longTensor(idxBuf).to(assignments.device(), ScalarType.Long)
      val c = assignments.index_select(0, idxT).item().toInt
      idxT.close()
      invertedLists(c) += i
      i += 1
    }
    assignments.close()

    println(s"  [Faiss IVF] Built inverted lists for $nlist clusters")
    new FaissIVFIndexer(vecs, centroids, invertedLists, nprobeActual, metric)
  }

  def from_index_file(filePath: String): BaseIndexer = {
    val loaded = torch.load(filePath)
    new VectorIndexer(loaded.asInstanceOf[Tensor], metric)
  }
}

// ============================================================================
// AnnoyBuilder — Random Projection Forest (Pure Scala)
// ============================================================================

/** A node in Annoy's random projection tree. */
private class RPTreeNode(
    val indices: Array[Int],
    val left: RPTreeNode,
    val right: RPTreeNode,
    val hyperplane: Tensor,
    val offset: Float
)

/**
 * AnnoyIndexer — Random Projection Forest for approximate NN search.
 * Algorithm:
 *   1. Build nTrees binary trees by recursively splitting with random hyperplanes
 *   2. Each leaf stores a small set of vector indices
 *   3. Query: traverse all trees, collect leaf candidates, brute-force within
 */
class AnnoyIndexer(
    private val vectors: Tensor,
    private val trees: List[RPTreeNode],
    private val metric: IndexMetric
) extends BaseIndexer {

  private val n = vectors.size(0).toInt

  def query(embeddings: Tensor, topK: Int): (Tensor, Tensor) = {
    val q = embeddings.contiguous()
    val nq = q.size(0).toInt
    val effectiveSK = topK * trees.size * 2

    val indicesArr = new Array[Long](nq * topK)
    val distsArr = new Array[Float](nq * topK)

    var qi = 0
    while (qi < nq) {
      val queryVec = q.select(0, qi)

      // Collect candidates from all trees
      val candidates = new java.util.HashSet[Int]()
      for (tree <- trees) {
        collectLeaves(tree, queryVec, effectiveSK, candidates)
      }

      if (candidates.isEmpty) {
        var k = 0
        while (k < topK) { indicesArr(qi * topK + k) = 0L; distsArr(qi * topK + k) = Float.MaxValue; k += 1 }
      } else {
        val candArray = candidates.toArray.map(_.asInstanceOf[Int])
        val nc = candArray.length

        val candIndicesLong = candArray.map(_.toLong).toArray
        val candIndicesTensor = torchrec.Implicits.longTensor(candIndicesLong).to(vectors.device(), ScalarType.Long)
        val candTensor = if (nc == n) vectors else vectors.index_select(0, candIndicesTensor)
        candIndicesTensor.close()

        val dists: Tensor = metric match {
          case IndexMetric.L2 =>
            val q_sq = queryVec.square.sum()
            val c_sq = candTensor.square.sum(1)
            val qc = torch.matmul(queryVec.unsqueeze(0), candTensor.t())
            q_sq.add(c_sq).add(qc.neg().mul(new Scalar(2.0f)))
          case IndexMetric.IP =>
            val ip = torch.matmul(queryVec.unsqueeze(0), candTensor.t())
            torch.neg(ip.squeeze(0))
          case IndexMetric.COSINE =>
            def norm(t: Tensor): Tensor = {
              val s = t.square.sum(1).unsqueeze(1)
              t.div(s.sqrt().add(new Scalar(1e-8f)))
            }
            val ip = torch.matmul(norm(queryVec.unsqueeze(0)), norm(candTensor).t())
            torch.neg(ip.squeeze(0))
        }

        val distsHost = dists.to(ScalarType.Float).contiguous().cpu()
        val flatData = distsHost.toFloatArray
        val sorted = flatData.zipWithIndex.sortBy(_._1)

        val nRet = Math.min(topK, sorted.length)
        var k = 0
        while (k < nRet) {
          indicesArr(qi * topK + k) = candArray(sorted(k)._2).toLong
          distsArr(qi * topK + k) = sorted(k)._1
          k += 1
        }
        while (k < topK) {
          indicesArr(qi * topK + k) = 0L; distsArr(qi * topK + k) = Float.MaxValue; k += 1
        }

        distsHost.close()
        dists.close()
        if (nc != n) candTensor.close()
      }
      qi += 1
    }
    q.close()

    val indicesFlatF = tensor(indicesArr.map(_.toFloat), Array((nq.toLong * topK.toLong)))
    val distsFlatF = tensor(distsArr, Array((nq.toLong * topK.toLong)))
    val indicesTensor = indicesFlatF.reshape(nq.toLong, topK.toLong).toType(ScalarType.Long)
    val distsTensor = distsFlatF.reshape(nq.toLong, topK.toLong)

    (indicesTensor, distsTensor)
  }

  private def collectLeaves(
      node: RPTreeNode, query: Tensor, maxLeaves: Int, result: java.util.Set[Int]
  ): Unit = {
    if (result.size >= maxLeaves || node == null) return
    if (node.hyperplane == null) {
      for (idx <- node.indices) result.add(idx)
      return
    }
    val dot = torch.dot(query, node.hyperplane).item().toFloat()
    if (dot < node.offset) collectLeaves(node.left, query, maxLeaves, result)
    else collectLeaves(node.right, query, maxLeaves, result)
  }
}

private def buildRPTree(
    vectors: Tensor, indices: Array[Int], depth: Int, leafSize: Int, rng: scala.util.Random
): RPTreeNode = {
  if (indices.length <= leafSize || depth <= 0) {
    return new RPTreeNode(indices, null.asInstanceOf[RPTreeNode], null.asInstanceOf[RPTreeNode], null.asInstanceOf[Tensor], 0f)
  }

  val i1 = indices(rng.nextInt(indices.length))
  val i2 = indices(rng.nextInt(indices.length))
  val v1 = vectors.select(0, i1)
  val v2 = vectors.select(0, i2)
  val hyperplane = v1.sub(v2)
  hyperplane.div(new Scalar(math.sqrt(hyperplane.square.sum().item().toFloat).toFloat))

  val dot1 = torch.dot(v1, hyperplane).item().toFloat()
  val dot2 = torch.dot(v2, hyperplane).item().toFloat()
  val offset = (dot1 + dot2) / 2.0f
  v1.close(); v2.close()

  val leftArr = mutable.ArrayBuilder.make[Int]
  val rightArr = mutable.ArrayBuilder.make[Int]
  for (idx <- indices) {
    val v = vectors.select(0, idx)
    val proj = torch.dot(v, hyperplane).item().toFloat()
    if (proj < offset) leftArr += idx else rightArr += idx
    v.close()
  }
  hyperplane.close()

  val left = buildRPTree(vectors, leftArr.result(), depth - 1, leafSize, rng)
  val right = buildRPTree(vectors, rightArr.result(), depth - 1, leafSize, rng)
  new RPTreeNode(indices, left, right, null.asInstanceOf[Tensor], offset)
}

class AnnoyBuilder(
    d: Int,
    metric: IndexMetric = IndexMetric.L2,
    nTrees: Int = 10,
    searchK: Int = -1
) extends BaseBuilder {

  def from_embeddings(embeddings: Tensor): BaseIndexer = {
    val vecs = embeddings.contiguous()
    val n = vecs.size(0).toInt

    if (n <= 100) return new VectorIndexer(vecs, metric)

    val leafSize = Math.max(100, n / (nTrees * 10))
    val maxDepth = Math.ceil(Math.log(n.toDouble / leafSize) / Math.log(2)).toInt
    val rng = new scala.util.Random(42)

    println(s"  [Annoy RP-Tree] Building $nTrees trees (leaf_size=$leafSize)...")
    val trees = (0 until nTrees).map { _ =>
      buildRPTree(vecs, (0 until n).toArray, maxDepth, leafSize, rng)
    }.toList

    println(s"  [Annoy RP-Tree] Built ${trees.size} trees")
    new AnnoyIndexer(vecs, trees, metric)
  }

  def from_index_file(filePath: String): BaseIndexer = {
    val loaded = torch.load(filePath)
    new VectorIndexer(loaded.asInstanceOf[Tensor], metric)
  }
}

// ============================================================================
// MilvusBuilder — HNSW (Hierarchical Navigable Small World) (Pure Scala)
// ============================================================================

private class HNSWNode(
    val id: Int,
    val neighbors: Array[Array[Int]]
)

/**
 * HNSWIndexer — Hierarchical Navigable Small World graph.
 * Algorithm:
 *   1. Assign random level L_i to each vector (geometric distribution)
 *   2. Build layers: connect each node to nearest m neighbors at each level
 *   3. Query: greedy search from top layer down, using efSearch candidates
 */
class HNSWIndexer(
    private val vectors: Tensor,
    private val graph: Array[HNSWNode],
    private val maxLevel: Int,
    private val m: Int,
    private val efConstruction: Int,
    private val metric: IndexMetric
) extends BaseIndexer {

  private val n = vectors.size(0).toInt

  def query(embeddings: Tensor, topK: Int): (Tensor, Tensor) = {
    val q = embeddings.contiguous()
    val nq = q.size(0).toInt
    val ef = Math.max(topK * 2, m * 2)

    val indicesArr = new Array[Long](nq * topK)
    val distsArr = new Array[Float](nq * topK)

    var qi = 0
    while (qi < nq) {
      val queryVec = q.select(0, qi)
      val results = searchLayer(queryVec, ef)
      val sorted = results.sortBy(_._2)

      val nRet = Math.min(topK, sorted.length)
      var k = 0
      while (k < nRet) {
        indicesArr(qi * topK + k) = sorted(k)._1.toLong
        distsArr(qi * topK + k) = sorted(k)._2
        k += 1
      }
      while (k < topK) {
        indicesArr(qi * topK + k) = 0L; distsArr(qi * topK + k) = Float.MaxValue; k += 1
      }
      qi += 1
    }
    q.close()

    val indicesFlatF = tensor(indicesArr.map(_.toFloat), Array((nq.toLong * topK.toLong)))
    val distsFlatF = tensor(distsArr, Array((nq.toLong * topK.toLong)))
    val indicesTensor = indicesFlatF.reshape(nq.toLong, topK.toLong).toType(ScalarType.Long)
    val distsTensor = distsFlatF.reshape(nq.toLong, topK.toLong)

    (indicesTensor, distsTensor)
  }

  /**
   * Greedy descent through HNSW layers.
   * At each level: start from entry node, greedily move to the best (nearest) neighbor
   * until reaching a local minimum (current node is better than all its neighbors).
   * Then descend to the next level with the best node as the new entry point.
   * Finally, collect k-NN candidates by exploring ef neighbors from the best node at level 0.
   */
  private def searchLayer(query: Tensor, ef: Int): Array[(Int, Float)] = {
    var entryNode = 0
    var level = maxLevel - 1

    // Phase 1: Greedy descent from top level to level 1
    while (level > 0) {
      val node = graph(entryNode)
      if (level >= node.neighbors.length) { level -= 1; () }
      else {
        val neighbors = node.neighbors(level)
        var bestId = entryNode
        var bestDist = computeDist(query, entryNode)
        var changed = false
        for (nid <- neighbors) {
          val d = computeDist(query, nid)
          if (d < bestDist) { bestDist = d; bestId = nid; changed = true }
        }
        if (!changed) level -= 1  // local minimum reached, descend
        else entryNode = bestId
      }
    }

    // Phase 2: Collect ef candidates from level 0 using a PQ (best-first)
    val visited = mutable.HashSet[Int]()
    val pq = new mutable.PriorityQueue[(Int, Float)]()(Ordering.by(-_._2))
    visited.add(entryNode)
    pq.enqueue((entryNode, computeDist(query, entryNode)))
    var bestId = entryNode
    var bestDist = computeDist(query, entryNode)

    while (pq.nonEmpty) {
      val (currId, currDist) = pq.dequeue()
      if (currDist > bestDist) {}  // all remaining are worse
      else {
        val node = graph(currId)
        if (0 < node.neighbors.length) {
          for (nid <- node.neighbors(0)) {
            if (!visited.contains(nid)) {
              visited.add(nid)
              val d = computeDist(query, nid)
              if (d < bestDist) { bestDist = d; bestId = nid }
              if (pq.size < ef) pq.enqueue((nid, d))
            }
          }
        }
      }
    }

    visited.toArray.map(id => (id, computeDist(query, id)))
  }

  private def computeDist(query: Tensor, idx: Int): Float = {
    val v = vectors.select(0, idx)
    val d: Float = metric match {
      case IndexMetric.L2 =>
        val diff = query.sub(v)
        val sq = diff.square.sum().item().toFloat
        diff.close(); v.close(); sq
      case IndexMetric.IP =>
        val dot = torch.dot(query, v).item().toFloat
        v.close(); -dot
      case IndexMetric.COSINE =>
        val qn = math.sqrt(query.square.sum().item().toFloat)
        val vn = math.sqrt(v.square.sum().item().toFloat)
        val dot = torch.dot(query, v).item().toFloat
        v.close()
        if (qn > 0 && vn > 0) -(dot / (qn * vn)).toFloat else 0f
    }
    d
  }
}

class MilvusBuilder(
    d: Int,
    indexType: IndexType = IndexType.FLAT,
    metric: IndexMetric = IndexMetric.COSINE,
    m: Int = 30,
    nlist: Int = 128,
    ef: Option[Int] = None,
    nprobe: Option[Int] = None
) extends BaseBuilder {

  def from_embeddings(embeddings: Tensor): BaseIndexer = {
    val vecs = embeddings.contiguous()
    val n = vecs.size(0).toInt

    if (n <= 100) return new VectorIndexer(vecs, metric)

    val mConn = Math.min(m, n - 1)
    val efConst = Math.max(ef.getOrElse(mConn * 2), mConn)

    println(s"  [HNSW] Building: n=$n, m=$mConn, efConstruction=$efConst...")

    val rng = new scala.util.Random(42)
    val lambda = 1.0 / Math.log(mConn.toDouble)
    val nodeLevels = Array.tabulate(n) { _ =>
      val l = (-lambda * Math.log(rng.nextDouble())).toInt
      Math.min(l, Math.ceil(Math.log(n)).toInt)
    }
    val maxLevel = nodeLevels.max

    val graph = Array.tabulate(n)(i => new HNSWNode(i, Array.fill(maxLevel + 1)(Array.emptyIntArray)))

    // Build layer by layer (bottom-up)
    var level = maxLevel
    while (level >= 0) {
      val nodesAtLevel = (0 until n).filter(nodeLevels(_) >= level).toArray
      if (nodesAtLevel.length > mConn) {
        val batchIdx = torchrec.Implicits.longTensor(nodesAtLevel.map(_.toLong).toArray).to(vecs.device(), ScalarType.Long)
        val batch = vecs.index_select(0, batchIdx)
        batchIdx.close()

        val distMat: Tensor = metric match {
          case IndexMetric.L2 =>
            val b_sq = batch.square.sum(1).unsqueeze(1)
            val bb = torch.matmul(batch, batch.t())
            b_sq.add(b_sq.t()).add(bb.neg().mul(new Scalar(2.0f)))
          case IndexMetric.IP => torch.neg(torch.matmul(batch, batch.t()))
          case IndexMetric.COSINE =>
            def norm(t: Tensor): Tensor = {
              val s = t.square.sum(1).unsqueeze(1)
              t.div(s.sqrt().add(new Scalar(1e-8f)))
            }
            torch.neg(torch.matmul(norm(batch), norm(batch).t()))
        }

        nodesAtLevel.indices.foreach { bi =>
          val nnId = nodesAtLevel(bi)
          val rowDists = (0 until nodesAtLevel.length).map(j =>
            (j, distMat.select(0, bi).select(0, j).item().toFloat)
          ).filter(_._1 != bi).sortBy(_._2).take(mConn)
          graph(nnId).neighbors(level) = rowDists.map(p => nodesAtLevel(p._1)).toArray
        }

        distMat.close()
        batch.close()
      }
      level -= 1
    }

    println(s"  [HNSW] Built graph with maxLevel=$maxLevel")
    new HNSWIndexer(vecs, graph, maxLevel + 1, mConn, efConst, metric)
  }

  def from_index_file(filePath: String): BaseIndexer = {
    val loaded = torch.load(filePath)
    new VectorIndexer(loaded.asInstanceOf[Tensor], metric)
  }
}
