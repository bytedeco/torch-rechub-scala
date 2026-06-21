package tutorials

import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.tensor
import torchrec.basic.features.SparseFeature
import torchrec.data.TensorDataset

import scala.collection.mutable
import scala.io.Source
import scala.util.Random

object AmazonBooksBeautyCtrSupport {

  case class CtrSplit(features: List[SparseFeature], train: TensorDataset, valid: TensorDataset, test: TensorDataset)

  private case class RawRow(user: Int, item: Int, time: Long)

  def load(csvPath: String, seed: Int = 2022, maxSeqLen: Int = 20): CtrSplit = {
    val src = Source.fromFile(csvPath)
    val rows = try {
      val it = src.getLines()
      val header = it.next().split(",", -1).map(_.trim)
      val col = header.zipWithIndex.toMap
      val userIdx = col("user_id")
      val itemIdx = col("item_id")
      val timeIdx = col.getOrElse("time", -1)
      it.map { line =>
        val arr = line.split(",", -1).map(_.trim)
        val user = arr(userIdx).toInt
        val item = arr(itemIdx).toInt
        val time = if (timeIdx >= 0) arr(timeIdx).toLong else 0L
        RawRow(user, item, time)
      }.toVector
    } finally {
      src.close()
    }

    val byUser = rows.groupBy(_.user).view.mapValues(_.sortBy(_.time)).toMap
    val allItems = rows.map(_.item).distinct.sorted
    val rng = new Random(seed)

    val userBuf = mutable.ArrayBuffer.empty[Float]
    val targetBuf = mutable.ArrayBuffer.empty[Float]
    val histLastBuf = mutable.ArrayBuffer.empty[Float]
    val labelBuf = mutable.ArrayBuffer.empty[Float]

    byUser.foreach { case (u, seq) =>
      if (seq.size > 1) {
        for (i <- 1 until seq.size) {
          val histStart = math.max(0, i - maxSeqLen)
          val hist = seq.slice(histStart, i)
          val histLast = hist.last.item
          val pos = seq(i).item

          userBuf += u.toFloat
          targetBuf += pos.toFloat
          histLastBuf += histLast.toFloat
          labelBuf += 1.0f

          var neg = pos
          while (neg == pos) {
            neg = allItems(rng.nextInt(allItems.size))
          }
          userBuf += u.toFloat
          targetBuf += neg.toFloat
          histLastBuf += histLast.toFloat
          labelBuf += 0.0f
        }
      }
    }

    val n = labelBuf.size
    val idx = rng.shuffle((0 until n).toList)
    val trainN = (n * 0.8).toInt
    val validN = (n * 0.1).toInt

    def build(indices: Seq[Int]): TensorDataset = {
      val uu = indices.map(i => userBuf(i)).toArray
      val tt = indices.map(i => targetBuf(i)).toArray
      val hh = indices.map(i => histLastBuf(i)).toArray
      val yy = indices.map(i => labelBuf(i)).toArray
      new TensorDataset(
        sparseFeatures = Map(
          "user_id" -> tensor(uu, Array(indices.size.toLong)).toType(ScalarType.Long),
          "target_item_id" -> tensor(tt, Array(indices.size.toLong)).toType(ScalarType.Long),
          "hist_item_last" -> tensor(hh, Array(indices.size.toLong)).toType(ScalarType.Long)
        ),
        labels = Some(tensor(yy, Array(indices.size.toLong)))
      )
    }

    val trainDs = build(idx.take(trainN))
    val validDs = build(idx.slice(trainN, trainN + validN))
    val testDs = build(idx.drop(trainN + validN))

    val maxUser = rows.map(_.user).max + 2L
    val maxItem = rows.map(_.item).max + 2L
    val features = List(
      SparseFeature("user_id", maxUser, 8),
      SparseFeature("target_item_id", maxItem, 8),
      SparseFeature("hist_item_last", maxItem, 8)
    )

    CtrSplit(features, trainDs, validDs, testDs)
  }
}

