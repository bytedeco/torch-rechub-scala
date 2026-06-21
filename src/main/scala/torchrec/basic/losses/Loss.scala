package torchrec.basic.losses

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.immutable.Seq

import torchrec.Implicits.SeqTensorRichSeq
import torchrec.Implicits.RichTensor

class BCELoss(
  reduction: String = "mean",
  posWeight: Option[Tensor] = None
) {
  def apply(predictions: Tensor, targets: Tensor): Tensor = {
    val dev = predictions.device()
    val eps: Float = 1e-7f
    // torch.clamp with scalar args preserves the requires_grad graph correctly
    val clipped = predictions.clamp(new ScalarOptional(new Scalar(eps)) , new ScalarOptional(new Scalar(1.0f - eps)))
    val logClipped = clipped.log()
    val oneMinusClipped = predictions.neg().add(new Scalar(1.0f)).clamp(new ScalarOptional(new Scalar(eps)), new ScalarOptional(new Scalar(1.0f - eps))).log()
    val loss = targets.mul(logClipped.neg())
      .add(targets.neg().add(new Scalar(1.0f)).mul(oneMinusClipped.neg()))

    reduction match {
      case "sum" => loss.sum()
      case "none" => loss
      case _ => loss.mean()
    }
  }

  def apply(predictions: Array[Float], targets: Array[Float]): Float = {
    val predPtr = new org.bytedeco.javacpp.FloatPointer()
    var i = 0
    while (i < predictions.length) { predPtr.put(i, predictions(i)); i += 1 }
    val targetPtr = new org.bytedeco.javacpp.FloatPointer()
    i = 0
    while (i < targets.length) { targetPtr.put(i, targets(i)); i += 1 }
    val sizes = Array(predictions.length.toLong)
    val opts = new TensorOptions()
    opts.dtype().put(ScalarType.Float)
    val predTensor = torch.from_blob(predPtr, sizes, null.asInstanceOf[org.bytedeco.pytorch.PointerConsumer], opts).clone()
    val targetTensor = torch.from_blob(targetPtr, sizes, null.asInstanceOf[org.bytedeco.pytorch.PointerConsumer], opts).clone()
    apply(predTensor, targetTensor).item().toFloat
  }
}

class BCEWithLogitsLoss(reduction: String = "mean") {
  def apply(predictions: Tensor, targets: Tensor): Tensor = {
    val dev = predictions.device()
    val targetsOnDev = if (!targets.device().equals(dev)) targets.to(dev, targets.dtype()) else targets

    // Be permissive about (batch,) vs (batch,1) shapes — adapt targets/predictions so they match
    val rawLoss = {
      val p = predictions
      val t = targetsOnDev
      if (p.dim() == 1 && t.dim() == 2 && t.size(1) == 1) {
        // predictions: (batch,), targets: (batch,1) -> squeeze targets
        torch.binary_cross_entropy_with_logits(p, t.squeeze(1))
      } else if (p.dim() == 2 && p.size(1) == 1 && t.dim() == 1) {
        // predictions: (batch,1), targets: (batch,) -> squeeze predictions
        torch.binary_cross_entropy_with_logits(p.squeeze(1), t)
      } else {
        torch.binary_cross_entropy_with_logits(p, t)
      }
    }
    rawLoss match {
      case l if reduction == "sum" => l.sum()
      case l if reduction == "none" => l
      case l => l.mean()
    }
  }
}

class CrossEntropyLoss(
  reduction: String = "mean",
  labelSmoothing: Float = 0.0f
) {
  def apply(predictions: Tensor, targets: Tensor): Tensor = {
    val dev = predictions.device()
    val targetsOnDev = if (!targets.device().equals(dev)) targets.to(dev, targets.dtype()) else targets
    val loss = torch.cross_entropy(predictions, targetsOnDev)
    reduction match {
      case "sum" => loss.sum()
      case "none" => loss
      case _ => loss.mean()
    }
  }
}

class BPRLoss(margin: Float = 1.0f) {
  def apply(posScores: Tensor, negScores: Tensor): Tensor = {
    val diff = posScores.sub(negScores)
    val losses = diff.sigmoid().log().neg()
    losses.mean()
  }

  def apply(posScore: Float, negScore: Float): Float = {
    val diff = posScore - negScore
    val sigmoid = 1.0f / (1.0f + scala.math.exp(-diff).toFloat)
    -scala.math.log(sigmoid + 1e-8).toFloat
  }
}

class HingeLoss(margin: Float = 1.0f) {
  def apply(posScores: Tensor, negScores: Tensor): Tensor = {
    val diff = posScores.sub(negScores)
    val dev = diff.device()
    val mTensor = torch.full(Array(1L), new Scalar(margin)).to(dev, ScalarType.Float)
    val zeroTensor = torch.full(Array(1L), new Scalar(0.0f)).to(dev, ScalarType.Float)
    val losses = diff.neg().add(mTensor).maximum(zeroTensor)
    losses.mean()
  }
}

class TripletMarginLoss(margin: Float = 1.0f) {
  def apply(anchor: Tensor, positive: Tensor, negative: Tensor): Tensor = {
    val distPos = anchor.sub(positive).pow(new Scalar(2.0f)).sum(1)
    val distNeg = anchor.sub(negative).pow(new Scalar(2.0f)).sum(1)
    val dev = distPos.device()
    val mTensor = torch.full(Array(1L), new Scalar(margin)).to(dev, ScalarType.Float)
    val zeroTensor = torch.full(Array(1L), new Scalar(0.0f)).to(dev, ScalarType.Float)
    val losses = distPos.sub(distNeg).add(mTensor).maximum(zeroTensor)
    losses.mean()
  }
}

class InBatchNCELoss(temperature: Float = 0.07f) {
  def apply(userEmbeds: Tensor, posItemEmbeds: Tensor, negItemEmbeds: Tensor): Tensor = {
    val posScores = userEmbeds.mul(posItemEmbeds).sum(1)
    val negScores = torch.bmm(
      negItemEmbeds,
      userEmbeds.unsqueeze(2)
    ).squeeze(2)
    val allScores = torch.cat(Seq(posScores.unsqueeze(1), negScores).toTensorVector, 1)
    val scaledScores = allScores.div(new Scalar(temperature))
    val labels = torch.zeros(userEmbeds.size(0).toLong).toType(ScalarType.Long).to(userEmbeds.device(), ScalarType.Long)
    torch.cross_entropy(scaledScores, labels)
  }
}

class MaskedCrossEntropyLoss {
  def apply(logits: Tensor, targets: Tensor, mask: Tensor): Tensor = {
    val dev = logits.device()
    val targetsOnDev = targets.to(dev, ScalarType.Long)
    val loss = torch.cross_entropy(logits, targetsOnDev)
    val maskOnDev = if (!mask.device().equals(dev)) mask.to(dev, mask.dtype()) else mask
    val maskedLoss = loss.mul(maskOnDev)
    maskedLoss.sum().div(maskOnDev.sum())
  }
}

class FocalLoss(
  alpha: Float = 0.25f,
  gamma: Float = 2.0f
) {
  def apply(predictions: Tensor, targets: Tensor): Tensor = {
    val dev = predictions.device()
    val targetsOnDev = if (!targets.device().equals(dev)) targets.to(dev, targets.dtype()) else targets
    val p = predictions.sigmoid()
    val ceLoss = torch.binary_cross_entropy(predictions, targetsOnDev)
    val one = new Scalar(1.0f)
    val pTerm = p.mul(targets).add(p.neg().add(one).mul(targets.neg().add(one)))
    val focalWeight = pTerm.neg().add(one).pow(new Scalar(gamma.toDouble))
    val alphaWeight = targets.mul(new Scalar(alpha.toDouble)).add(targets.neg().add(one).mul(new Scalar((1.0 - alpha).toDouble)))
    alphaWeight.mul(focalWeight).mul(ceLoss).mean()
  }
}
