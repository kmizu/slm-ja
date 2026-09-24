package slm

/** 重み減衰をかける区間（Layout の重み行列）。埋め込み・LayerNorm・バイアスには掛けない。 */
final class DecayRanges(val ranges: Array[(Int, Int)]):
  /** [start, end) を「減衰する/しない」の小区間に分けて f(s, e, decay) を呼ぶ。 */
  def foreachSegment(start: Int, end: Int)(f: (Int, Int, Boolean) => Unit): Unit =
    var cursor = start
    var k = 0
    while k < ranges.length && ranges(k)._2 <= start do k += 1
    while cursor < end do
      if k < ranges.length && ranges(k)._1 < end then
        val (rs, re) = ranges(k)
        if rs > cursor then f(cursor, rs, false)
        val segEnd = math.min(re, end)
        f(math.max(rs, cursor), segEnd, true)
        cursor = segEnd
        k += 1
      else
        f(cursor, end, false)
        cursor = end

object DecayRanges:
  def of(layout: Layout): DecayRanges =
    val cfg = layout.cfg
    val d = cfg.d
    val r = layout.layer.flatMap(l => Vector((l.wq, d * d), (l.wk, d * d), (l.wv, d * d), (l.wo, d * d), (l.w1, cfg.ff * d), (l.w2, d * cfg.ff)))
      .map((at, n) => (at, at + n)).sortBy(_._1)
    new DecayRanges(r.toArray)

/** AdamW（Float32 のパラメータ、状態も Float32）。
  *
  * `beginStep` で一度だけ step を進めて共通係数を作り、`updateRange` をパラメータ区間ごとに（並列に）呼ぶ。
  * clip は grad を書き戻さず、読み込み時に clipScale を掛ける。
  */
final class AdamW(val size: Int, val beta1: Double = 0.9, val beta2: Double = 0.95, val eps: Double = 1e-8):
  val m: Array[Float] = new Array[Float](size)
  val v: Array[Float] = new Array[Float](size)
  var step: Long = 0L

  final case class StepCoefficients(lr: Double, c1: Double, c2: Double, clipScale: Float, weightDecay: Double)

  def beginStep(lr: Double, clipScale: Double, weightDecay: Double): StepCoefficients =
    step += 1
    StepCoefficients(lr, 1.0 / (1.0 - math.pow(beta1, step.toDouble)), 1.0 / (1.0 - math.pow(beta2, step.toDouble)), clipScale.toFloat, weightDecay)

  /** [start, end) を更新する。丸めの位置は旧 update と同じ（m,v は Float、update は Double で計算して Float へ）。 */
  def updateRange(p: Array[Float], grad: Array[Float], start: Int, end: Int, k: StepCoefficients, decay: DecayRanges): Unit =
    val b1 = beta1.toFloat; val b2 = beta2.toFloat
    decay.foreachSegment(start, end) { (s, e, decayed) =>
      val wd = if decayed then k.weightDecay else 0.0
      var i = s
      while i < e do
        val gi = grad(i) * k.clipScale
        m(i) = b1 * m(i) + (1 - b1) * gi
        v(i) = b2 * v(i) + (1 - b2) * gi * gi
        val update = (m(i) * k.c1) / (math.sqrt(v(i) * k.c2) + eps)
        p(i) = (p(i) - k.lr * (update + wd * p(i))).toFloat
        i += 1
    }

object Optimizer:

  /** 勾配の二乗和（区間）。Double で返す。 */
  def sumSquares(g: Array[Float], start: Int, end: Int): Double =
    var s = 0.0
    var i = start
    while i < end do { s += g(i).toDouble * g(i); i += 1 }
    s

  /** worker ごとの勾配を区間 [start, end) について固定順に集約し、grad に書き切る。返り値は区間の二乗和。非有限値は例外。 */
  def reduceRange(buffers: Array[Array[Float]], grad: Array[Float], start: Int, end: Int, invBatch: Float): Double =
    var sumSq = 0.0
    var i = start
    while i < end do
      var acc = 0.0f
      var w = 0
      while w < buffers.length do
        acc += buffers(w)(i) * invBatch
        w += 1
      if !java.lang.Float.isFinite(acc) then throw new ArithmeticException(s"勾配に非有限値: index=$i value=$acc")
      grad(i) = acc
      sumSq += acc.toDouble * acc.toDouble
      i += 1
    sumSq

  def clipScale(norm: Double, clip: Double): Double = if norm > clip then clip / norm else 1.0
