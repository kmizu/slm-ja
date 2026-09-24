package slm

/** AdamW（Float32 のパラメータ、状態も Float32）。パラメータ配列を直接更新する。 */
final class AdamW(size: Int, val beta1: Double = 0.9, val beta2: Double = 0.95, val eps: Double = 1e-8):
  private val m = new Array[Float](size)
  private val v = new Array[Float](size)
  var step: Int = 0

  /** `decayMask(i)` が真のパラメータにだけ重み減衰をかける。 */
  def update(p: Array[Float], g: Array[Float], lr: Double, weightDecay: Double, decayMask: Array[Boolean]): Unit =
    step += 1
    val c1 = 1.0 / (1.0 - math.pow(beta1, step))
    val c2 = 1.0 / (1.0 - math.pow(beta2, step))
    val b1 = beta1.toFloat; val b2 = beta2.toFloat
    var i = 0
    while i < size do
      val gi = g(i)
      m(i) = b1 * m(i) + (1 - b1) * gi
      v(i) = b2 * v(i) + (1 - b2) * gi * gi
      val update = (m(i) * c1) / (math.sqrt(v(i) * c2) + eps)
      val decay = if decayMask(i) then weightDecay * p(i) else 0.0
      p(i) = (p(i) - lr * (update + decay)).toFloat
      i += 1

object Optimizer:

  /** 勾配の全体ノルムを clip 以下に抑える。返り値は元のノルム。 */
  def clipGlobalNorm(g: Array[Float], clip: Double): Double =
    var s = 0.0
    var i = 0
    while i < g.length do { s += g(i).toDouble * g(i); i += 1 }
    val norm = math.sqrt(s)
    if norm > clip then
      val f = (clip / norm).toFloat
      i = 0
      while i < g.length do { g(i) *= f; i += 1 }
    norm

  /** 予熱つき cosine スケジュール。 */
  def schedule(step: Int, total: Int, warmup: Int, peak: Double, floor: Double): Double =
    if step < warmup then peak * (step + 1) / warmup
    else
      val progress = (step - warmup).toDouble / math.max(1, total - warmup)
      floor + (peak - floor) * 0.5 * (1 + math.cos(math.Pi * math.min(1.0, progress)))
