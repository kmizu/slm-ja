package slm

/** AdamW。パラメータ配列を直接更新する（学習ループの中でだけ使う）。 */
final class AdamW(size: Int, val beta1: Double = 0.9, val beta2: Double = 0.95, val eps: Double = 1e-8):
  private val m = new Array[Double](size)
  private val v = new Array[Double](size)
  var step: Int = 0

  /** `decayMask(i)` が真のパラメータにだけ重み減衰をかける。 */
  def update(p: Array[Double], g: Array[Double], lr: Double, weightDecay: Double, decayMask: Int => Boolean): Unit =
    step += 1
    val c1 = 1.0 / (1.0 - math.pow(beta1, step))
    val c2 = 1.0 / (1.0 - math.pow(beta2, step))
    var i = 0
    while i < size do
      val gi = g(i)
      m(i) = beta1 * m(i) + (1 - beta1) * gi
      v(i) = beta2 * v(i) + (1 - beta2) * gi * gi
      val update = (m(i) * c1) / (math.sqrt(v(i) * c2) + eps)
      val decay = if decayMask(i) then weightDecay * p(i) else 0.0
      p(i) -= lr * (update + decay)
      i += 1

object Optimizer:

  /** 勾配の全体ノルムを clip 以下に抑える。返り値は元のノルム。 */
  def clipGlobalNorm(g: Array[Double], clip: Double): Double =
    var s = 0.0
    var i = 0
    while i < g.length do { s += g(i) * g(i); i += 1 }
    val norm = math.sqrt(s)
    if norm > clip then
      val f = clip / norm
      i = 0
      while i < g.length do { g(i) *= f; i += 1 }
    norm

  /** 予熱つき cosine スケジュール。 */
  def schedule(step: Int, total: Int, warmup: Int, peak: Double, floor: Double): Double =
    if step < warmup then peak * (step + 1) / warmup
    else
      val progress = (step - warmup).toDouble / math.max(1, total - warmup)
      floor + (peak - floor) * 0.5 * (1 + math.cos(math.Pi * math.min(1.0, progress)))
