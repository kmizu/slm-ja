package slm

import scala.util.Random

/** モデルの大きさ。 */
final case class Config(vocab: Int, d: Int, heads: Int, layers: Int, context: Int, ff: Int):
  require(d % heads == 0, "d はヘッド数で割り切れること")
  val headDim: Int = d / heads

/** パラメータは 1 本の配列。各部品はその中の区間（オフセットと長さ）。
  *
  * 重み W は「出力ごとのニューロン」を並べたもの: W(o*in + i) が o 番目のニューロンの i 番目の重み。
  */
final class Layout(val cfg: Config):
  private var cursor = 0
  private def take(n: Int): Int = { val at = cursor; cursor += n; at }

  val tok: Int = take(cfg.vocab * cfg.d)          // トークン埋め込み（出力ヘッドと共有）
  val pos: Int = take(cfg.context * cfg.d)        // 位置埋め込み
  final case class Layer(ln1g: Int, ln1b: Int, wq: Int, bq: Int, wk: Int, bk: Int, wv: Int, bv: Int, wo: Int, bo: Int,
                         ln2g: Int, ln2b: Int, w1: Int, b1: Int, w2: Int, b2: Int)
  val layer: Vector[Layer] = Vector.fill(cfg.layers) {
    val d = cfg.d
    Layer(take(d), take(d), take(d * d), take(d), take(d * d), take(d), take(d * d), take(d), take(d * d), take(d),
      take(d), take(d), take(cfg.ff * d), take(cfg.ff), take(d * cfg.ff), take(d))
  }
  val lnfg: Int = take(cfg.d)
  val lnfb: Int = take(cfg.d)
  val headb: Int = take(cfg.vocab)
  val size: Int = cursor

  /** 初期値: 重みは小さな乱数、バイアスは 0、LayerNorm のゲインは 1。残差の枝（wo, w2）は小さめ。 */
  def init(rng: Random): Array[Double] =
    val p = new Array[Double](size)
    def fill(at: Int, n: Int, scale: Double): Unit =
      var i = 0
      while i < n do
        p(at + i) = rng.nextGaussian() * scale
        i += 1
    def ones(at: Int, n: Int): Unit = java.util.Arrays.fill(p, at, at + n, 1.0)
    fill(tok, cfg.vocab * cfg.d, 0.02)
    fill(pos, cfg.context * cfg.d, 0.02)
    val residualScale = 0.02 / math.sqrt(2.0 * cfg.layers)
    for l <- layer do
      ones(l.ln1g, cfg.d); ones(l.ln2g, cfg.d)
      fill(l.wq, cfg.d * cfg.d, 0.02); fill(l.wk, cfg.d * cfg.d, 0.02); fill(l.wv, cfg.d * cfg.d, 0.02)
      fill(l.wo, cfg.d * cfg.d, residualScale)
      fill(l.w1, cfg.ff * cfg.d, 0.02)
      fill(l.w2, cfg.d * cfg.ff, residualScale)
    ones(lnfg, cfg.d)
    p

/** 1 系列ぶんの順伝播の途中結果。逆伝播で使う。 */
final class Cache(cfg: Config, val T: Int):
  val d = cfg.d
  val x: Array[Array[Double]] = Array.ofDim(cfg.layers + 1, T * d)  // 各層の入力（x(0) は埋め込み）
  val a: Array[Array[Double]] = Array.ofDim(cfg.layers, T * d)      // LN1 の出力
  val mean1: Array[Array[Double]] = Array.ofDim(cfg.layers, T)
  val rstd1: Array[Array[Double]] = Array.ofDim(cfg.layers, T)
  val q: Array[Array[Double]] = Array.ofDim(cfg.layers, T * d)
  val k: Array[Array[Double]] = Array.ofDim(cfg.layers, T * d)
  val v: Array[Array[Double]] = Array.ofDim(cfg.layers, T * d)
  val prob: Array[Array[Double]] = Array.ofDim(cfg.layers, cfg.heads * T * T) // 注意の割合
  val o: Array[Array[Double]] = Array.ofDim(cfg.layers, T * d)      // ヘッド連結後
  val x1: Array[Array[Double]] = Array.ofDim(cfg.layers, T * d)     // 注意の残差後
  val b: Array[Array[Double]] = Array.ofDim(cfg.layers, T * d)      // LN2 の出力
  val mean2: Array[Array[Double]] = Array.ofDim(cfg.layers, T)
  val rstd2: Array[Array[Double]] = Array.ofDim(cfg.layers, T)
  val u: Array[Array[Double]] = Array.ofDim(cfg.layers, T * cfg.ff) // FF の中間（ReLU 前）
  val f: Array[Double] = new Array(T * d)                            // 最終 LN の出力
  val meanf: Array[Double] = new Array(T)
  val rstdf: Array[Double] = new Array(T)
  val logits: Array[Double] = new Array(T * cfg.vocab)

/** Transformer 言語モデル。順伝播と、手で書いた逆伝播。行列ライブラリは使わず、ニューロンごとのループで書く。 */
final class Model(val cfg: Config):
  val layout = new Layout(cfg)
  private val d = cfg.d
  private val hd = cfg.headDim
  private val scale = 1.0 / math.sqrt(hd.toDouble)

  // ---- 基本部品 ----

  /** y[o] = Σ_i W[o][i] x[i] + b[o]  （out 個のニューロン） */
  private def dense(p: Array[Double], w: Int, b: Int, in: Int, out: Int, x: Array[Double], xAt: Int, y: Array[Double], yAt: Int): Unit =
    var o = 0
    while o < out do
      var s = p(b + o)
      val row = w + o * in
      var i = 0
      while i < in do
        s += p(row + i) * x(xAt + i)
        i += 1
      y(yAt + o) = s
      o += 1

  /** dense の逆: dx += Wᵀ dy, dW += dy xᵀ, db += dy */
  private def denseBackward(p: Array[Double], g: Array[Double], w: Int, b: Int, in: Int, out: Int,
                            x: Array[Double], xAt: Int, dy: Array[Double], dyAt: Int, dx: Array[Double], dxAt: Int): Unit =
    var o = 0
    while o < out do
      val gy = dy(dyAt + o)
      if gy != 0.0 then
        g(b + o) += gy
        val row = w + o * in
        var i = 0
        while i < in do
          g(row + i) += gy * x(xAt + i)
          dx(dxAt + i) += gy * p(row + i)
          i += 1
      o += 1

  private def layerNorm(p: Array[Double], gAt: Int, bAt: Int, x: Array[Double], xAt: Int, y: Array[Double], yAt: Int,
                        meanOut: Array[Double], rstdOut: Array[Double], t: Int): Unit =
    var s = 0.0
    var i = 0
    while i < d do { s += x(xAt + i); i += 1 }
    val mean = s / d
    var v = 0.0
    i = 0
    while i < d do { val c = x(xAt + i) - mean; v += c * c; i += 1 }
    val rstd = 1.0 / math.sqrt(v / d + 1e-5)
    meanOut(t) = mean
    rstdOut(t) = rstd
    i = 0
    while i < d do
      y(yAt + i) = (x(xAt + i) - mean) * rstd * p(gAt + i) + p(bAt + i)
      i += 1

  private def layerNormBackward(p: Array[Double], g: Array[Double], gAt: Int, bAt: Int, x: Array[Double], xAt: Int,
                                mean: Double, rstd: Double, dy: Array[Double], dyAt: Int, dx: Array[Double], dxAt: Int): Unit =
    // xhat = (x - mean) * rstd; dy_hat = dy * gain
    var sumDyHat = 0.0
    var sumDyHatXhat = 0.0
    var i = 0
    while i < d do
      val xhat = (x(xAt + i) - mean) * rstd
      val dyh = dy(dyAt + i) * p(gAt + i)
      g(gAt + i) += dy(dyAt + i) * xhat
      g(bAt + i) += dy(dyAt + i)
      sumDyHat += dyh
      sumDyHatXhat += dyh * xhat
      i += 1
    i = 0
    while i < d do
      val xhat = (x(xAt + i) - mean) * rstd
      val dyh = dy(dyAt + i) * p(gAt + i)
      dx(dxAt + i) += rstd * (dyh - sumDyHat / d - xhat * sumDyHatXhat / d)
      i += 1

  // ---- 順伝播 ----

  /** ids を通して各位置のロジットを cache に入れる。 */
  def forward(p: Array[Double], ids: Array[Int], cache: Cache): Unit =
    val T = ids.length
    require(T <= cfg.context && T == cache.T)
    val L = layout
    // 埋め込み + 位置
    val x0 = cache.x(0)
    var t = 0
    while t < T do
      val tokAt = L.tok + ids(t) * d
      val posAt = L.pos + t * d
      var i = 0
      while i < d do
        x0(t * d + i) = p(tokAt + i) + p(posAt + i)
        i += 1
      t += 1
    for l <- 0 until cfg.layers do
      val ly = L.layer(l)
      val x = cache.x(l)
      val a = cache.a(l)
      t = 0
      while t < T do
        layerNorm(p, ly.ln1g, ly.ln1b, x, t * d, a, t * d, cache.mean1(l), cache.rstd1(l), t)
        dense(p, ly.wq, ly.bq, d, d, a, t * d, cache.q(l), t * d)
        dense(p, ly.wk, ly.bk, d, d, a, t * d, cache.k(l), t * d)
        dense(p, ly.wv, ly.bv, d, d, a, t * d, cache.v(l), t * d)
        t += 1
      attention(cache.q(l), cache.k(l), cache.v(l), cache.prob(l), cache.o(l), T)
      val x1 = cache.x1(l)
      val tmp = new Array[Double](d)
      t = 0
      while t < T do
        dense(p, ly.wo, ly.bo, d, d, cache.o(l), t * d, tmp, 0)
        var i = 0
        while i < d do
          x1(t * d + i) = x(t * d + i) + tmp(i)
          i += 1
        layerNorm(p, ly.ln2g, ly.ln2b, x1, t * d, cache.b(l), t * d, cache.mean2(l), cache.rstd2(l), t)
        dense(p, ly.w1, ly.b1, d, cfg.ff, cache.b(l), t * d, cache.u(l), t * cfg.ff)
        val r = new Array[Double](cfg.ff)
        var j = 0
        while j < cfg.ff do { r(j) = math.max(0.0, cache.u(l)(t * cfg.ff + j)); j += 1 }
        dense(p, ly.w2, ly.b2, cfg.ff, d, r, 0, tmp, 0)
        val next = cache.x(l + 1)
        i = 0
        while i < d do
          next(t * d + i) = x1(t * d + i) + tmp(i)
          i += 1
        t += 1
    // 最終 LN と出力（埋め込みと共有）
    val xl = cache.x(cfg.layers)
    t = 0
    while t < T do
      layerNorm(p, L.lnfg, L.lnfb, xl, t * d, cache.f, t * d, cache.meanf, cache.rstdf, t)
      var vIdx = 0
      while vIdx < cfg.vocab do
        var s = p(L.headb + vIdx)
        val row = L.tok + vIdx * d
        var i = 0
        while i < d do { s += p(row + i) * cache.f(t * d + i); i += 1 }
        cache.logits(t * cfg.vocab + vIdx) = s
        vIdx += 1
      t += 1

  /** 因果的注意。各ヘッド、各位置 i について、j ≤ i との内積 → softmax → v の混合。 */
  private def attention(q: Array[Double], k: Array[Double], v: Array[Double], prob: Array[Double], o: Array[Double], T: Int): Unit =
    java.util.Arrays.fill(o, 0.0)
    var h = 0
    while h < cfg.heads do
      val off = h * hd
      var i = 0
      while i < T do
        val pAt = (h * T + i) * T
        var maxS = Double.NegativeInfinity
        var j = 0
        while j <= i do
          var s = 0.0
          var c = 0
          while c < hd do { s += q(i * d + off + c) * k(j * d + off + c); c += 1 }
          s *= scale
          prob(pAt + j) = s
          if s > maxS then maxS = s
          j += 1
        var total = 0.0
        j = 0
        while j <= i do { val e = math.exp(prob(pAt + j) - maxS); prob(pAt + j) = e; total += e; j += 1 }
        j = 0
        while j <= i do
          val w = prob(pAt + j) / total
          prob(pAt + j) = w
          var c = 0
          while c < hd do { o(i * d + off + c) += w * v(j * d + off + c); c += 1 }
          j += 1
        i += 1
      h += 1

  /** 損失（各位置の交差エントロピーの平均）と、ロジットの勾配（cache.logits を上書き）。 */
  def lossAndLogitGrad(cache: Cache, targets: Array[Int]): Double =
    val T = cache.T
    var loss = 0.0
    var t = 0
    while t < T do
      val at = t * cfg.vocab
      var maxL = Double.NegativeInfinity
      var vIdx = 0
      while vIdx < cfg.vocab do { if cache.logits(at + vIdx) > maxL then maxL = cache.logits(at + vIdx); vIdx += 1 }
      var total = 0.0
      vIdx = 0
      while vIdx < cfg.vocab do { total += math.exp(cache.logits(at + vIdx) - maxL); vIdx += 1 }
      val logZ = math.log(total) + maxL
      loss += logZ - cache.logits(at + targets(t))
      vIdx = 0
      while vIdx < cfg.vocab do
        val prob = math.exp(cache.logits(at + vIdx) - logZ)
        cache.logits(at + vIdx) = (prob - (if vIdx == targets(t) then 1.0 else 0.0)) / T
        vIdx += 1
      t += 1
    loss / T

  // ---- 逆伝播 ----

  /** forward と lossAndLogitGrad のあとに呼ぶ。勾配を g に足し込む。 */
  def backward(p: Array[Double], g: Array[Double], ids: Array[Int], cache: Cache): Unit =
    val T = cache.T
    val L = layout
    val dx = new Array[Double](T * d)      // いま逆伝播している層の入力に対する勾配
    val df = new Array[Double](T * d)
    // 出力ヘッド（埋め込みと共有）
    var t = 0
    while t < T do
      var vIdx = 0
      while vIdx < cfg.vocab do
        val gl = cache.logits(t * cfg.vocab + vIdx)
        if gl != 0.0 then
          g(L.headb + vIdx) += gl
          val row = L.tok + vIdx * d
          var i = 0
          while i < d do
            g(row + i) += gl * cache.f(t * d + i)
            df(t * d + i) += gl * p(row + i)
            i += 1
        vIdx += 1
      layerNormBackward(p, g, L.lnfg, L.lnfb, cache.x(cfg.layers), t * d, cache.meanf(t), cache.rstdf(t), df, t * d, dx, t * d)
      t += 1
    // ブロックを逆順に
    var l = cfg.layers - 1
    while l >= 0 do
      val ly = L.layer(l)
      val dx1 = new Array[Double](T * d)   // x1 に対する勾配（残差ぶんを先に入れる）
      System.arraycopy(dx, 0, dx1, 0, T * d)
      val db = new Array[Double](d)
      val du = new Array[Double](cfg.ff)
      val r = new Array[Double](cfg.ff)
      t = 0
      while t < T do
        // FF: y = W2 relu(W1 b + b1) + b2
        var j = 0
        while j < cfg.ff do { r(j) = math.max(0.0, cache.u(l)(t * cfg.ff + j)); j += 1 }
        java.util.Arrays.fill(du, 0.0)
        denseBackward(p, g, ly.w2, ly.b2, cfg.ff, d, r, 0, dx, t * d, du, 0)
        j = 0
        while j < cfg.ff do { if cache.u(l)(t * cfg.ff + j) <= 0.0 then du(j) = 0.0; j += 1 }
        java.util.Arrays.fill(db, 0.0)
        denseBackward(p, g, ly.w1, ly.b1, d, cfg.ff, cache.b(l), t * d, du, 0, db, 0)
        layerNormBackward(p, g, ly.ln2g, ly.ln2b, cache.x1(l), t * d, cache.mean2(l)(t), cache.rstd2(l)(t), db, 0, dx1, t * d)
        t += 1
      // 注意: x1 = x + Wo o + bo
      val dO = new Array[Double](T * d)
      t = 0
      while t < T do
        denseBackward(p, g, ly.wo, ly.bo, d, d, cache.o(l), t * d, dx1, t * d, dO, t * d)
        t += 1
      val dq = new Array[Double](T * d)
      val dk = new Array[Double](T * d)
      val dv = new Array[Double](T * d)
      attentionBackward(cache.q(l), cache.k(l), cache.v(l), cache.prob(l), dO, dq, dk, dv, T)
      val da = new Array[Double](T * d)
      val dxNext = new Array[Double](T * d)
      System.arraycopy(dx1, 0, dxNext, 0, T * d) // 残差
      t = 0
      while t < T do
        denseBackward(p, g, ly.wq, ly.bq, d, d, cache.a(l), t * d, dq, t * d, da, t * d)
        denseBackward(p, g, ly.wk, ly.bk, d, d, cache.a(l), t * d, dk, t * d, da, t * d)
        denseBackward(p, g, ly.wv, ly.bv, d, d, cache.a(l), t * d, dv, t * d, da, t * d)
        layerNormBackward(p, g, ly.ln1g, ly.ln1b, cache.x(l), t * d, cache.mean1(l)(t), cache.rstd1(l)(t), da, t * d, dxNext, t * d)
        t += 1
      System.arraycopy(dxNext, 0, dx, 0, T * d)
      l -= 1
    // 埋め込み
    t = 0
    while t < T do
      val tokAt = L.tok + ids(t) * d
      val posAt = L.pos + t * d
      var i = 0
      while i < d do
        g(tokAt + i) += dx(t * d + i)
        g(posAt + i) += dx(t * d + i)
        i += 1
      t += 1

  private def attentionBackward(q: Array[Double], k: Array[Double], v: Array[Double], prob: Array[Double],
                                dO: Array[Double], dq: Array[Double], dk: Array[Double], dv: Array[Double], T: Int): Unit =
    val dp = new Array[Double](T)
    var h = 0
    while h < cfg.heads do
      val off = h * hd
      var i = 0
      while i < T do
        val pAt = (h * T + i) * T
        // dv_j += w_ij dO_i ;  dw_ij = dO_i · v_j
        var j = 0
        while j <= i do
          val w = prob(pAt + j)
          var s = 0.0
          var c = 0
          while c < hd do
            val go = dO(i * d + off + c)
            dv(j * d + off + c) += w * go
            s += go * v(j * d + off + c)
            c += 1
          dp(j) = s
          j += 1
        // softmax の逆: ds_ij = w_ij (dw_ij - Σ_j w_ij dw_ij)
        var dot = 0.0
        j = 0
        while j <= i do { dot += prob(pAt + j) * dp(j); j += 1 }
        j = 0
        while j <= i do
          val ds = prob(pAt + j) * (dp(j) - dot) * scale
          var c = 0
          while c < hd do
            dq(i * d + off + c) += ds * k(j * d + off + c)
            dk(j * d + off + c) += ds * q(i * d + off + c)
            c += 1
          j += 1
        i += 1
      h += 1

  /** 1 系列の損失と勾配（g に足し込む）。 */
  def lossAndGrad(p: Array[Double], g: Array[Double], window: Array[Int]): Double =
    val ids = window.dropRight(1)
    val targets = window.drop(1)
    val cache = new Cache(cfg, ids.length)
    forward(p, ids, cache)
    val loss = lossAndLogitGrad(cache, targets)
    backward(p, g, ids, cache)
    loss

  /** 最後の位置のロジット（生成用）。 */
  def lastLogits(p: Array[Double], ids: Array[Int]): Array[Double] =
    val cache = new Cache(cfg, ids.length)
    forward(p, ids, cache)
    cache.logits.slice((ids.length - 1) * cfg.vocab, ids.length * cfg.vocab)

  def parameterCount: Int = layout.size
