package slm.reference

import scala.util.Random

/** モデルの大きさ。 */
final case class Config(vocab: Int, d: Int, heads: Int, layers: Int, context: Int, ff: Int, attention: String = "softmax"):
  require(d % heads == 0, "d はヘッド数で割り切れること")
  val headDim: Int = d / heads
  def isLinear: Boolean = attention == "linear"
  def gamma(h: Int): Double = 1.0 - math.pow(2.0, -(5.0 + h))

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

/** 1 系列ぶんの順伝播の途中結果と、逆伝播の作業領域。スレッドごとに 1 つ持って使い回す。 */
final class Workspace(cfg: Config):
  val T: Int = cfg.context
  private val d = cfg.d
  val x: Array[Array[Double]] = Array.ofDim(cfg.layers + 1, T * d)  // 各層の入力（x(0) は埋め込み）
  val a: Array[Array[Double]] = Array.ofDim(cfg.layers, T * d)      // LN1 の出力
  val mean1: Array[Array[Double]] = Array.ofDim(cfg.layers, T)
  val rstd1: Array[Array[Double]] = Array.ofDim(cfg.layers, T)
  val q: Array[Array[Double]] = Array.ofDim(cfg.layers, T * d)
  val k: Array[Array[Double]] = Array.ofDim(cfg.layers, T * d)
  val v: Array[Array[Double]] = Array.ofDim(cfg.layers, T * d)
  val prob: Array[Array[Double]] = Array.ofDim(cfg.layers, if cfg.isLinear then 1 else cfg.heads * T * T)
  val fq: Array[Array[Double]] = Array.ofDim(cfg.layers, T * d)
  val fk: Array[Array[Double]] = Array.ofDim(cfg.layers, T * d)
  val den: Array[Array[Double]] = Array.ofDim(cfg.layers, cfg.heads * T)
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
  // 逆伝播の作業領域
  val dx: Array[Double] = new Array(T * d)
  val df: Array[Double] = new Array(T * d)
  val dx1: Array[Double] = new Array(T * d)
  val dO: Array[Double] = new Array(T * d)
  val dq: Array[Double] = new Array(T * d)
  val dk: Array[Double] = new Array(T * d)
  val dv: Array[Double] = new Array(T * d)
  val da: Array[Double] = new Array(T * d)
  val dxNext: Array[Double] = new Array(T * d)
  val tmp: Array[Double] = new Array(T * d)     // 注意の出力 Dense の結果（全トークン）
  val rAll: Array[Double] = new Array(T * cfg.ff) // ReLU 後（全トークン）
  val duAll: Array[Double] = new Array(T * cfg.ff)
  val dp: Array[Double] = new Array(T)

/** Transformer 言語モデル。順伝播と、手で書いた逆伝播。行列ライブラリは使わず、ニューロンごとのループで書く。 */
final class Model(val cfg: Config):
  val layout = new Layout(cfg)
  private val d = cfg.d
  private val hd = cfg.headDim
  private val scale = 1.0 / math.sqrt(hd.toDouble)

  def workspace(): Workspace = new Workspace(cfg)

  // ---- 基本部品 ----

  /** 内積（SIMD 版）。 */
  private def dot(a: Array[Double], aAt: Int, b: Array[Double], bAt: Int, n: Int): Double = Simd.dot(a, aAt, b, bAt, n)

  /** 全トークンぶんの y[t][o] = Σ_i W[o][i] x[t][i] + b[o]。
    * ニューロン（重みの行）を外側、トークンを内側に回す: 重み 1 行を L1 に置いたまま T 回使い、重みの読み出しを系列あたり 1 回にする。
    */
  private def denseAll(p: Array[Double], w: Int, b: Int, in: Int, out: Int, x: Array[Double], y: Array[Double], T: Int): Unit =
    var o = 0
    while o < out do
      val row = w + o * in
      val bias = p(b + o)
      var t = 0
      while t < T do
        y(t * out + o) = bias + dot(p, row, x, t * in, in)
        t += 1
      o += 1

  /** denseAll の逆: dx[t] += Wᵀ dy[t], dW += Σ_t dy[t] x[t]ᵀ, db += Σ_t dy[t]。同じくニューロン外側。 */
  private def denseBackwardAll(p: Array[Double], g: Array[Double], w: Int, b: Int, in: Int, out: Int,
                               x: Array[Double], dy: Array[Double], dx: Array[Double], T: Int): Unit =
    var o = 0
    while o < out do
      val row = w + o * in
      var t = 0
      while t < T do
        val gy = dy(t * out + o)
        if gy != 0.0 then
          g(b + o) += gy
          val xAt = t * in
          Simd.axpy(g, row, gy, x, xAt, in)
          Simd.axpy(dx, xAt, gy, p, row, in)
        t += 1
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
    val meanDyHat = sumDyHat / d
    val meanDyHatXhat = sumDyHatXhat / d
    i = 0
    while i < d do
      val xhat = (x(xAt + i) - mean) * rstd
      val dyh = dy(dyAt + i) * p(gAt + i)
      dx(dxAt + i) += rstd * (dyh - meanDyHat - xhat * meanDyHatXhat)
      i += 1

  // ---- 順伝播 ----

  /** ids を通して各位置のロジットを ws に入れる。 */
  def forward(p: Array[Double], ids: Array[Int], ws: Workspace): Unit =
    val T = ids.length
    require(T <= cfg.context && T <= ws.T)
    val L = layout
    val x0 = ws.x(0)
    var t = 0
    while t < T do
      val tokAt = L.tok + ids(t) * d
      val posAt = L.pos + t * d
      var i = 0
      while i < d do
        x0(t * d + i) = p(tokAt + i) + p(posAt + i)
        i += 1
      t += 1
    val tmp = ws.tmp
    val rAll = ws.rAll
    for l <- 0 until cfg.layers do
      val ly = L.layer(l)
      val x = ws.x(l)
      val a = ws.a(l)
      t = 0
      while t < T do
        layerNorm(p, ly.ln1g, ly.ln1b, x, t * d, a, t * d, ws.mean1(l), ws.rstd1(l), t)
        t += 1
      denseAll(p, ly.wq, ly.bq, d, d, a, ws.q(l), T)
      denseAll(p, ly.wk, ly.bk, d, d, a, ws.k(l), T)
      denseAll(p, ly.wv, ly.bv, d, d, a, ws.v(l), T)
      if cfg.isLinear then linearAttention(ws.q(l), ws.k(l), ws.v(l), ws.fq(l), ws.fk(l), ws.den(l), ws.o(l), T)
      else attention(ws.q(l), ws.k(l), ws.v(l), ws.prob(l), ws.o(l), T)
      val x1 = ws.x1(l)
      val u = ws.u(l)
      val next = ws.x(l + 1)
      denseAll(p, ly.wo, ly.bo, d, d, ws.o(l), tmp, T)
      var i = 0
      while i < T * d do { x1(i) = x(i) + tmp(i); i += 1 }
      t = 0
      while t < T do
        layerNorm(p, ly.ln2g, ly.ln2b, x1, t * d, ws.b(l), t * d, ws.mean2(l), ws.rstd2(l), t)
        t += 1
      denseAll(p, ly.w1, ly.b1, d, cfg.ff, ws.b(l), u, T)
      var j = 0
      while j < T * cfg.ff do { rAll(j) = math.max(0.0, u(j)); j += 1 }
      denseAll(p, ly.w2, ly.b2, cfg.ff, d, rAll, tmp, T)
      i = 0
      while i < T * d do { next(i) = x1(i) + tmp(i); i += 1 }
    // 最終 LN と出力（埋め込みと共有）。語彙を外側、トークンを内側に
    val xl = ws.x(cfg.layers)
    t = 0
    while t < T do
      layerNorm(p, L.lnfg, L.lnfb, xl, t * d, ws.f, t * d, ws.meanf, ws.rstdf, t)
      t += 1
    var vIdx = 0
    while vIdx < cfg.vocab do
      val row = L.tok + vIdx * d
      val bias = p(L.headb + vIdx)
      t = 0
      while t < T do
        ws.logits(t * cfg.vocab + vIdx) = bias + dot(p, row, ws.f, t * d, d)
        t += 1
      vIdx += 1

  /** 因果的注意。各ヘッド、各位置 i について、j ≤ i との内積 → softmax → v の混合。 */
  private def attention(q: Array[Double], k: Array[Double], v: Array[Double], prob: Array[Double], o: Array[Double], T: Int): Unit =
    java.util.Arrays.fill(o, 0, T * d, 0.0)
    var h = 0
    while h < cfg.heads do
      val off = h * hd
      var i = 0
      while i < T do
        val pAt = (h * T + i) * T
        var maxS = Double.NegativeInfinity
        var j = 0
        while j <= i do
          val s = dot(q, i * d + off, k, j * d + off, hd) * scale
          prob(pAt + j) = s
          if s > maxS then maxS = s
          j += 1
        var total = 0.0
        j = 0
        while j <= i do { val e = math.exp(prob(pAt + j) - maxS); prob(pAt + j) = e; total += e; j += 1 }
        val inv = 1.0 / total
        j = 0
        while j <= i do
          val w = prob(pAt + j) * inv
          prob(pAt + j) = w
          Simd.axpy(o, i * d + off, w, v, j * d + off, hd)
          j += 1
        i += 1
      h += 1


  // ---- 線形注意（Double 参照実装、Float 版と同じ式を素直なループで） ----
  private val eps = 1e-6
  private def phi(x: Double): Double = if x > 0 then x + 1 else math.exp(x)
  private def phiPrime(x: Double): Double = if x > 0 then 1.0 else math.exp(x)

  private def linearAttention(q: Array[Double], k: Array[Double], v: Array[Double], fq: Array[Double], fk: Array[Double],
                              den: Array[Double], o: Array[Double], T: Int): Unit =
    val s = new Array[Double](hd * hd); val z = new Array[Double](hd)
    for h <- 0 until cfg.heads do
      val g = cfg.gamma(h); val off = h * hd
      java.util.Arrays.fill(s, 0.0); java.util.Arrays.fill(z, 0.0)
      for i <- 0 until T do
        val at = i * d + off
        for c <- 0 until hd do { fq(at + c) = phi(q(at + c)); fk(at + c) = phi(k(at + c)) }
        for r <- 0 until hd do
          z(r) = g * z(r) + fk(at + r)
          for c <- 0 until hd do s(r * hd + c) = g * s(r * hd + c) + fk(at + r) * v(at + c)
        var dn = eps
        for r <- 0 until hd do dn += fq(at + r) * z(r)
        den(h * T + i) = dn
        for c <- 0 until hd do
          var a = 0.0
          for r <- 0 until hd do a += fq(at + r) * s(r * hd + c)
          o(at + c) = a / dn

  private def linearAttentionBackward(q: Array[Double], k: Array[Double], v: Array[Double], fq: Array[Double], fk: Array[Double],
                                      den: Array[Double], o: Array[Double], dO: Array[Double],
                                      dq: Array[Double], dk: Array[Double], dv: Array[Double], T: Int): Unit =
    val s = new Array[Double](hd * hd); val z = new Array[Double](hd)
    val da = new Array[Double](T * d); val dden = new Array[Double](cfg.heads * T)
    for h <- 0 until cfg.heads do
      val g = cfg.gamma(h); val off = h * hd
      java.util.Arrays.fill(s, 0.0); java.util.Arrays.fill(z, 0.0)
      for i <- 0 until T do
        val at = i * d + off
        for r <- 0 until hd do
          z(r) = g * z(r) + fk(at + r)
          for c <- 0 until hd do s(r * hd + c) = g * s(r * hd + c) + fk(at + r) * v(at + c)
        val dn = den(h * T + i)
        var dotDoO = 0.0
        for c <- 0 until hd do { da(at + c) = dO(at + c) / dn; dotDoO += dO(at + c) * o(at + c) }
        val dd = -dotDoO / dn
        dden(h * T + i) = dd
        for r <- 0 until hd do
          var dfq = dd * z(r)
          for c <- 0 until hd do dfq += da(at + c) * s(r * hd + c)
          dq(at + r) += dfq * phiPrime(q(at + r))
      java.util.Arrays.fill(s, 0.0); java.util.Arrays.fill(z, 0.0)
      for i <- (T - 1) to 0 by -1 do
        val at = i * d + off
        val dd = dden(h * T + i)
        for r <- 0 until hd do
          z(r) = g * z(r) + dd * fq(at + r)
          for c <- 0 until hd do s(r * hd + c) = g * s(r * hd + c) + fq(at + r) * da(at + c)
        for r <- 0 until hd do
          var dfk = z(r)
          for c <- 0 until hd do dfk += s(r * hd + c) * v(at + c)
          dk(at + r) += dfk * phiPrime(k(at + r))
          for c <- 0 until hd do dv(at + c) += s(r * hd + c) * fk(at + r)

  /** 損失（各位置の交差エントロピーの平均）と、ロジットの勾配（ws.logits を上書き）。 */
  def lossAndLogitGrad(ws: Workspace, T: Int, targets: Array[Int]): Double =
    var loss = 0.0
    var t = 0
    while t < T do
      val at = t * cfg.vocab
      var maxL = Double.NegativeInfinity
      var vIdx = 0
      while vIdx < cfg.vocab do { if ws.logits(at + vIdx) > maxL then maxL = ws.logits(at + vIdx); vIdx += 1 }
      var total = 0.0
      vIdx = 0
      while vIdx < cfg.vocab do { total += math.exp(ws.logits(at + vIdx) - maxL); vIdx += 1 }
      val logZ = math.log(total) + maxL
      loss += logZ - ws.logits(at + targets(t))
      vIdx = 0
      while vIdx < cfg.vocab do
        val prob = math.exp(ws.logits(at + vIdx) - logZ)
        ws.logits(at + vIdx) = (prob - (if vIdx == targets(t) then 1.0 else 0.0)) / T
        vIdx += 1
      t += 1
    loss / T

  // ---- 逆伝播 ----

  /** forward と lossAndLogitGrad のあとに呼ぶ。勾配を g に足し込む。 */
  def backward(p: Array[Double], g: Array[Double], ids: Array[Int], ws: Workspace): Unit =
    val T = ids.length
    val L = layout
    val n = T * d
    val dx = ws.dx
    val df = ws.df
    java.util.Arrays.fill(dx, 0, n, 0.0)
    java.util.Arrays.fill(df, 0, n, 0.0)
    // 出力ヘッド（埋め込みと共有）。語彙を外側、トークンを内側に
    var vIdx = 0
    while vIdx < cfg.vocab do
      val row = L.tok + vIdx * d
      var t = 0
      while t < T do
        val gl = ws.logits(t * cfg.vocab + vIdx)
        g(L.headb + vIdx) += gl
        Simd.axpy(g, row, gl, ws.f, t * d, d)
        Simd.axpy(df, t * d, gl, p, row, d)
        t += 1
      vIdx += 1
    var t = 0
    while t < T do
      layerNormBackward(p, g, L.lnfg, L.lnfb, ws.x(cfg.layers), t * d, ws.meanf(t), ws.rstdf(t), df, t * d, dx, t * d)
      t += 1
    // ブロックを逆順に
    val dx1 = ws.dx1; val dO = ws.dO; val dq = ws.dq; val dk = ws.dk; val dv = ws.dv; val da = ws.da; val dxNext = ws.dxNext
    val duAll = ws.duAll; val rAll = ws.rAll; val tmp = ws.tmp
    var l = cfg.layers - 1
    while l >= 0 do
      val ly = L.layer(l)
      System.arraycopy(dx, 0, dx1, 0, n)   // 残差ぶん
      val u = ws.u(l)
      // FF: y = W2 relu(W1 b + b1) + b2
      var j = 0
      while j < T * cfg.ff do { rAll(j) = math.max(0.0, u(j)); j += 1 }
      java.util.Arrays.fill(duAll, 0, T * cfg.ff, 0.0)
      denseBackwardAll(p, g, ly.w2, ly.b2, cfg.ff, d, rAll, dx, duAll, T)
      j = 0
      while j < T * cfg.ff do { if u(j) <= 0.0 then duAll(j) = 0.0; j += 1 }
      java.util.Arrays.fill(tmp, 0, n, 0.0)
      denseBackwardAll(p, g, ly.w1, ly.b1, d, cfg.ff, ws.b(l), duAll, tmp, T)
      t = 0
      while t < T do
        layerNormBackward(p, g, ly.ln2g, ly.ln2b, ws.x1(l), t * d, ws.mean2(l)(t), ws.rstd2(l)(t), tmp, t * d, dx1, t * d)
        t += 1
      // 注意: x1 = x + Wo o + bo
      java.util.Arrays.fill(dO, 0, n, 0.0)
      denseBackwardAll(p, g, ly.wo, ly.bo, d, d, ws.o(l), dx1, dO, T)
      java.util.Arrays.fill(dq, 0, n, 0.0); java.util.Arrays.fill(dk, 0, n, 0.0); java.util.Arrays.fill(dv, 0, n, 0.0)
      if cfg.isLinear then linearAttentionBackward(ws.q(l), ws.k(l), ws.v(l), ws.fq(l), ws.fk(l), ws.den(l), ws.o(l), dO, dq, dk, dv, T)
      else attentionBackward(ws.q(l), ws.k(l), ws.v(l), ws.prob(l), dO, dq, dk, dv, ws.dp, T)
      java.util.Arrays.fill(da, 0, n, 0.0)
      System.arraycopy(dx1, 0, dxNext, 0, n) // 残差
      denseBackwardAll(p, g, ly.wq, ly.bq, d, d, ws.a(l), dq, da, T)
      denseBackwardAll(p, g, ly.wk, ly.bk, d, d, ws.a(l), dk, da, T)
      denseBackwardAll(p, g, ly.wv, ly.bv, d, d, ws.a(l), dv, da, T)
      t = 0
      while t < T do
        layerNormBackward(p, g, ly.ln1g, ly.ln1b, ws.x(l), t * d, ws.mean1(l)(t), ws.rstd1(l)(t), da, t * d, dxNext, t * d)
        t += 1
      System.arraycopy(dxNext, 0, dx, 0, n)
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
                                dO: Array[Double], dq: Array[Double], dk: Array[Double], dv: Array[Double], dp: Array[Double], T: Int): Unit =
    var h = 0
    while h < cfg.heads do
      val off = h * hd
      var i = 0
      while i < T do
        val pAt = (h * T + i) * T
        var j = 0
        while j <= i do
          val w = prob(pAt + j)
          Simd.axpy(dv, j * d + off, w, dO, i * d + off, hd)
          dp(j) = dot(dO, i * d + off, v, j * d + off, hd)
          j += 1
        var dotSum = 0.0
        j = 0
        while j <= i do { dotSum += prob(pAt + j) * dp(j); j += 1 }
        j = 0
        while j <= i do
          val ds = prob(pAt + j) * (dp(j) - dotSum) * scale
          Simd.axpy(dq, i * d + off, ds, k, j * d + off, hd)
          Simd.axpy(dk, j * d + off, ds, q, i * d + off, hd)
          j += 1
        i += 1
      h += 1

  /** 1 系列の損失と勾配（g に足し込む）。作業領域を渡す版。 */
  def lossAndGrad(p: Array[Double], g: Array[Double], window: Array[Int], ws: Workspace): Double =
    val ids = window.dropRight(1)
    val targets = window.drop(1)
    forward(p, ids, ws)
    val loss = lossAndLogitGrad(ws, ids.length, targets)
    backward(p, g, ids, ws)
    loss

  def lossAndGrad(p: Array[Double], g: Array[Double], window: Array[Int]): Double =
    lossAndGrad(p, g, window, workspace())

  /** 最後の位置のロジット（生成用）。 */
  def lastLogits(p: Array[Double], ids: Array[Int], ws: Workspace): Array[Double] =
    forward(p, ids, ws)
    ws.logits.slice((ids.length - 1) * cfg.vocab, ids.length * cfg.vocab)

  def parameterCount: Int = layout.size
