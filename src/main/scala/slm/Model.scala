package slm

import scala.util.Random

/** モデルの大きさ。 */
final case class Config(vocab: Int, d: Int, heads: Int, layers: Int, context: Int, ff: Int, attention: String = "softmax"):
  require(d % heads == 0, "d はヘッド数で割り切れること")
  require(attention == "softmax" || attention == "linear", s"attention は softmax か linear: $attention")
  val headDim: Int = d / heads
  def isLinear: Boolean = attention == "linear"
  /** 線形注意の減衰（ヘッドごと）。RetNet と同じ 1 - 2^-(5+h)。 */
  def gamma(h: Int): Float = (1.0 - math.pow(2.0, -(5.0 + h))).toFloat

/** パラメータは 1 本の配列（Float32）。各部品はその中の区間（オフセットと長さ）。
  *
  * 重み W は「出力ごとのニューロン」を並べたもの: W(o*in + i) が o 番目のニューロンの i 番目の重み。
  */
final class Layout(val cfg: Config):
  private var cursor = 0L
  private def take(n: Long): Int =
    val at = cursor
    cursor += n
    require(cursor <= Int.MaxValue, s"パラメータ配列が Int の範囲を超える: $cursor")
    at.toInt

  val tok: Int = take(cfg.vocab.toLong * cfg.d)   // トークン埋め込み（出力ヘッドと共有）
  val pos: Int = take(cfg.context.toLong * cfg.d)        // 位置埋め込み
  final case class Layer(ln1g: Int, ln1b: Int, wq: Int, bq: Int, wk: Int, bk: Int, wv: Int, bv: Int, wo: Int, bo: Int,
                         ln2g: Int, ln2b: Int, w1: Int, b1: Int, w2: Int, b2: Int)
  val layer: Vector[Layer] = Vector.fill(cfg.layers) {
    val d = cfg.d.toLong
    Layer(take(d), take(d), take(d * d), take(d), take(d * d), take(d), take(d * d), take(d), take(d * d), take(d),
      take(d), take(d), take(cfg.ff * d), take(cfg.ff), take(d * cfg.ff), take(d))
  }
  val lnfg: Int = take(cfg.d)
  val lnfb: Int = take(cfg.d)
  val headb: Int = take(cfg.vocab)
  val size: Int = cursor.toInt

  /** 初期値: 重みは小さな乱数、バイアスは 0、LayerNorm のゲインは 1。残差の枝（wo, w2）は小さめ。 */
  def init(rng: Random): Array[Float] =
    val p = new Array[Float](size)
    def fill(at: Int, n: Int, scale: Double): Unit =
      var i = 0
      while i < n do
        p(at + i) = (rng.nextGaussian() * scale).toFloat
        i += 1
    def ones(at: Int, n: Int): Unit = java.util.Arrays.fill(p, at, at + n, 1.0f)
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
  val x: Array[Array[Float]] = Array.ofDim(cfg.layers + 1, T * d)  // 各層の入力（x(0) は埋め込み）
  val a: Array[Array[Float]] = Array.ofDim(cfg.layers, T * d)      // LN1 の出力
  val mean1: Array[Array[Float]] = Array.ofDim(cfg.layers, T)
  val rstd1: Array[Array[Float]] = Array.ofDim(cfg.layers, T)
  val q: Array[Array[Float]] = Array.ofDim(cfg.layers, T * d)
  val k: Array[Array[Float]] = Array.ofDim(cfg.layers, T * d)
  val v: Array[Array[Float]] = Array.ofDim(cfg.layers, T * d)
  val prob: Array[Array[Float]] = Array.ofDim(cfg.layers, if cfg.isLinear then 1 else cfg.heads * T * T) // 注意の割合（softmax 型）
  // 線形注意用
  val fq: Array[Array[Float]] = Array.ofDim(cfg.layers, if cfg.isLinear then T * d else 1) // φ(q)
  val fk: Array[Array[Float]] = Array.ofDim(cfg.layers, if cfg.isLinear then T * d else 1) // φ(k)
  val den: Array[Array[Float]] = Array.ofDim(cfg.layers, if cfg.isLinear then cfg.heads * T else 1) // 分母
  val sState: Array[Float] = new Array(cfg.headDim * cfg.headDim)
  val zState: Array[Float] = new Array(cfg.headDim)
  val aTmp: Array[Float] = new Array(cfg.headDim)
  val daAll: Array[Float] = new Array(if cfg.isLinear then T * d else 1)
  val ddenAll: Array[Float] = new Array(if cfg.isLinear then cfg.heads * T else 1)
  val o: Array[Array[Float]] = Array.ofDim(cfg.layers, T * d)      // ヘッド連結後
  val x1: Array[Array[Float]] = Array.ofDim(cfg.layers, T * d)     // 注意の残差後
  val b: Array[Array[Float]] = Array.ofDim(cfg.layers, T * d)      // LN2 の出力
  val mean2: Array[Array[Float]] = Array.ofDim(cfg.layers, T)
  val rstd2: Array[Array[Float]] = Array.ofDim(cfg.layers, T)
  val u: Array[Array[Float]] = Array.ofDim(cfg.layers, T * cfg.ff) // FF の中間（ReLU 前）
  val f: Array[Float] = new Array(T * d)                            // 最終 LN の出力
  val meanf: Array[Float] = new Array(T)
  val rstdf: Array[Float] = new Array(T)
  val logits: Array[Float] = new Array(T * cfg.vocab)
  // 逆伝播の作業領域
  val dx: Array[Float] = new Array(T * d)
  val df: Array[Float] = new Array(T * d)
  val dx1: Array[Float] = new Array(T * d)
  val dO: Array[Float] = new Array(T * d)
  val dq: Array[Float] = new Array(T * d)
  val dk: Array[Float] = new Array(T * d)
  val dv: Array[Float] = new Array(T * d)
  val da: Array[Float] = new Array(T * d)
  val dxNext: Array[Float] = new Array(T * d)
  val tmp: Array[Float] = new Array(T * d)
  val rAll: Array[Float] = new Array(T * cfg.ff)
  val duAll: Array[Float] = new Array(T * cfg.ff)
  val dp: Array[Float] = new Array(T)
  val out4: Array[Float] = new Array(8)

/** Transformer 言語モデル（Float32）。順伝播と、手で書いた逆伝播。行列ライブラリは使わず、ニューロンごとのループで書く。 */
final class Model(val cfg: Config):
  val layout = new Layout(cfg)
  private val d = cfg.d
  private val hd = cfg.headDim
  private val scale = (1.0 / math.sqrt(hd.toDouble)).toFloat

  def workspace(): Workspace = new Workspace(cfg)

  // ---- 基本部品 ----

  /** 全トークンぶんの y[t][o] = Σ_i W[o][i] x[t][i] + b[o]。
    * ニューロン（重みの行）を外側、トークンを内側に。重み 1 行の読み込みをトークン 4 本で共有する。
    */
  private def denseAll(p: Array[Float], w: Int, b: Int, in: Int, out: Int, x: Array[Float], y: Array[Float], T: Int, out4: Array[Float]): Unit =
    var o = 0
    while o + 1 < out do
      val row0 = w + o * in
      val row1 = row0 + in
      val bias0 = p(b + o)
      val bias1 = p(b + o + 1)
      var t = 0
      while t + 3 < T do
        Simd.dot4x2(p, row0, row1, x, t * in, (t + 1) * in, (t + 2) * in, (t + 3) * in, in, out4)
        y(t * out + o) = bias0 + out4(0); y((t + 1) * out + o) = bias0 + out4(1)
        y((t + 2) * out + o) = bias0 + out4(2); y((t + 3) * out + o) = bias0 + out4(3)
        y(t * out + o + 1) = bias1 + out4(4); y((t + 1) * out + o + 1) = bias1 + out4(5)
        y((t + 2) * out + o + 1) = bias1 + out4(6); y((t + 3) * out + o + 1) = bias1 + out4(7)
        t += 4
      while t < T do
        y(t * out + o) = bias0 + Simd.dot(p, row0, x, t * in, in)
        y(t * out + o + 1) = bias1 + Simd.dot(p, row1, x, t * in, in)
        t += 1
      o += 2
    while o < out do
      val row = w + o * in
      val bias = p(b + o)
      var t = 0
      while t < T do
        y(t * out + o) = bias + Simd.dot(p, row, x, t * in, in)
        t += 1
      o += 1

  /** denseAll の逆: dx[t] += Wᵀ dy[t], dW += Σ_t dy[t] x[t]ᵀ, db += Σ_t dy[t]。
    * dW はニューロン外側・トークン 4 本まとめ（重みの行 1 本を書き戻す）、dx はニューロン 4 本まとめ・トークン内側（dx の行 1 本を書き戻す）。
    */
  private def denseBackwardAll(p: Array[Float], g: Array[Float], w: Int, b: Int, in: Int, out: Int,
                               x: Array[Float], dy: Array[Float], dx: Array[Float], T: Int): Unit =
    var o = 0
    while o < out do
      val row = w + o * in
      var t = 0
      while t + 3 < T do
        val g0 = dy(t * out + o); val g1 = dy((t + 1) * out + o); val g2 = dy((t + 2) * out + o); val g3 = dy((t + 3) * out + o)
        if g0 != 0f || g1 != 0f || g2 != 0f || g3 != 0f then
          g(b + o) += g0 + g1 + g2 + g3
          Simd.axpy4(g, row, g0, g1, g2, g3, x, t * in, (t + 1) * in, (t + 2) * in, (t + 3) * in, in)
        t += 4
      while t < T do
        val gy = dy(t * out + o)
        if gy != 0f then
          g(b + o) += gy
          Simd.axpy(g, row, gy, x, t * in, in)
        t += 1
      o += 1
    o = 0
    while o + 3 < out do
      val row0 = w + o * in
      var t = 0
      while t < T do
        val at = t * out + o
        val g0 = dy(at); val g1 = dy(at + 1); val g2 = dy(at + 2); val g3 = dy(at + 3)
        if g0 != 0f || g1 != 0f || g2 != 0f || g3 != 0f then
          Simd.axpy4(dx, t * in, g0, g1, g2, g3, p, row0, row0 + in, row0 + 2 * in, row0 + 3 * in, in)
        t += 1
      o += 4
    while o < out do
      val row = w + o * in
      var t = 0
      while t < T do
        val gy = dy(t * out + o)
        if gy != 0f then Simd.axpy(dx, t * in, gy, p, row, in)
        t += 1
      o += 1

  private def layerNorm(p: Array[Float], gAt: Int, bAt: Int, x: Array[Float], xAt: Int, y: Array[Float], yAt: Int,
                        meanOut: Array[Float], rstdOut: Array[Float], t: Int): Unit =
    var s = 0.0
    var i = 0
    while i < d do { s += x(xAt + i); i += 1 }
    val mean = (s / d).toFloat
    var v = 0.0
    i = 0
    while i < d do { val c = x(xAt + i) - mean; v += c * c; i += 1 }
    val rstd = (1.0 / math.sqrt(v / d + 1e-5)).toFloat
    meanOut(t) = mean
    rstdOut(t) = rstd
    i = 0
    while i < d do
      y(yAt + i) = (x(xAt + i) - mean) * rstd * p(gAt + i) + p(bAt + i)
      i += 1

  private def layerNormBackward(p: Array[Float], g: Array[Float], gAt: Int, bAt: Int, x: Array[Float], xAt: Int,
                                mean: Float, rstd: Float, dy: Array[Float], dyAt: Int, dx: Array[Float], dxAt: Int): Unit =
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
    val meanDyHat = (sumDyHat / d).toFloat
    val meanDyHatXhat = (sumDyHatXhat / d).toFloat
    i = 0
    while i < d do
      val xhat = (x(xAt + i) - mean) * rstd
      val dyh = dy(dyAt + i) * p(gAt + i)
      dx(dxAt + i) += rstd * (dyh - meanDyHat - xhat * meanDyHatXhat)
      i += 1

  // ---- 順伝播 ----

  /** ids を通して各位置のロジットを ws に入れる。 */
  def forward(p: Array[Float], ids: Array[Int], ws: Workspace): Unit = forward(p, ids, 0, ids.length, ws)

  /** tokens(start until start+T) を入力として順伝播する（配列の切り出しをしない版）。 */
  def forward(p: Array[Float], tokens: Array[Int], start: Int, T: Int, ws: Workspace): Unit =
    require(T >= 1 && T <= cfg.context && T <= ws.T, s"系列長 $T が不正")
    val L = layout
    val x0 = ws.x(0)
    var t = 0
    while t < T do
      val tokAt = L.tok + tokens(start + t) * d
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
      denseAll(p, ly.wq, ly.bq, d, d, a, ws.q(l), T, ws.out4)
      denseAll(p, ly.wk, ly.bk, d, d, a, ws.k(l), T, ws.out4)
      denseAll(p, ly.wv, ly.bv, d, d, a, ws.v(l), T, ws.out4)
      if cfg.isLinear then linearAttention(ws.q(l), ws.k(l), ws.v(l), ws.fq(l), ws.fk(l), ws.den(l), ws.o(l), T, ws)
      else attention(ws.q(l), ws.k(l), ws.v(l), ws.prob(l), ws.o(l), T)
      val x1 = ws.x1(l)
      val u = ws.u(l)
      val next = ws.x(l + 1)
      denseAll(p, ly.wo, ly.bo, d, d, ws.o(l), tmp, T, ws.out4)
      var i = 0
      while i < T * d do { x1(i) = x(i) + tmp(i); i += 1 }
      t = 0
      while t < T do
        layerNorm(p, ly.ln2g, ly.ln2b, x1, t * d, ws.b(l), t * d, ws.mean2(l), ws.rstd2(l), t)
        t += 1
      denseAll(p, ly.w1, ly.b1, d, cfg.ff, ws.b(l), u, T, ws.out4)
      var j = 0
      while j < T * cfg.ff do { rAll(j) = math.max(0f, u(j)); j += 1 }
      denseAll(p, ly.w2, ly.b2, cfg.ff, d, rAll, tmp, T, ws.out4)
      i = 0
      while i < T * d do { next(i) = x1(i) + tmp(i); i += 1 }
    // 最終 LN と出力（埋め込みと共有）。語彙を外側、トークンを内側に
    val xl = ws.x(cfg.layers)
    t = 0
    while t < T do
      layerNorm(p, L.lnfg, L.lnfb, xl, t * d, ws.f, t * d, ws.meanf, ws.rstdf, t)
      t += 1
    denseAll(p, L.tok, L.headb, d, cfg.vocab, ws.f, ws.logits, T, ws.out4)

  /** 因果的注意。各ヘッド、各位置 i について、j ≤ i との内積 → softmax → v の混合。 */
  private def attention(q: Array[Float], k: Array[Float], v: Array[Float], prob: Array[Float], o: Array[Float], T: Int): Unit =
    java.util.Arrays.fill(o, 0, T * d, 0f)
    var h = 0
    while h < cfg.heads do
      val off = h * hd
      var i = 0
      while i < T do
        val pAt = (h * T + i) * T
        var maxS = Float.NegativeInfinity
        var j = 0
        while j <= i do
          val s = Simd.dot(q, i * d + off, k, j * d + off, hd) * scale
          prob(pAt + j) = s
          if s > maxS then maxS = s
          j += 1
        var total = 0.0
        j = 0
        while j <= i do { val e = math.exp((prob(pAt + j) - maxS).toDouble); prob(pAt + j) = e.toFloat; total += e; j += 1 }
        val inv = (1.0 / total).toFloat
        j = 0
        while j <= i do
          val w = prob(pAt + j) * inv
          prob(pAt + j) = w
          Simd.axpy(o, i * d + off, w, v, j * d + off, hd)
          j += 1
        i += 1
      h += 1


  // ---- 線形注意（因果的、ヘッドごとの減衰つき） ----
  //   φ(x) = elu(x) + 1（正の特徴写像）
  //   s_i = γ s_{i-1} + φ(k_i) v_iᵀ   （hd×hd の状態）,  z_i = γ z_{i-1} + φ(k_i)
  //   o_i = φ(q_i)ᵀ s_i / (φ(q_i)·z_i + ε)
  // 状態の大きさが系列長に依らないので、計算量は系列長に対して線形。

  private val eps: Float = 1e-6f
  private def phi(x: Float): Float = if x > 0f then x + 1f else math.exp(x.toDouble).toFloat
  private def phiPrime(x: Float): Float = if x > 0f then 1f else math.exp(x.toDouble).toFloat

  private def linearAttention(q: Array[Float], k: Array[Float], v: Array[Float], fq: Array[Float], fk: Array[Float],
                              den: Array[Float], o: Array[Float], T: Int, ws: Workspace): Unit =
    val s = ws.sState; val z = ws.zState; val a = ws.aTmp
    var h = 0
    while h < cfg.heads do
      val g = cfg.gamma(h)
      val off = h * hd
      java.util.Arrays.fill(s, 0f); java.util.Arrays.fill(z, 0f)
      var i = 0
      while i < T do
        val at = i * d + off
        var c = 0
        while c < hd do { fq(at + c) = phi(q(at + c)); fk(at + c) = phi(k(at + c)); c += 1 }
        // 状態を減衰させてから今のトークンを足す
        var r = 0
        while r < hd do
          val fkr = fk(at + r)
          z(r) = g * z(r) + fkr
          val row = r * hd
          c = 0
          while c < hd do { s(row + c) = g * s(row + c); c += 1 }
          Simd.axpy(s, row, fkr, v, at, hd)
          r += 1
        // a = φ(q)ᵀ s,  dn = φ(q)·z + ε
        java.util.Arrays.fill(a, 0f)
        r = 0
        while r < hd do { Simd.axpy(a, 0, fq(at + r), s, r * hd, hd); r += 1 }
        val dn = Simd.dot(fq, at, z, 0, hd) + eps
        den(h * T + i) = dn
        val inv = 1f / dn
        c = 0
        while c < hd do { o(at + c) = a(c) * inv; c += 1 }
        i += 1
      h += 1

  private def linearAttentionBackward(q: Array[Float], k: Array[Float], v: Array[Float], fq: Array[Float], fk: Array[Float],
                                      den: Array[Float], o: Array[Float], dO: Array[Float],
                                      dq: Array[Float], dk: Array[Float], dv: Array[Float], T: Int, ws: Workspace): Unit =
    val s = ws.sState; val z = ws.zState; val da = ws.daAll; val dden = ws.ddenAll
    var h = 0
    while h < cfg.heads do
      val g = cfg.gamma(h)
      val off = h * hd
      // pass A（前向き）: 状態を再計算しながら dφq を出し、da と dden を保存する
      java.util.Arrays.fill(s, 0f); java.util.Arrays.fill(z, 0f)
      var i = 0
      while i < T do
        val at = i * d + off
        var r = 0
        while r < hd do
          val fkr = fk(at + r)
          z(r) = g * z(r) + fkr
          val row = r * hd
          var c = 0
          while c < hd do { s(row + c) = g * s(row + c); c += 1 }
          Simd.axpy(s, row, fkr, v, at, hd)
          r += 1
        val dn = den(h * T + i)
        val inv = 1f / dn
        // a = o * dn, da = dO / dn, dden = -Σ dO·a / dn² = -Σ dO·o / dn
        var dotDoO = 0f
        var c = 0
        while c < hd do { da(at + c) = dO(at + c) * inv; dotDoO += dO(at + c) * o(at + c); c += 1 }
        val dd = -dotDoO * inv
        dden(h * T + i) = dd
        // dφq[r] = Σ_c da[c] s[r][c] + dd z[r]  → dq = dφq · φ'(q)
        r = 0
        while r < hd do
          val dfq = Simd.dot(s, r * hd, da, at, hd) + dd * z(r)
          dq(at + r) += dfq * phiPrime(q(at + r))
          r += 1
        i += 1
      // pass B（後ろ向き）: DS_j = γ DS_{j+1} + φq_j da_jᵀ, DZ_j = γ DZ_{j+1} + dden_j φq_j
      java.util.Arrays.fill(s, 0f); java.util.Arrays.fill(z, 0f)
      i = T - 1
      while i >= 0 do
        val at = i * d + off
        val dd = dden(h * T + i)
        var r = 0
        while r < hd do
          val fqr = fq(at + r)
          z(r) = g * z(r) + dd * fqr
          val row = r * hd
          var c = 0
          while c < hd do { s(row + c) = g * s(row + c); c += 1 }
          Simd.axpy(s, row, fqr, da, at, hd)
          r += 1
        // dφk[r] = Σ_c DS[r][c] v[c] + DZ[r] → dk;  dv[c] = Σ_r DS[r][c] φk[r]
        r = 0
        while r < hd do
          val dfk = Simd.dot(s, r * hd, v, at, hd) + z(r)
          dk(at + r) += dfk * phiPrime(k(at + r))
          Simd.axpy(dv, at, fk(at + r), s, r * hd, hd)
          r += 1
        i -= 1
      h += 1

  /** 損失だけ（各位置の交差エントロピーの平均）。ロジットは書き換えない。逆伝播も勾配配列も使わない。 */
  def lossFromLogits(ws: Workspace, T: Int, tokens: Array[Int], targetStart: Int): Double =
    var loss = 0.0
    var t = 0
    while t < T do
      val at = t * cfg.vocab
      var maxL = Float.NegativeInfinity
      var vIdx = 0
      while vIdx < cfg.vocab do { if ws.logits(at + vIdx) > maxL then maxL = ws.logits(at + vIdx); vIdx += 1 }
      var total = 0.0
      vIdx = 0
      while vIdx < cfg.vocab do { total += math.exp((ws.logits(at + vIdx) - maxL).toDouble); vIdx += 1 }
      loss += math.log(total) + maxL - ws.logits(at + tokens(targetStart + t))
      t += 1
    loss / T

  /** 評価用: 順伝播だけで損失を出す。 */
  def lossOnly(p: Array[Float], tokens: Array[Int], start: Int, length: Int, ws: Workspace): Double =
    val T = length - 1
    forward(p, tokens, start, T, ws)
    lossFromLogits(ws, T, tokens, start + 1)

  def lossOnly(p: Array[Float], window: Array[Int], ws: Workspace): Double = lossOnly(p, window, 0, window.length, ws)

  /** 損失（各位置の交差エントロピーの平均）と、ロジットの勾配（ws.logits を上書き）。 */
  def lossAndLogitGrad(ws: Workspace, T: Int, targets: Array[Int]): Double = lossAndLogitGrad(ws, T, targets, 0)

  def lossAndLogitGrad(ws: Workspace, T: Int, tokens: Array[Int], targetStart: Int): Double =
    var loss = 0.0
    var t = 0
    while t < T do
      val at = t * cfg.vocab
      var maxL = Float.NegativeInfinity
      var vIdx = 0
      while vIdx < cfg.vocab do { if ws.logits(at + vIdx) > maxL then maxL = ws.logits(at + vIdx); vIdx += 1 }
      var total = 0.0
      vIdx = 0
      while vIdx < cfg.vocab do { total += math.exp((ws.logits(at + vIdx) - maxL).toDouble); vIdx += 1 }
      val logZ = math.log(total) + maxL
      val target = tokens(targetStart + t)
      loss += logZ - ws.logits(at + target)
      vIdx = 0
      while vIdx < cfg.vocab do
        val prob = math.exp(ws.logits(at + vIdx) - logZ)
        ws.logits(at + vIdx) = ((prob - (if vIdx == target then 1.0 else 0.0)) / T).toFloat
        vIdx += 1
      t += 1
    loss / T

  // ---- 逆伝播 ----

  /** forward と lossAndLogitGrad のあとに呼ぶ。勾配を g に足し込む。 */
  def backward(p: Array[Float], g: Array[Float], ids: Array[Int], ws: Workspace): Unit = backward(p, g, ids, 0, ids.length, ws)

  def backward(p: Array[Float], g: Array[Float], tokens: Array[Int], start: Int, T: Int, ws: Workspace): Unit =
    val L = layout
    val n = T * d
    val dx = ws.dx
    val df = ws.df
    java.util.Arrays.fill(dx, 0, n, 0f)
    java.util.Arrays.fill(df, 0, n, 0f)
    // 出力ヘッド（埋め込みと共有）
    denseBackwardAll(p, g, L.tok, L.headb, d, cfg.vocab, ws.f, ws.logits, df, T)
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
      while j < T * cfg.ff do { rAll(j) = math.max(0f, u(j)); j += 1 }
      java.util.Arrays.fill(duAll, 0, T * cfg.ff, 0f)
      denseBackwardAll(p, g, ly.w2, ly.b2, cfg.ff, d, rAll, dx, duAll, T)
      j = 0
      while j < T * cfg.ff do { if u(j) <= 0f then duAll(j) = 0f; j += 1 }
      java.util.Arrays.fill(tmp, 0, n, 0f)
      denseBackwardAll(p, g, ly.w1, ly.b1, d, cfg.ff, ws.b(l), duAll, tmp, T)
      t = 0
      while t < T do
        layerNormBackward(p, g, ly.ln2g, ly.ln2b, ws.x1(l), t * d, ws.mean2(l)(t), ws.rstd2(l)(t), tmp, t * d, dx1, t * d)
        t += 1
      // 注意: x1 = x + Wo o + bo
      java.util.Arrays.fill(dO, 0, n, 0f)
      denseBackwardAll(p, g, ly.wo, ly.bo, d, d, ws.o(l), dx1, dO, T)
      java.util.Arrays.fill(dq, 0, n, 0f); java.util.Arrays.fill(dk, 0, n, 0f); java.util.Arrays.fill(dv, 0, n, 0f)
      if cfg.isLinear then linearAttentionBackward(ws.q(l), ws.k(l), ws.v(l), ws.fq(l), ws.fk(l), ws.den(l), ws.o(l), dO, dq, dk, dv, T, ws)
      else attentionBackward(ws.q(l), ws.k(l), ws.v(l), ws.prob(l), dO, dq, dk, dv, ws.dp, T)
      java.util.Arrays.fill(da, 0, n, 0f)
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
      val tokAt = L.tok + tokens(start + t) * d
      val posAt = L.pos + t * d
      var i = 0
      while i < d do
        g(tokAt + i) += dx(t * d + i)
        g(posAt + i) += dx(t * d + i)
        i += 1
      t += 1

  private def attentionBackward(q: Array[Float], k: Array[Float], v: Array[Float], prob: Array[Float],
                                dO: Array[Float], dq: Array[Float], dk: Array[Float], dv: Array[Float], dp: Array[Float], T: Int): Unit =
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
          dp(j) = Simd.dot(dO, i * d + off, v, j * d + off, hd)
          j += 1
        var dotSum = 0.0
        j = 0
        while j <= i do { dotSum += prob(pAt + j) * dp(j); j += 1 }
        val ds0 = dotSum.toFloat
        j = 0
        while j <= i do
          val ds = prob(pAt + j) * (dp(j) - ds0) * scale
          Simd.axpy(dq, i * d + off, ds, k, j * d + off, hd)
          Simd.axpy(dk, j * d + off, ds, q, i * d + off, hd)
          j += 1
        i += 1
      h += 1

  /** 1 系列の損失と勾配（g に足し込む）。作業領域を渡す版。 */
  def lossAndGrad(p: Array[Float], g: Array[Float], window: Array[Int], ws: Workspace): Double =
    lossAndGrad(p, g, window, 0, window.length, ws)

  /** tokens(start until start+length) を 1 窓として、損失と勾配（g に足し込む）。 */
  def lossAndGrad(p: Array[Float], g: Array[Float], tokens: Array[Int], start: Int, length: Int, ws: Workspace): Double =
    val T = length - 1
    forward(p, tokens, start, T, ws)
    val loss = lossAndLogitGrad(ws, T, tokens, start + 1)
    backward(p, g, tokens, start, T, ws)
    loss

  def lossAndGrad(p: Array[Float], g: Array[Float], window: Array[Int]): Double =
    lossAndGrad(p, g, window, workspace())

  /** 最後の位置のロジット（生成用）。 */
  def lastLogits(p: Array[Float], ids: Array[Int], ws: Workspace): Array[Float] =
    forward(p, ids, ws)
    ws.logits.slice((ids.length - 1) * cfg.vocab, ids.length * cfg.vocab)

  def parameterCount: Int = layout.size
