package slm

/** 1 文字ずつの推論（生成用）。
  *
  * `Model.forward` は系列全体を毎回計算するので、生成で 1 文字足すたびに直近 context 文字をやり直すと、
  * 1 文字あたり context 倍の計算がかかる。Decoder は層ごとの状態を持ち回して、足した 1 文字ぶんだけ計算する。
  *   - 線形注意: ヘッドごとの s（hd×hd）と z（hd）を減衰させてから今の文字を足す。1 文字あたりの計算は文脈長に依らない
  *   - softmax 注意: 過去の k と v を溜めておき（KV キャッシュ）、今の q とだけ内積をとる
  *
  * 計算の順序は `Model.forward` と同じにしてあり、出力は系列全体の順伝播の最後の位置と丸め誤差の範囲で一致する。
  * 位置埋め込みは context 個しかないので、position が context に達したら `reset` して窓を詰め直す（呼び出し側の責任）。
  * `pool` を渡すと、密な層をニューロンの区間で分けて並列に計算する（各ニューロンの計算は同じなので結果は 1 スレッド版と bit 一致）。
  */
final class Decoder(val model: Model, p: Array[Float], pool: Option[WorkerPool] = None):
  private val cfg = model.cfg
  private val L = model.layout
  private val d = cfg.d
  private val hd = cfg.headDim
  private val scale = (1.0 / math.sqrt(hd.toDouble)).toFloat
  private val eps = 1e-6f
  require(p.length == L.size, s"重みの数が違う: ${p.length} vs ${L.size}")

  private val x = new Array[Float](d)
  private val a = new Array[Float](d)
  private val q = new Array[Float](d)
  private val k = new Array[Float](d)
  private val v = new Array[Float](d)
  private val o = new Array[Float](d)
  private val tmp = new Array[Float](d)
  private val x1 = new Array[Float](d)
  private val b = new Array[Float](d)
  private val u = new Array[Float](cfg.ff)
  private val f = new Array[Float](d)
  /** 直近の step が返した次の文字の logits（次の step で上書きされる）。 */
  val logits: Array[Float] = new Array[Float](cfg.vocab)

  // 線形注意の状態（層ごと、ヘッドごと）
  private val sState: Array[Array[Float]] = if cfg.isLinear then Array.fill(cfg.layers)(new Array[Float](cfg.heads * hd * hd)) else Array.empty
  private val zState: Array[Array[Float]] = if cfg.isLinear then Array.fill(cfg.layers)(new Array[Float](cfg.heads * hd)) else Array.empty
  private val fq = new Array[Float](hd)
  private val acc = new Array[Float](hd)
  // softmax 注意の KV キャッシュ（層ごと、位置 × d）
  private val kCache: Array[Array[Float]] = if cfg.isLinear then Array.empty else Array.fill(cfg.layers)(new Array[Float](cfg.context * d))
  private val vCache: Array[Array[Float]] = if cfg.isLinear then Array.empty else Array.fill(cfg.layers)(new Array[Float](cfg.context * d))
  private val scores = new Array[Float](cfg.context)

  private var pos = 0

  /** これまでに読んだ文字数（次の文字の位置）。 */
  def position: Int = pos

  /** 文脈長で止まるか。位置埋め込みがある（位置が context 個しかない）か、softmax 注意（KV キャッシュが context 個）なら true。
    * 線形注意 + 位置埋め込みなしなら、状態を持ち回すだけなので何文字でも続けられる。
    */
  val bounded: Boolean = cfg.hasPositions || !cfg.isLinear

  /** 状態を捨てて位置 0 からやり直す。 */
  def reset(): Unit =
    pos = 0
    sState.foreach(java.util.Arrays.fill(_, 0f))
    zState.foreach(java.util.Arrays.fill(_, 0f))

  private lazy val ws: Workspace = model.workspace()

  /** 状態を捨てて `tokens` を位置 0 から一括で読み、最後の文字の次の logits を返す。
    *
    * 学習用の `Model.forward`（64 文字ずつまとめて重みを使い回す速い経路）で順伝播し、その k と v から
    * 線形注意の状態（または KV キャッシュ）を組み立てる。プロンプトの読み込みと、窓の詰め直しに使う。
    */
  def prefill(tokens: Array[Int]): Array[Float] =
    require(tokens.nonEmpty && tokens.length <= cfg.context, s"prefill の長さ ${tokens.length} は 1 以上 ${cfg.context} 以下")
    require(tokens.forall(t => t >= 0 && t < cfg.vocab), "語彙の外の文字 id がある")
    reset()
    val T = tokens.length
    model.forward(p, tokens, 0, T, ws)
    var l = 0
    while l < cfg.layers do
      if cfg.isLinear then
        var h = 0
        while h < cfg.heads do
          var i = 0
          while i < T do { absorb(l, h, ws.k(l), i * d + h * hd, ws.v(l), i * d + h * hd); i += 1 }
          h += 1
      else
        System.arraycopy(ws.k(l), 0, kCache(l), 0, T * d)
        System.arraycopy(ws.v(l), 0, vCache(l), 0, T * d)
      l += 1
    System.arraycopy(ws.logits, (T - 1) * cfg.vocab, logits, 0, cfg.vocab)
    pos = T
    logits

  /** 文字 `token` を位置 `position` に読み、次の文字の logits を返す（返す配列は内部のもので、次の step で上書きされる）。 */
  def step(token: Int): Array[Float] =
    require(!bounded || pos < cfg.context, s"位置 $pos は文脈長 ${cfg.context} を超える。reset して窓を詰め直す")
    require(token >= 0 && token < cfg.vocab, s"文字 id $token が語彙の外")
    val tokAt = L.tok + token * d
    var i = 0
    if cfg.hasPositions then
      val posAt = L.pos + pos * d
      while i < d do { x(i) = p(tokAt + i) + p(posAt + i); i += 1 }
    else
      while i < d do { x(i) = p(tokAt + i); i += 1 }
    var l = 0
    while l < cfg.layers do
      val ly = L.layer(l)
      layerNorm(ly.ln1g, ly.ln1b, x, a)
      dense(ly.wq, ly.bq, d, d, a, q)
      dense(ly.wk, ly.bk, d, d, a, k)
      dense(ly.wv, ly.bv, d, d, a, v)
      if cfg.isLinear then linearAttention(l) else softmaxAttention(l)
      dense(ly.wo, ly.bo, d, d, o, tmp)
      i = 0
      while i < d do { x1(i) = x(i) + tmp(i); i += 1 }
      layerNorm(ly.ln2g, ly.ln2b, x1, b)
      dense(ly.w1, ly.b1, d, cfg.ff, b, u)
      var j = 0
      while j < cfg.ff do { u(j) = math.max(0f, u(j)); j += 1 }
      dense(ly.w2, ly.b2, cfg.ff, d, u, tmp)
      i = 0
      while i < d do { x(i) = x1(i) + tmp(i); i += 1 }
      l += 1
    layerNorm(L.lnfg, L.lnfb, x, f)
    dense(L.tok, L.headb, d, cfg.vocab, f, logits)
    pos += 1
    logits

  /** y[o] = Σ_i W[o][i] x[i] + b[o]。`Model.denseAll` の 1 トークン版。 */
  private def dense(w: Int, bias: Int, in: Int, out: Int, xin: Array[Float], y: Array[Float]): Unit =
    def rows(from: Int, until: Int): Unit =
      var o = from
      while o < until do { y(o) = p(bias + o) + Simd.dot(p, w + o * in, xin, 0, in); o += 1 }
    pool match
      case Some(pl) if out >= 64 => pl.ranges(out)((_, from, until) => rows(from, until))
      case _ => rows(0, out)

  /** `Model.layerNorm` と同じ計算（平均と分散は Double で足す）。 */
  private def layerNorm(gAt: Int, bAt: Int, xin: Array[Float], y: Array[Float]): Unit =
    var s = 0.0
    var i = 0
    while i < d do { s += xin(i); i += 1 }
    val mean = (s / d).toFloat
    var vs = 0.0
    i = 0
    while i < d do { val c = xin(i) - mean; vs += c * c; i += 1 }
    val rstd = (1.0 / math.sqrt(vs / d + 1e-5)).toFloat
    i = 0
    while i < d do { y(i) = (xin(i) - mean) * rstd * p(gAt + i) + p(bAt + i); i += 1 }

  private def phi(z: Float): Float = if z > 0f then z + 1f else math.exp(z.toDouble).toFloat

  /** 層 l・ヘッド h の状態に 1 文字ぶんを足す: s = γ s + φ(k) vᵀ,  z = γ z + φ(k)（`Model.linearAttention` と同じ順序）。 */
  private def absorb(l: Int, h: Int, kArr: Array[Float], kAt: Int, vArr: Array[Float], vAt: Int): Unit =
    val sAll = sState(l); val zAll = zState(l)
    val g = cfg.gamma(h)
    val sAt = h * hd * hd
    val zAt = h * hd
    var r = 0
    while r < hd do
      val fkr = phi(kArr(kAt + r))
      zAll(zAt + r) = g * zAll(zAt + r) + fkr
      val row = sAt + r * hd
      var c = 0
      while c < hd do { sAll(row + c) = g * sAll(row + c); c += 1 }
      Simd.axpy(sAll, row, fkr, vArr, vAt, hd)
      r += 1

  /** 状態に今の文字を足してから読み出す: o = φ(q)ᵀ s / (φ(q)·z + ε)。 */
  private def linearAttention(l: Int): Unit =
    val sAll = sState(l); val zAll = zState(l)
    var h = 0
    while h < cfg.heads do
      val off = h * hd
      val sAt = h * hd * hd
      val zAt = h * hd
      absorb(l, h, k, off, v, off)
      var c = 0
      while c < hd do { fq(c) = phi(q(off + c)); c += 1 }
      var r = 0
      java.util.Arrays.fill(acc, 0f)
      r = 0
      while r < hd do { Simd.axpy(acc, 0, fq(r), sAll, sAt + r * hd, hd); r += 1 }
      val dn = Simd.dot(fq, 0, zAll, zAt, hd) + eps
      val inv = 1f / dn
      c = 0
      while c < hd do { o(off + c) = acc(c) * inv; c += 1 }
      h += 1

  /** 今の k, v をキャッシュに足し、今の q と位置 0..pos の k との内積 → softmax → v の混合。 */
  private def softmaxAttention(l: Int): Unit =
    val kc = kCache(l); val vc = vCache(l)
    System.arraycopy(k, 0, kc, pos * d, d)
    System.arraycopy(v, 0, vc, pos * d, d)
    java.util.Arrays.fill(o, 0f)
    var h = 0
    while h < cfg.heads do
      val off = h * hd
      var maxS = Float.NegativeInfinity
      var j = 0
      while j <= pos do
        val s = Simd.dot(q, off, kc, j * d + off, hd) * scale
        scores(j) = s
        if s > maxS then maxS = s
        j += 1
      var total = 0.0
      j = 0
      while j <= pos do { val e = math.exp((scores(j) - maxS).toDouble); scores(j) = e.toFloat; total += e; j += 1 }
      val inv = (1.0 / total).toFloat
      j = 0
      while j <= pos do
        Simd.axpy(o, off, scores(j) * inv, vc, j * d + off, hd)
        j += 1
      h += 1
