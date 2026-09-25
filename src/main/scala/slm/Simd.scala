package slm

import jdk.incubator.vector.{FloatVector, VectorShuffle, VectorSpecies}

/** SIMD（1 命令で複数の数を同時に扱う）版の内積と axpy。Float32、JDK の Vector API。
  *
  * 行列ではない。「同じ計算を同時にやる」を命令レベルでやっているだけで、意味は while ループと同じ。
  * 実行時に `--add-modules=jdk.incubator.vector` が要る。
  *
  * `dot4` / `axpy4` / `spread4` は「重み 1 行の読み込みをトークン 4 本で共有する」ためのもの（レジスタブロッキング）。
  */
object Simd:
  private val S: VectorSpecies[java.lang.Float] = FloatVector.SPECIES_PREFERRED
  val lanes: Int = S.length()

  /** 順序を固定した木構造の加算。`reduceLanes(ADD)` は加算順が未規定で JIT の段階により丸めが変わり得るため、
    * 半分ずつ入れ替えて足す（16 → 8 → 4 → 2 → 1）。要素ごとの演算と並べ替えだけなので、どの実行経路でも同じ結果になる。
    */
  private val swaps: Array[VectorShuffle[java.lang.Float]] =
    Iterator.iterate(lanes / 2)(_ / 2).takeWhile(_ >= 1).map(h => VectorShuffle.fromOp(S, (i: Int) => i ^ h)).toArray
  private def orderedSum(v: FloatVector): Float =
    var x = v
    var k = 0
    while k < swaps.length do
      x = x.add(x.rearrange(swaps(k)))
      k += 1
    x.lane(0)

  /** Σ_i a[aAt+i] * b[bAt+i] */
  def dot(a: Array[Float], aAt: Int, b: Array[Float], bAt: Int, n: Int): Float =
    var acc0 = FloatVector.zero(S)
    var acc1 = FloatVector.zero(S)
    var i = 0
    val twice = 2 * lanes
    val bound2 = n - (n % twice)
    while i < bound2 do
      acc0 = FloatVector.fromArray(S, a, aAt + i).fma(FloatVector.fromArray(S, b, bAt + i), acc0)
      acc1 = FloatVector.fromArray(S, a, aAt + i + lanes).fma(FloatVector.fromArray(S, b, bAt + i + lanes), acc1)
      i += twice
    val bound = n - (n % lanes)
    while i < bound do
      acc0 = FloatVector.fromArray(S, a, aAt + i).fma(FloatVector.fromArray(S, b, bAt + i), acc0)
      i += lanes
    var s = orderedSum(acc0.add(acc1))
    while i < n do
      s += a(aAt + i) * b(bAt + i)
      i += 1
    s

  /** 同じ w 行と、x の 4 行（x0At..x3At）との内積を同時に。結果は out(0..3)。 */
  def dot4(w: Array[Float], wAt: Int, x: Array[Float], x0At: Int, x1At: Int, x2At: Int, x3At: Int, n: Int, out: Array[Float]): Unit =
    var a0 = FloatVector.zero(S); var a1 = FloatVector.zero(S); var a2 = FloatVector.zero(S); var a3 = FloatVector.zero(S)
    var i = 0
    val bound = n - (n % lanes)
    while i < bound do
      val wv = FloatVector.fromArray(S, w, wAt + i)
      a0 = FloatVector.fromArray(S, x, x0At + i).fma(wv, a0)
      a1 = FloatVector.fromArray(S, x, x1At + i).fma(wv, a1)
      a2 = FloatVector.fromArray(S, x, x2At + i).fma(wv, a2)
      a3 = FloatVector.fromArray(S, x, x3At + i).fma(wv, a3)
      i += lanes
    var s0 = orderedSum(a0); var s1 = orderedSum(a1)
    var s2 = orderedSum(a2); var s3 = orderedSum(a3)
    while i < n do
      val wi = w(wAt + i)
      s0 += wi * x(x0At + i); s1 += wi * x(x1At + i); s2 += wi * x(x2At + i); s3 += wi * x(x3At + i)
      i += 1
    out(0) = s0; out(1) = s1; out(2) = s2; out(3) = s3


  /** w の 2 行（w0At, w1At）と x の 4 行との内積 8 個を同時に。out(0..3) が w0 と x0..x3、out(4..7) が w1 と x0..x3。 */
  def dot4x2(w: Array[Float], w0At: Int, w1At: Int, x: Array[Float], x0At: Int, x1At: Int, x2At: Int, x3At: Int, n: Int, out: Array[Float]): Unit =
    var a00 = FloatVector.zero(S); var a01 = FloatVector.zero(S); var a02 = FloatVector.zero(S); var a03 = FloatVector.zero(S)
    var a10 = FloatVector.zero(S); var a11 = FloatVector.zero(S); var a12 = FloatVector.zero(S); var a13 = FloatVector.zero(S)
    var i = 0
    val bound = n - (n % lanes)
    while i < bound do
      val w0 = FloatVector.fromArray(S, w, w0At + i)
      val w1 = FloatVector.fromArray(S, w, w1At + i)
      val x0 = FloatVector.fromArray(S, x, x0At + i); a00 = x0.fma(w0, a00); a10 = x0.fma(w1, a10)
      val x1 = FloatVector.fromArray(S, x, x1At + i); a01 = x1.fma(w0, a01); a11 = x1.fma(w1, a11)
      val x2 = FloatVector.fromArray(S, x, x2At + i); a02 = x2.fma(w0, a02); a12 = x2.fma(w1, a12)
      val x3 = FloatVector.fromArray(S, x, x3At + i); a03 = x3.fma(w0, a03); a13 = x3.fma(w1, a13)
      i += lanes
    out(0) = orderedSum(a00); out(1) = orderedSum(a01); out(2) = orderedSum(a02); out(3) = orderedSum(a03)
    out(4) = orderedSum(a10); out(5) = orderedSum(a11); out(6) = orderedSum(a12); out(7) = orderedSum(a13)
    while i < n do
      val w0 = w(w0At + i); val w1 = w(w1At + i)
      out(0) += w0 * x(x0At + i); out(1) += w0 * x(x1At + i); out(2) += w0 * x(x2At + i); out(3) += w0 * x(x3At + i)
      out(4) += w1 * x(x0At + i); out(5) += w1 * x(x1At + i); out(6) += w1 * x(x2At + i); out(7) += w1 * x(x3At + i)
      i += 1

  /** y[yAt+i] += alpha * x[xAt+i] */
  def axpy(y: Array[Float], yAt: Int, alpha: Float, x: Array[Float], xAt: Int, n: Int): Unit =
    val av = FloatVector.broadcast(S, alpha)
    var i = 0
    val bound = n - (n % lanes)
    while i < bound do
      FloatVector.fromArray(S, x, xAt + i).fma(av, FloatVector.fromArray(S, y, yAt + i)).intoArray(y, yAt + i)
      i += lanes
    while i < n do
      y(yAt + i) += alpha * x(xAt + i)
      i += 1

  /** y += a0*x0 + a1*x1 + a2*x2 + a3*x3（y の読み書きを 4 本ぶんまとめる） */
  def axpy4(y: Array[Float], yAt: Int, a0: Float, a1: Float, a2: Float, a3: Float,
            x: Array[Float], x0At: Int, x1At: Int, x2At: Int, x3At: Int, n: Int): Unit =
    val v0 = FloatVector.broadcast(S, a0); val v1 = FloatVector.broadcast(S, a1)
    val v2 = FloatVector.broadcast(S, a2); val v3 = FloatVector.broadcast(S, a3)
    var i = 0
    val bound = n - (n % lanes)
    while i < bound do
      var acc = FloatVector.fromArray(S, y, yAt + i)
      acc = FloatVector.fromArray(S, x, x0At + i).fma(v0, acc)
      acc = FloatVector.fromArray(S, x, x1At + i).fma(v1, acc)
      acc = FloatVector.fromArray(S, x, x2At + i).fma(v2, acc)
      acc = FloatVector.fromArray(S, x, x3At + i).fma(v3, acc)
      acc.intoArray(y, yAt + i)
      i += lanes
    while i < n do
      y(yAt + i) += a0 * x(x0At + i) + a1 * x(x1At + i) + a2 * x(x2At + i) + a3 * x(x3At + i)
      i += 1

  /** y0 += a00 x0 + a01 x1 + a02 x2 + a03 x3、y1 += a10 x0 + a11 x1 + a12 x2 + a13 x3（x の 4 本の読み込みを 2 本の出力で共有）。
    * 出力 1 本ぶんの積和の順序と端数処理は `axpy4` と同じ（結果は bit 一致）。
    */
  def axpy4x2(y: Array[Float], y0At: Int, y1At: Int,
              a00: Float, a01: Float, a02: Float, a03: Float, a10: Float, a11: Float, a12: Float, a13: Float,
              x: Array[Float], x0At: Int, x1At: Int, x2At: Int, x3At: Int, n: Int): Unit =
    val v00 = FloatVector.broadcast(S, a00); val v01 = FloatVector.broadcast(S, a01)
    val v02 = FloatVector.broadcast(S, a02); val v03 = FloatVector.broadcast(S, a03)
    val v10 = FloatVector.broadcast(S, a10); val v11 = FloatVector.broadcast(S, a11)
    val v12 = FloatVector.broadcast(S, a12); val v13 = FloatVector.broadcast(S, a13)
    var i = 0
    val bound = n - (n % lanes)
    while i < bound do
      val x0 = FloatVector.fromArray(S, x, x0At + i)
      val x1 = FloatVector.fromArray(S, x, x1At + i)
      val x2 = FloatVector.fromArray(S, x, x2At + i)
      val x3 = FloatVector.fromArray(S, x, x3At + i)
      var acc0 = FloatVector.fromArray(S, y, y0At + i)
      var acc1 = FloatVector.fromArray(S, y, y1At + i)
      acc0 = x0.fma(v00, acc0); acc1 = x0.fma(v10, acc1)
      acc0 = x1.fma(v01, acc0); acc1 = x1.fma(v11, acc1)
      acc0 = x2.fma(v02, acc0); acc1 = x2.fma(v12, acc1)
      acc0 = x3.fma(v03, acc0); acc1 = x3.fma(v13, acc1)
      acc0.intoArray(y, y0At + i)
      acc1.intoArray(y, y1At + i)
      i += lanes
    while i < n do
      y(y0At + i) += a00 * x(x0At + i) + a01 * x(x1At + i) + a02 * x(x2At + i) + a03 * x(x3At + i)
      y(y1At + i) += a10 * x(x0At + i) + a11 * x(x1At + i) + a12 * x(x2At + i) + a13 * x(x3At + i)
      i += 1

  /** y0 += a0*w, y1 += a1*w, y2 += a2*w, y3 += a3*w（w の読み込みを 4 本で共有） */
  def spread4(y: Array[Float], y0At: Int, y1At: Int, y2At: Int, y3At: Int, a0: Float, a1: Float, a2: Float, a3: Float,
              w: Array[Float], wAt: Int, n: Int): Unit =
    val v0 = FloatVector.broadcast(S, a0); val v1 = FloatVector.broadcast(S, a1)
    val v2 = FloatVector.broadcast(S, a2); val v3 = FloatVector.broadcast(S, a3)
    var i = 0
    val bound = n - (n % lanes)
    while i < bound do
      val wv = FloatVector.fromArray(S, w, wAt + i)
      wv.fma(v0, FloatVector.fromArray(S, y, y0At + i)).intoArray(y, y0At + i)
      wv.fma(v1, FloatVector.fromArray(S, y, y1At + i)).intoArray(y, y1At + i)
      wv.fma(v2, FloatVector.fromArray(S, y, y2At + i)).intoArray(y, y2At + i)
      wv.fma(v3, FloatVector.fromArray(S, y, y3At + i)).intoArray(y, y3At + i)
      i += lanes
    while i < n do
      val wi = w(wAt + i)
      y(y0At + i) += a0 * wi; y(y1At + i) += a1 * wi; y(y2At + i) += a2 * wi; y(y3At + i) += a3 * wi
      i += 1
