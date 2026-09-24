package slm

import jdk.incubator.vector.{FloatVector, VectorOperators, VectorSpecies}

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
    var s = acc0.add(acc1).reduceLanes(VectorOperators.ADD)
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
    var s0 = a0.reduceLanes(VectorOperators.ADD); var s1 = a1.reduceLanes(VectorOperators.ADD)
    var s2 = a2.reduceLanes(VectorOperators.ADD); var s3 = a3.reduceLanes(VectorOperators.ADD)
    while i < n do
      val wi = w(wAt + i)
      s0 += wi * x(x0At + i); s1 += wi * x(x1At + i); s2 += wi * x(x2At + i); s3 += wi * x(x3At + i)
      i += 1
    out(0) = s0; out(1) = s1; out(2) = s2; out(3) = s3

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
