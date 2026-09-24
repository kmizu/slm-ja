package slm.reference

import jdk.incubator.vector.{DoubleVector, VectorOperators, VectorSpecies}

/** SIMD（1 命令で複数の数を同時に扱う）版の内積と axpy。JDK の Vector API を使う。
  *
  * 行列ではない。「同じ計算を同時にやる」を命令レベルでやっているだけで、意味は while ループと同じ。
  * 実行時に `--add-modules=jdk.incubator.vector` が要る。
  */
object Simd:
  private val S: VectorSpecies[java.lang.Double] = DoubleVector.SPECIES_PREFERRED
  val lanes: Int = S.length()

  /** Σ_i a[aAt+i] * b[bAt+i] */
  def dot(a: Array[Double], aAt: Int, b: Array[Double], bAt: Int, n: Int): Double =
    var acc0 = DoubleVector.zero(S)
    var acc1 = DoubleVector.zero(S)
    var i = 0
    val twice = 2 * lanes
    val bound2 = n - (n % twice)
    while i < bound2 do
      acc0 = DoubleVector.fromArray(S, a, aAt + i).fma(DoubleVector.fromArray(S, b, bAt + i), acc0)
      acc1 = DoubleVector.fromArray(S, a, aAt + i + lanes).fma(DoubleVector.fromArray(S, b, bAt + i + lanes), acc1)
      i += twice
    val bound = n - (n % lanes)
    while i < bound do
      acc0 = DoubleVector.fromArray(S, a, aAt + i).fma(DoubleVector.fromArray(S, b, bAt + i), acc0)
      i += lanes
    var s = acc0.add(acc1).reduceLanes(VectorOperators.ADD)
    while i < n do
      s += a(aAt + i) * b(bAt + i)
      i += 1
    s

  /** y[yAt+i] += alpha * x[xAt+i] */
  def axpy(y: Array[Double], yAt: Int, alpha: Double, x: Array[Double], xAt: Int, n: Int): Unit =
    val av = DoubleVector.broadcast(S, alpha)
    var i = 0
    val bound = n - (n % lanes)
    while i < bound do
      DoubleVector.fromArray(S, x, xAt + i).fma(av, DoubleVector.fromArray(S, y, yAt + i)).intoArray(y, yAt + i)
      i += lanes
    while i < n do
      y(yAt + i) += alpha * x(xAt + i)
      i += 1
