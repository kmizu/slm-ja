package slm

import scala.util.Random

/** カーネルを書き換えても、順伝播の logits と逆伝播の勾配が bit 単位で変わらないことの検証。
  *
  * 各要素への積和の順序を保ったまま読み込みの共有だけを変える最適化なら、結果は bit 一致する。
  * 指紋の値は書き換え前のカーネル（dot4x2 / axpy4）で記録したもの。積和の順序を意図して変えたときだけ更新する。
  * C のカーネル（16 レーンの CPU で既定）もこの指紋と一致する。
  * 幅 48・FF 80・語彙 37・文脈 11 は、SIMD の端数（16 レーン）、行の端数（4 行・2 行）、トークンの端数（4 本）の分岐を全部通すため。
  */
class KernelGoldenTest extends munit.FunSuite:

  private def fingerprint(attention: String): (Long, Long) =
    val cfg = Config(vocab = 37, d = 48, heads = 3, layers = 2, context = 11, ff = 80, attention = attention)
    val model = new Model(cfg)
    val p = model.layout.init(new Random(11))
    val r = new Random(12)
    var i = 0
    while i < p.length do { p(i) += (r.nextGaussian() * 0.2).toFloat; i += 1 }
    val tokens = Array.fill(cfg.context + 1)(r.nextInt(cfg.vocab))
    val ws = model.workspace()
    val g = new Array[Float](p.length)
    model.lossAndGrad(p, g, tokens, 0, cfg.context + 1, ws)
    val logits = ws.logits.take(cfg.context * cfg.vocab)
    (hash(logits), hash(g))

  private def hash(a: Array[Float]): Long =
    var h = 1125899906842597L
    var i = 0
    while i < a.length do { h = 31 * h + java.lang.Float.floatToRawIntBits(a(i)); i += 1 }
    h

  /** レーン数ごとの指紋。16 レーン（AVX-512）と 8 レーン（AVX2、Ryzen 9 5900X など）では合計の畳み方が違うので値も違う。
    * 8 レーンの値は `-XX:MaxVectorSize=32` で記録した。ベクトル演算の結果はレーンごとに決まるので、同じレーン数ならどの CPU でも同じ値になる。
    */
  private val expected: Map[(Int, String), (Long, Long)] = Map(
    (16, "linear") -> (7889698432477047824L, -3943957894805727470L),
    (16, "softmax") -> (-2060405393796565635L, 4813400518847319763L),
    (8, "linear") -> (-6233283105396001394L, 1839746162347260054L),
    (8, "softmax") -> (-7126880552004492285L, 3714910034561462412L))

  for attention <- Seq("linear", "softmax") do
    test(s"$attention: 順伝播の logits と勾配の指紋が書き換え前のカーネルと一致する") {
      val key = (Simd.lanes, attention)
      assume(expected.contains(key), s"${Simd.lanes} レーンの指紋は記録していない")
      assertEquals(fingerprint(attention), expected(key))
    }
