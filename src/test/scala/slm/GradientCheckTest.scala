package slm

import scala.util.Random

/** 手で書いた逆伝播が、数値微分（パラメータを少し動かして損失の差を取る）と一致するか。 */
class GradientCheckTest extends munit.FunSuite:

  private val cfg = Config(vocab = 7, d = 8, heads = 2, layers = 2, context = 6, ff = 12)

  test("全パラメータ種別について、解析勾配と数値勾配が 1e-5 で一致する") {
    val model = new Model(cfg)
    val rng = new Random(1)
    val p = model.layout.init(rng)
    // 初期値が小さすぎると勾配が退化するので少し大きくする
    for i <- p.indices do p(i) += rng.nextGaussian() * 0.3
    val window = Array(1, 3, 2, 6, 5, 0, 4)
    val g = new Array[Double](p.length)
    val loss = model.lossAndGrad(p, g, window)
    assert(loss > 0.0)
    val L = model.layout
    val probes = Vector(L.tok + 3, L.tok + 17, L.pos + 5, L.layer(0).ln1g + 2, L.layer(0).ln1b + 4, L.layer(0).wq + 11,
      L.layer(0).bk + 1, L.layer(0).wv + 30, L.layer(0).wo + 7, L.layer(0).bo + 3, L.layer(0).ln2g + 1, L.layer(0).w1 + 40,
      L.layer(0).b1 + 5, L.layer(0).w2 + 20, L.layer(0).b2 + 6, L.layer(1).wq + 2, L.layer(1).w2 + 9, L.lnfg + 3, L.lnfb + 0,
      L.headb + 2) ++ Vector.fill(30)(rng.nextInt(p.length))
    val h = 1e-5
    var worst = 0.0
    for i <- probes do
      val saved = p(i)
      p(i) = saved + h
      val lp = model.lossAndGrad(p, new Array[Double](p.length), window)
      p(i) = saved - h
      val lm = model.lossAndGrad(p, new Array[Double](p.length), window)
      p(i) = saved
      val numeric = (lp - lm) / (2 * h)
      val err = math.abs(numeric - g(i)) / math.max(1e-8, math.abs(numeric) + math.abs(g(i)))
      worst = math.max(worst, err)
      assert(err < 1e-5 || math.abs(numeric - g(i)) < 1e-9, s"index $i: analytic=${g(i)} numeric=$numeric err=$err")
    println(s"gradient check: worst relative error $worst")
  }

  test("パラメータ数が手計算と一致する") {
    val model = new Model(cfg)
    val d = cfg.d
    val perLayer = 2 * d + 4 * (d * d + d) + 2 * d + (cfg.ff * d + cfg.ff) + (d * cfg.ff + d)
    assertEquals(model.parameterCount, cfg.vocab * d + cfg.context * d + cfg.layers * perLayer + 2 * d + cfg.vocab)
  }

  test("因果的: 未来のトークンを変えても前の位置のロジットは変わらない") {
    val model = new Model(cfg)
    val p = model.layout.init(new Random(2))
    val a = Array(1, 2, 3, 4)
    val b = Array(1, 2, 3, 6)
    val ca = new Cache(cfg, 4); model.forward(p, a, ca)
    val cb = new Cache(cfg, 4); model.forward(p, b, cb)
    for i <- 0 until 3 * cfg.vocab do assertEqualsDouble(ca.logits(i), cb.logits(i), 1e-12)
  }

class TokenizerTest extends munit.FunSuite:
  test("出現回数の少ない文字は UNK になり、往復できる") {
    val t = Tokenizer.fromText("あああいいう", minCount = 2)
    assertEquals(t.vocabSize, 3)
    assertEquals(t.encode("あいう").toVector, Vector(1, 2, 0))
    assertEquals(t.decode(t.encode("あい")), "あい")
  }
