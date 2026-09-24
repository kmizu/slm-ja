package slm

import scala.util.Random

/** Float32 の手書き逆伝播が、Double の参照実装（数値微分で検証）と一致するか。softmax 注意と線形注意の両方。 */
class GradientCheckTest extends munit.FunSuite:

  private val cfg = Config(vocab = 11, d = 16, heads = 2, layers = 2, context = 9, ff = 24)
  private val refCfg = reference.Config(cfg.vocab, cfg.d, cfg.heads, cfg.layers, cfg.context, cfg.ff)
  private val linCfg = cfg.copy(attention = "linear")
  private val linRef = refCfg.copy(attention = "linear")

  private def numericCheck(refCfg: reference.Config): Unit =
    val model = new reference.Model(refCfg)
    val rng = new Random(1)
    val p = model.layout.init(rng)
    for i <- p.indices do p(i) += rng.nextGaussian() * 0.3
    val window = Array(1, 3, 2, 6, 5, 0, 4, 9, 10, 7)
    val g = new Array[Double](p.length)
    model.lossAndGrad(p, g, window)
    val h = 1e-5
    var worst = 0.0
    for i <- Vector.fill(60)(rng.nextInt(p.length)) do
      val saved = p(i)
      p(i) = saved + h; val lp = model.lossAndGrad(p, new Array[Double](p.length), window)
      p(i) = saved - h; val lm = model.lossAndGrad(p, new Array[Double](p.length), window)
      p(i) = saved
      val numeric = (lp - lm) / (2 * h)
      val err = math.abs(numeric - g(i)) / math.max(1e-8, math.abs(numeric) + math.abs(g(i)))
      worst = math.max(worst, err)
      assert(err < 1e-5 || math.abs(numeric - g(i)) < 1e-9, s"index $i: analytic=${g(i)} numeric=$numeric")
    println(f"numeric check (${refCfg.attention}): worst relative error $worst%.2e")

  test("参照実装（Double）は数値微分と 1e-5 で一致する") { numericCheck(refCfg) }
  test("線形注意の参照実装（Double）も数値微分と 1e-5 で一致する") { numericCheck(linRef) }

  private def floatVsDouble(cfg: Config, refCfg: reference.Config): Unit =
    val ref = new reference.Model(refCfg)
    val model = new Model(cfg)
    val rng = new Random(2)
    val pd = ref.layout.init(rng)
    for i <- pd.indices do pd(i) += rng.nextGaussian() * 0.3
    val pf = pd.map(_.toFloat)
    for window <- Vector(Array(1, 3, 2, 6, 5, 0, 4, 9, 10), Array(2, 7, 1, 8, 3, 3, 10, 0, 5, 6)) do
      val gd = new Array[Double](pd.length)
      val gf = new Array[Float](pf.length)
      val ld = ref.lossAndGrad(pd, gd, window)
      val lf = model.lossAndGrad(pf, gf, window)
      assertEqualsDouble(lf, ld, 1e-4 * ld)
      var worst = 0.0
      for i <- pd.indices do
        val err = math.abs(gf(i) - gd(i)) / math.max(1e-3, math.abs(gd(i)))
        worst = math.max(worst, err)
        assert(err < 2e-3, s"index $i: float=${gf(i)} double=${gd(i)}")
      println(f"float vs double (${cfg.attention}): worst relative error $worst%.2e (T=${window.length - 1})")

  test("Float32 版の損失と勾配は Double の参照実装と一致する（系列長が 4 の倍数でも、そうでなくても）") { floatVsDouble(cfg, refCfg) }
  test("線形注意でも Float32 版は Double 参照と一致する") { floatVsDouble(linCfg, linRef) }

  test("パラメータ数が手計算と一致する") {
    val model = new Model(cfg)
    val d = cfg.d
    val perLayer = 2 * d + 4 * (d * d + d) + 2 * d + (cfg.ff * d + cfg.ff) + (d * cfg.ff + d)
    assertEquals(model.parameterCount, cfg.vocab * d + cfg.context * d + cfg.layers * perLayer + 2 * d + cfg.vocab)
  }

  private def causal(cfg: Config): Unit =
    val model = new Model(cfg)
    val p = model.layout.init(new Random(2))
    val a = Array(1, 2, 3, 4, 5)
    val b = Array(1, 2, 3, 4, 9)
    val ca = model.workspace(); model.forward(p, a, ca)
    val cb = model.workspace(); model.forward(p, b, cb)
    for i <- 0 until 4 * cfg.vocab do assertEquals(ca.logits(i), cb.logits(i))

  test("因果的: 未来のトークンを変えても前の位置のロジットは変わらない") { causal(cfg) }
  test("線形注意も因果的") { causal(linCfg) }

  test("チェックポイントは float32 で保存され、旧 float64 も読め、attention 種別も往復する") {
    val dir = java.nio.file.Files.createTempDirectory("ckpt")
    val model = new Model(linCfg)
    val p = model.layout.init(new Random(3))
    val tok = new Tokenizer("あいうえおかきくけこ".toVector)
    Checkpoint.save(dir, linCfg, p, tok)
    val (c2, p2, t2) = Checkpoint.load(dir)
    assertEquals(c2, linCfg); assertEquals(p2.toVector, p.toVector); assertEquals(t2.chars, tok.chars)
    val buf = java.nio.ByteBuffer.allocate(p.length * 8).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    p.foreach(x => buf.putDouble(x.toDouble))
    java.nio.file.Files.write(dir.resolve("params.bin"), buf.array())
    assertEquals(Checkpoint.load(dir)._2.toVector, p.toVector)
  }

class TokenizerTest extends munit.FunSuite:
  test("出現回数の少ない文字は UNK になり、往復できる") {
    val t = Tokenizer.fromText("あああいいう", minCount = 2)
    assertEquals(t.vocabSize, 3)
    assertEquals(t.encode("あいう").toVector, Vector(1, 2, 0))
    assertEquals(t.decode(t.encode("あい")), "あい")
  }
