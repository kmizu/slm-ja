package slm

import java.nio.file.Files
import scala.util.Random

/** 位置埋め込みなし（`positions=none`）の検証。位置の情報は線形注意の減衰（または softmax 注意の因果マスク）だけが担う。 */
class PositionsTest extends munit.FunSuite:

  private val chars = "あいうえおかきくけこ".toVector
  private val cfg = Config(vocab = chars.length + 1, d = 16, heads = 2, layers = 2, context = 8, ff = 24, attention = "linear", positions = "none")

  private def paramsOf(c: Config, seed: Int): Array[Float] =
    val p = new Layout(c).init(new Random(seed))
    val r = new Random(seed + 1)
    var i = 0
    while i < p.length do { p(i) += (r.nextGaussian() * 0.3).toFloat; i += 1 }
    p

  private def close(a: Array[Float], b: Array[Float]): Boolean =
    a.length == b.length && a.indices.forall(i => math.abs(a(i) - b(i)) <= 1e-4f * (1f + math.abs(b(i))))

  test("パラメータ数は位置埋め込み（context × d）の分だけ減り、Preflight の式とも一致する") {
    val learned = cfg.copy(positions = "learned")
    assertEquals(new Model(cfg).parameterCount, new Model(learned).parameterCount - cfg.context * cfg.d)
    assertEquals(Preflight.parameterCount(cfg), new Layout(cfg).size.toLong)
    assertEquals(Preflight.parameterCount(learned), new Layout(learned).size.toLong)
    intercept[IllegalArgumentException](cfg.copy(positions = "rope"))
  }

  test("重みは文脈長に依らず、同じ重みなら文脈長の違う模型でも同じ出力になる") {
    val long = cfg.copy(context = 16)
    assertEquals(new Layout(long).size, new Layout(cfg).size)
    val p = paramsOf(cfg, 3)
    val r = new Random(4)
    val tokens = Array.fill(cfg.context)(r.nextInt(cfg.vocab))
    val a = new Model(cfg); val wa = a.workspace()
    val b = new Model(long); val wb = b.workspace()
    a.forward(p, tokens, wa)
    b.forward(p, tokens, wb)
    assertEquals(wa.logits.take(cfg.context * cfg.vocab).toVector, wb.logits.take(cfg.context * cfg.vocab).toVector)
  }

  test("線形注意 + 位置埋め込みなし: 1 文字ずつの推論は文脈長を超えても続けられ、長い文脈の順伝播と一致する") {
    val p = paramsOf(cfg, 5)
    val n = 3 * cfg.context
    val r = new Random(6)
    val tokens = Array.fill(n)(r.nextInt(cfg.vocab))
    val dec = new Decoder(new Model(cfg), p)
    assert(!dec.bounded)
    val long = new Model(cfg.copy(context = n))
    val ws = long.workspace()
    for t <- 0 until n do
      val step = dec.step(tokens(t)).clone()
      val full = long.lastLogits(p, tokens.take(t + 1), ws)
      assert(close(step, full), s"t=$t で不一致")
    assertEquals(dec.position, n)
  }

  test("位置埋め込みあり、または softmax 注意なら、推論は文脈長で止まる（窓の詰め直しが要る）") {
    assert(new Decoder(new Model(cfg.copy(positions = "learned")), paramsOf(cfg.copy(positions = "learned"), 7)).bounded)
    val soft = cfg.copy(attention = "softmax")
    val dec = new Decoder(new Model(soft), paramsOf(soft, 8))
    assert(dec.bounded)
    for _ <- 0 until soft.context do dec.step(1)
    intercept[IllegalArgumentException](dec.step(1))
  }

  test("線形注意 + 位置埋め込みなし: 生成は窓を詰め直さずに文脈長を超えて続く") {
    val p = paramsOf(cfg, 9)
    val tok = new Tokenizer(chars)
    val text = Generate.sample(new Model(cfg), p, tok, "あい", 40, 0.7, 5, new Random(1))
    assertEquals(text.length, 42)
  }

  test("positions は設定テキストと checkpoint を往復し、項目の無い古い設定は位置埋め込みありとして読める") {
    def parse(text: String) = text.linesIterator.filter(_.contains('=')).map { l => val i = l.indexOf('='); l.take(i) -> l.drop(i + 1) }.toMap
    val run = RunConfig(cfg, batch = 4, steps = 10, warmup = 2, peakLr = 1e-3, floorLr = 1e-4, weightDecay = 0.1, clip = 1.0, seed = 1L)
    assertEquals(RunConfig.fromMap(parse(run.toText)), run)
    val old = parse(run.toText) - "positions"
    assertEquals(RunConfig.fromMap(old).cfg.positions, "learned")
    val dir = Files.createTempDirectory("slm-pos")
    val p = paramsOf(cfg, 10)
    Checkpoint.save(dir, cfg, p, new Tokenizer(chars))
    val (c2, p2, _) = Checkpoint.load(dir)
    assertEquals(c2, cfg)
    assertEquals(p2.toVector, p.toVector)
    Checkpoint.deleteRecursively(dir)
  }
