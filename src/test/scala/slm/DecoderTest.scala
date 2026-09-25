package slm

import scala.util.Random

/** 1 文字ずつの推論（Decoder）が、系列全体を計算し直す順伝播と同じ出力を返すことの検証。 */
class DecoderTest extends munit.FunSuite:

  private val chars = "あいうえおかきくけこ".toVector
  private def cfgOf(attention: String) = Config(vocab = chars.length + 1, d = 16, heads = 2, layers = 2, context = 8, ff = 24, attention = attention)

  /** 初期値のままだと出力がほぼ一様なので、重みを大きめに散らして差が見えるようにする。 */
  private def paramsOf(cfg: Config, seed: Int): Array[Float] =
    val p = new Layout(cfg).init(new Random(seed))
    val r = new Random(seed + 1)
    var i = 0
    while i < p.length do { p(i) += (r.nextGaussian() * 0.3).toFloat; i += 1 }
    p

  private def close(a: Array[Float], b: Array[Float]): Boolean =
    a.length == b.length && a.indices.forall(i => math.abs(a(i) - b(i)) <= 1e-4f * (1f + math.abs(b(i))))

  for attention <- Seq("linear", "softmax") do
    test(s"$attention: 1 文字ずつの推論は、系列全体を計算し直した最後の位置の出力と一致する") {
      val cfg = cfgOf(attention)
      val model = new Model(cfg)
      val p = paramsOf(cfg, 3)
      val r = new Random(5)
      val tokens = Array.fill(cfg.context)(r.nextInt(cfg.vocab))
      val ws = model.workspace()
      val dec = new Decoder(model, p)
      for t <- 0 until cfg.context do
        val step = dec.step(tokens(t)).clone()
        val full = model.lastLogits(p, tokens.take(t + 1), ws)
        assert(close(step, full), s"t=$t で不一致: ${step.take(4).toVector} vs ${full.take(4).toVector}")
        assertEquals(dec.position, t + 1)
    }

    test(s"$attention: 並列版は 1 スレッド版と bit 一致し、reset で最初からやり直せる") {
      val cfg = cfgOf(attention)
      val model = new Model(cfg)
      val p = paramsOf(cfg, 4)
      val pool = new WorkerPool(3)
      val serial = new Decoder(model, p)
      val parallel = new Decoder(model, p, Some(pool))
      val tokens = Seq(1, 4, 2, 9, 0, 7)
      val first = tokens.map(t => serial.step(t).clone())
      tokens.zip(first).foreach { (t, want) => assertEquals(parallel.step(t).toVector, want.toVector) }
      serial.reset()
      assertEquals(serial.position, 0)
      tokens.zip(first).foreach { (t, want) => assertEquals(serial.step(t).toVector, want.toVector) }
      pool.shutdown()
    }

    test(s"$attention: 位置埋め込みの数（context）を超えて進めようとすると止まる") {
      val cfg = cfgOf(attention)
      val dec = new Decoder(new Model(cfg), paramsOf(cfg, 6))
      for _ <- 0 until cfg.context do dec.step(1)
      intercept[IllegalArgumentException](dec.step(1))
      intercept[IllegalArgumentException] { dec.reset(); dec.step(cfg.vocab) }
    }

    test(s"$attention: 文脈長に収まる長さなら、生成は計算し直す版と同じ文字列、超えても指定の文字数を返す") {
      val cfg = cfgOf(attention)
      val model = new Model(cfg)
      val p = paramsOf(cfg, 7)
      val tok = new Tokenizer(chars)
      for seed <- 0 until 5 do
        val fast = Generate.sample(model, p, tok, "あい", 6, 0.7, 5, new Random(seed))
        val slow = Generate.sampleRecompute(model, p, tok, "あい", 6, 0.7, 5, new Random(seed))
        assertEquals(fast, slow)
      val long = Generate.sample(model, p, tok, "あい", 30, 0.7, 5, new Random(1))
      assertEquals(long.length, 32)
    }

    test(s"$attention: 一括の prefill は系列全体の順伝播と bit 一致し、続けて読んだ 1 文字の出力も一致する") {
      val cfg = cfgOf(attention)
      val model = new Model(cfg)
      val p = paramsOf(cfg, 8)
      val r = new Random(9)
      val tokens = Array.fill(cfg.context)(r.nextInt(cfg.vocab))
      val ws = model.workspace()
      for n <- 1 until cfg.context do
        val dec = new Decoder(model, p)
        val pre = dec.prefill(tokens.take(n)).clone()
        assertEquals(pre.toVector, model.lastLogits(p, tokens.take(n), ws).toVector)
        assertEquals(dec.position, n)
        val next = dec.step(tokens(n)).clone()
        assert(close(next, model.lastLogits(p, tokens.take(n + 1), ws)), s"n=$n の続きが不一致")
      intercept[IllegalArgumentException](new Decoder(model, p).prefill(Array.empty[Int]))
      intercept[IllegalArgumentException](new Decoder(model, p).prefill(Array.fill(cfg.context + 1)(1)))
    }
