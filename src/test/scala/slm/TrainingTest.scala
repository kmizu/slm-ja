package slm

import java.nio.file.Files
import scala.util.Random

/** 評価・抽出・集約・AdamW・checkpoint・再開の検証。 */
class TrainingTest extends munit.FunSuite:

  private val cfg = Config(vocab = 11, d = 16, heads = 2, layers = 2, context = 8, ff = 24)
  private def tinyRun(steps: Int = 4096) = RunConfig(cfg, batch = 6, steps = steps, warmup = 3, peakLr = 1e-2, floorLr = 1e-3, weightDecay = 0.1, clip = 1.0, seed = 7L)
  private def tokensOf(n: Int, seed: Int): Array[Int] = { val r = new Random(seed); Array.fill(n)(r.nextInt(cfg.vocab)) }
  private def newTrainer(pool: WorkerPool, tokens: Array[Int], run: RunConfig = tinyRun()): Trainer =
    val split = (tokens.length * 0.8).toInt
    new Trainer(run, new Model(cfg), tokens, split, split, tokens.length, pool, new Tokenizer("あいうえおかきくけこ".toVector), _ => ())

  test("lossOnly は lossAndGrad と同じ損失を返し、params と勾配配列に触らない") {
    val model = new Model(cfg)
    val p = model.layout.init(new Random(1))
    val before = p.clone()
    val window = Array(1, 3, 2, 6, 5, 0, 4, 9, 10)
    val ws = model.workspace()
    val g = new Array[Float](p.length)
    val withGrad = model.lossAndGrad(p, g, window, ws)
    val only = model.lossOnly(p, window, ws)
    assertEqualsDouble(only, withGrad, 1e-9)
    assertEquals(p.toVector, before.toVector)
    // offset 版も同じ
    val padded = Array(0, 0) ++ window ++ Array(0)
    assertEqualsDouble(model.lossOnly(p, padded, 2, window.length, ws), withGrad, 1e-9)
  }

  test("Sampler は seed と step から決定的で、開始位置は 0 <= s < N - T") {
    val a = Sampler.windowStarts(7L, 5L, 32, 1000, 8)
    val b = Sampler.windowStarts(7L, 5L, 32, 1000, 8)
    assertEquals(a.toVector, b.toVector)
    assert(a.forall(s => s >= 0 && s < 1000 - 8))
    assertNotEquals(Sampler.windowStarts(7L, 6L, 32, 1000, 8).toVector, a.toVector)
    assertEquals(Sampler.numberOfStarts(9, 8), 1) // N = T+1 でも唯一の窓が選べる
    assertEquals(Sampler.windowStarts(1L, 0L, 3, 9, 8).toVector, Vector(0, 0, 0))
    intercept[IllegalArgumentException](Sampler.windowStarts(1L, 0L, 3, 8, 8))
  }

  test("EvalWindows: 長さ T / T+1 / T+2 の領域から 0 / 1 / 2 窓、多いときは均等配置") {
    assertEquals(EvalWindows.starts(100, 108, 8, 64).length, 0)
    assertEquals(EvalWindows.starts(100, 109, 8, 64).toVector, Vector(100))
    assertEquals(EvalWindows.starts(100, 110, 8, 64).toVector, Vector(100, 101))
    val many = EvalWindows.starts(0, 10000, 8, 64)
    assertEquals(many.length, 64)
    assertEquals(many.head, 0); assertEquals(many.last, 10000 - 8 - 1)
    assert(many.sliding(2).forall(w => w(1) > w(0)))
  }

  test("DecayRanges は [0,P) を過不足なく分け、重み行列だけを減衰対象にする") {
    val layout = new Layout(cfg)
    val decay = DecayRanges.of(layout)
    var covered = 0
    var weights = 0
    decay.foreachSegment(0, layout.size) { (s, e, d) => assertEquals(s, covered); covered = e; if d then weights += e - s }
    assertEquals(covered, layout.size)
    assertEquals(weights, cfg.layers * (4 * cfg.d * cfg.d + 2 * cfg.d * cfg.ff))
    // 部分区間でも同じ
    var c2 = 100
    decay.foreachSegment(100, 300) { (s, e, _) => assertEquals(s, c2); c2 = e }
    assertEquals(c2, 300)
  }

  test("区間並列の集約と AdamW は、直列（旧式: clip を書き戻して更新）と一致する") {
    val layout = new Layout(cfg)
    val P = layout.size
    val r = new Random(3)
    val buffers = Array.fill(3)(Array.fill(P)((r.nextGaussian() * 0.5).toFloat))
    val invBatch = (1.0 / 6).toFloat
    // 直列
    val gSerial = new Array[Float](P)
    for buf <- buffers do { var i = 0; while i < P do { gSerial(i) += buf(i) * invBatch; i += 1 } }
    val normSerial = math.sqrt(Optimizer.sumSquares(gSerial, 0, P))
    // 並列（区間ごと）
    val pool = new WorkerPool(4)
    val gPar = new Array[Float](P)
    val parts = new Array[Double](4)
    pool.ranges(P) { (k, s, e) => parts(k) = Optimizer.reduceRange(buffers, gPar, s, e, invBatch) }
    assertEquals(gPar.toVector, gSerial.toVector)
    assertEqualsDouble(math.sqrt(parts.sum), normSerial, 1e-9)
    // AdamW: 旧式（clip を配列へ書き戻し、全体を直列に）と新式（clipScale を読み込み時に掛け、区間並列）
    val p1 = layout.init(new Random(5)); val p2 = p1.clone()
    val scale = Optimizer.clipScale(normSerial, 0.5)
    val gClipped = gSerial.map(x => x * scale.toFloat)
    val old = new AdamW(P); val nw = new AdamW(P)
    val kOld = old.beginStep(1e-2, 1.0, 0.1)
    old.updateRange(p1, gClipped, 0, P, kOld, DecayRanges.of(layout))
    val kNew = nw.beginStep(1e-2, scale, 0.1)
    pool.ranges(P) { (_, s, e) => nw.updateRange(p2, gSerial, s, e, kNew, DecayRanges.of(layout)) }
    assertEquals(p2.toVector, p1.toVector)
    assertEquals(nw.m.toVector, old.m.toVector)
    assertEquals(nw.step, 1L)
    // 非有限値は集約で止まる
    buffers(1)(10) = Float.NaN
    intercept[ArithmeticException](Optimizer.reduceRange(buffers, gPar, 0, P, invBatch))
    pool.shutdown()
  }

  test("norm=0 でも更新は有限で、clip 比 1.0") {
    val layout = new Layout(cfg)
    val p = layout.init(new Random(1))
    val a = new AdamW(layout.size)
    val k = a.beginStep(1e-2, Optimizer.clipScale(0.0, 1.0), 0.1)
    assertEquals(k.clipScale, 1.0f)
    a.updateRange(p, new Array[Float](layout.size), 0, layout.size, k, DecayRanges.of(layout))
    assert(p.forall(java.lang.Float.isFinite))
  }

  test("WorkerPool: 区間分割は [0,n) を覆い、worker の例外は main に伝わる") {
    val chunks = WorkerPool.split(1000, 4, 16)
    assertEquals(chunks.head._1, 0); assertEquals(chunks.last._2, 1000)
    assert(chunks.sliding(2).forall(w => w(0)._2 == w(1)._1))
    assertEquals(WorkerPool.split(5, 8, 16).toVector, Vector((0, 5)))
    val pool = new WorkerPool(3)
    intercept[IllegalStateException](pool.run(3)(i => if i == 1 then throw new IllegalStateException("boom")))
    pool.shutdown()
  }

  test("連続 10 step と、7 step → 保存 → 別インスタンスで読込 → 3 step が bit 一致する") {
    val tokens = tokensOf(2000, 11)
    val pool = new WorkerPool(2)
    val a = newTrainer(pool, tokens); a.initFresh(7L)
    for _ <- 1 to 10 do a.trainStep()
    val b = newTrainer(pool, tokens); b.initFresh(7L)
    for _ <- 1 to 7 do b.trainStep()
    val dir = Files.createTempDirectory("slm-state")
    val tok = b.tokenizer
    b.meta = Map("split" -> "1600")
    val gen = Checkpoint.saveState(dir, b.state(12.5), tok)
    assert(Files.exists(gen.resolve("COMPLETE")))
    assertEquals(Checkpoint.readPointer(dir.resolve("last")), Some("step-00000007"))
    val (st, tok2) = Checkpoint.loadState(Checkpoint.resolveLatest(dir).get)
    assertEquals(tok2.chars, tok.chars)
    assertEquals(st.run, b.run); assertEquals(st.optimizerStep, 7L); assertEquals(st.meta("split"), "1600")
    val c = newTrainer(pool, tokens); c.restore(st)
    assertEquals(c.stepsDone, 7L); assertEqualsDouble(c.elapsedBefore, 12.5, 1e-9)
    for _ <- 1 to 3 do c.trainStep()
    assertEquals(c.params.toVector, a.params.toVector)
    assertEquals(c.adam.m.toVector, a.adam.m.toVector)
    assertEquals(c.adam.v.toVector, a.adam.v.toVector)
    assertEquals(c.stepsDone, a.stepsDone); assertEquals(c.tokensSeen, a.tokensSeen)
    assertEquals(Sampler.windowStarts(7L, c.stepsDone, 6, 1600, 8).toVector, Sampler.windowStarts(7L, a.stepsDone, 6, 1600, 8).toVector)
    assertEqualsDouble(c.run.lr(c.stepsDone), a.run.lr(a.stepsDone), 0.0)
    pool.shutdown()
    Checkpoint.deleteRecursively(dir)
  }

  test("壊れた世代は読み飛ばし、直前の完全な世代を選ぶ（切り詰め・hash 不一致・pointer の先が無い）") {
    val tokens = tokensOf(1500, 12)
    val pool = new WorkerPool(1)
    val t = newTrainer(pool, tokens); t.initFresh(1L)
    val dir = Files.createTempDirectory("slm-state2")
    t.trainStep(); Checkpoint.saveState(dir, t.state(0), t.tokenizer)
    t.trainStep(); val g2 = Checkpoint.saveState(dir, t.state(0), t.tokenizer)
    assertEquals(Checkpoint.completeGenerations(dir).map(_.getFileName.toString), Vector("step-00000002", "step-00000001"))
    // 切り詰め
    val m = g2.resolve("m.bin")
    Files.write(m, Files.readAllBytes(m).take(100))
    assert(!Checkpoint.isValidGeneration(g2))
    assertEquals(Checkpoint.resolveLatest(dir).map(_.getFileName.toString), Some("step-00000001"))
    intercept[IllegalArgumentException](Checkpoint.loadState(g2))
    // hash 不一致（長さは同じでビットを反転）
    t.trainStep(); val g3 = Checkpoint.saveState(dir, t.state(0), t.tokenizer)
    val v = g3.resolve("v.bin"); val bytes = Files.readAllBytes(v); bytes(5) = (bytes(5) ^ 0x7f).toByte; Files.write(v, bytes)
    assert(!Checkpoint.isValidGeneration(g3))
    assertEquals(Checkpoint.resolveLatest(dir).map(_.getFileName.toString), Some("step-00000001"))
    // pointer の先が無い
    Checkpoint.writePointer(dir.resolve("last"), "step-00000099")
    assertEquals(Checkpoint.resolveLatest(dir).map(_.getFileName.toString), Some("step-00000001"))
    // 書きかけの partial は prune で消える、完全な世代は keep 数だけ残る
    Files.createDirectories(dir.resolve("step-00000005.partial-123"))
    Checkpoint.prune(dir, 1, "")
    assert(!Files.exists(dir.resolve("step-00000005.partial-123")))
    pool.shutdown()
    Checkpoint.deleteRecursively(dir)
  }

  test("Preflight: 10M と 100M のパラメータ数、Layout との一致、不正な設定の拒否") {
    assertEquals(Preflight.parameterCount(Config(4134, 256, 8, 12, 256, 1024)), 10605606L)
    assertEquals(Preflight.parameterCount(Config(4134, 768, 12, 14, 256, 3072)), 102607398L)
    assertEquals(Preflight.parameterCount(Config(4134, 640, 10, 20, 256, 2560)), 101285414L)
    assertEquals(Preflight.parameterCount(cfg), new Layout(cfg).size.toLong)
    val est = Preflight.estimate(Config(4134, 768, 12, 14, 256, 3072))
    assertEqualsDouble(Preflight.mib(est.paramBytes), 391.416, 0.01)
    assertEqualsDouble(Preflight.mib(est.workspaceBytes), 187.095, 0.01)
    intercept[IllegalArgumentException](Preflight.validate(Config(10, 16, 3, 1, 8, 8), 4, 2))
    intercept[IllegalArgumentException](Preflight.validate(cfg, 4, 8))
    intercept[IllegalArgumentException](Train.parseArgs(Array("steps=10", "bogus=1")))
    intercept[IllegalArgumentException](Train.parseArgs(Array("nokey")))
  }

  test("学習率: 予熱の最初と最後、最終 update で floor に一致、途中は単調減少") {
    val run = tinyRun(steps = 100)
    assertEqualsDouble(run.lr(0), run.peakLr / 3, 1e-12)
    assertEqualsDouble(run.lr(2), run.peakLr, 1e-12)
    assertEqualsDouble(run.lr(99), run.floorLr, 1e-12)
    assert((3 until 99).forall(i => run.lr(i) >= run.lr(i + 1)))
  }
