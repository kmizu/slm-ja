package slm

import java.nio.file.{Files, Path}
import java.util.stream.IntStream
import scala.util.Random

/** 学習: `runMain slm.Train [key=value ...]`
  *
  *   corpus=data/corpus.txt steps=20000 batch=32 lr=1e-3 warmup=300 wd=0.05 minCount=10
  *   d=128 heads=4 layers=3 context=128 ff=512 threads=16 evalEvery=200 out=checkpoints/ja1m resume=
  */
object Train:

  def main(args: Array[String]): Unit =
    val opt = args.map(_.split("=", 2)).collect { case Array(k, v) => k -> v }.toMap
    def str(k: String, d: String) = opt.getOrElse(k, d)
    def int(k: String, d: Int) = opt.get(k).map(_.toInt).getOrElse(d)
    def dbl(k: String, d: Double) = opt.get(k).map(_.toDouble).getOrElse(d)

    val corpusPath = Path.of(str("corpus", "data/corpus.txt"))
    val steps = int("steps", 20000)
    val batch = int("batch", 32)
    val peakLr = dbl("lr", 1e-3)
    val warmup = int("warmup", 300)
    val weightDecay = dbl("wd", 0.05)
    val threads = int("threads", Runtime.getRuntime.availableProcessors())
    val evalEvery = int("evalEvery", 200)
    val outDir = Path.of(str("out", "checkpoints/ja1m"))
    val seed = int("seed", 0)

    val text = Files.readString(corpusPath)
    val tokenizer = Tokenizer.fromText(text, int("minCount", 10))
    val cfg = Config(tokenizer.vocabSize, int("d", 128), int("heads", 4), int("layers", 3), int("context", 128), int("ff", 512))
    val model = new Model(cfg)
    val rng = new Random(seed)

    val ids = tokenizer.encode(text)
    val split = (ids.length * 0.97).toInt
    val train = ids.take(split)
    val valid = ids.drop(split)
    val validWindows = Vector.tabulate(math.min(64, (valid.length - 1) / (cfg.context + 1)))(i => valid.slice(i * (cfg.context + 1), (i + 1) * (cfg.context + 1)))
    println(s"corpus: ${text.length} 文字, 語彙 ${cfg.vocab}, 学習 ${train.length} トークン, 検証 ${valid.length} トークン")
    println(s"model: $cfg, パラメータ ${model.parameterCount} 個, threads=$threads, batch=$batch, steps=$steps")

    val params =
      opt.get("resume") match
        case Some(dir) if dir.nonEmpty =>
          val (c, p, _) = Checkpoint.load(Path.of(dir))
          require(c == cfg, s"設定が違う: $c vs $cfg")
          println(s"resume from $dir"); p
        case _ => model.layout.init(rng)
    val optimizer = new AdamW(params.length)
    val L = model.layout
    // 重み減衰は「重み行列」相当だけ（埋め込みは除く、バイアスと LN も除く）
    val decayMask: Int => Boolean =
      val ranges = L.layer.flatMap(l => Vector((l.wq, cfg.d * cfg.d), (l.wk, cfg.d * cfg.d), (l.wv, cfg.d * cfg.d), (l.wo, cfg.d * cfg.d), (l.w1, cfg.ff * cfg.d), (l.w2, cfg.d * cfg.ff)))
      val mask = new Array[Boolean](params.length)
      for (at, n) <- ranges do java.util.Arrays.fill(mask, at, at + n, true)
      i => mask(i)

    def evaluate(): Double =
      val g = new Array[Double](params.length) // 捨てる
      validWindows.map(w => model.lossAndGrad(params, g, w)).sum / validWindows.length

    Files.createDirectories(outDir)
    val log = Files.newBufferedWriter(outDir.resolve("train.log"), java.nio.charset.StandardCharsets.UTF_8,
      java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND)
    def out(s: String): Unit = { println(s); log.write(s); log.newLine(); log.flush() }

    val start = System.nanoTime()
    var best = Double.MaxValue
    val gradBuffers = Array.fill(threads)(new Array[Double](params.length))
    val grad = new Array[Double](params.length)
    var step = optimizer.step
    var recent = 0.0
    var recentCount = 0
    while step < steps do
      // バッチ: 乱数で窓を切り出す（乱数は逐次に引いて決定的にする）
      val windows = Array.fill(batch) { val s = rng.nextInt(train.length - cfg.context - 1); train.slice(s, s + cfg.context + 1) }
      gradBuffers.foreach(b => java.util.Arrays.fill(b, 0.0))
      val losses = new Array[Double](batch)
      IntStream.range(0, threads).parallel().forEach { th =>
        var b = th
        while b < batch do
          losses(b) = model.lossAndGrad(params, gradBuffers(th), windows(b))
          b += threads
      }
      java.util.Arrays.fill(grad, 0.0)
      for buf <- gradBuffers do
        var i = 0
        while i < grad.length do { grad(i) += buf(i) / batch; i += 1 }
      val norm = Optimizer.clipGlobalNorm(grad, 1.0)
      val lr = Optimizer.schedule(step, steps, warmup, peakLr, peakLr * 0.1)
      optimizer.update(params, grad, lr, weightDecay, decayMask)
      step += 1
      val loss = losses.sum / batch
      recent += loss; recentCount += 1
      if step % 20 == 0 then
        val elapsed = (System.nanoTime() - start) / 1e9
        out(f"step $step%6d  loss ${recent / recentCount}%.4f  lr $lr%.2e  |g| $norm%.2f  ${elapsed}%.0fs  ${step * batch * cfg.context / elapsed}%.0f tok/s")
        recent = 0.0; recentCount = 0
      if step % evalEvery == 0 || step == steps then
        val v = evaluate()
        out(f"eval  $step%6d  valid ${v}%.4f  (best ${math.min(best, v)}%.4f)")
        if v < best then
          best = v
          Checkpoint.save(outDir, cfg, params, tokenizer)
        Checkpoint.save(outDir.resolve("last"), cfg, params, tokenizer)
        val sample = Generate.sample(model, params, tokenizer, "　吾輩は", 80, 0.8, 40, new Random(step))
        out("sample: " + sample.replace("\n", "⏎"))
    log.close()
