package slm

import java.nio.file.{Files, Path}
import scala.util.Random

/** 学習本体と同じ 1-step 関数を回して内訳を測る: `runMain slm.BenchTrain d=768 heads=12 layers=14 ff=3072 context=256 batch=32 threads=8 steps=6 [corpus=...] [vocabFile=...]` */
object BenchTrain:

  def main(args: Array[String]): Unit =
    val opt = Train.parseArgs(args)
    def int(k: String, d: Int) = opt.get(k).map(_.toInt).getOrElse(d)
    val steps = int("steps", 6)
    val workers = int("threads", 8)
    val vocab = opt.get("vocabFile").map(f => Tokenizer.load(Path.of(f))).getOrElse(new Tokenizer((0x3041 to 0x30FF).map(_.toChar).toVector))
    val cfg = Config(vocab.vocabSize, int("d", 768), int("heads", 12), int("layers", 14), int("context", 256), int("ff", 3072), opt.getOrElse("attention", "softmax"),
      opt.getOrElse("positions", "learned"))
    val batch = int("batch", 32)
    Preflight.validate(cfg, batch, workers)
    val est = Preflight.estimate(cfg)
    println(est.toText(Seq(workers), Preflight.heapBytes))
    val run = RunConfig(cfg, batch, 4096, 256, 3e-4, 3e-5, 0.1, 1.0, 0L)
    // データ: 本物のコーパスがあればそれ、無ければ乱数トークン
    val tokens = opt.get("corpus").map(c => vocab.encode(Files.readString(Path.of(c)))).getOrElse {
      val r = new Random(1); Array.fill(2_000_000)(r.nextInt(cfg.vocab))
    }
    val split = (tokens.length * 0.97).toInt
    val pool = new WorkerPool(workers)
    val model = new Model(cfg)
    val trainer = new Trainer(run, model, tokens, split, split, tokens.length, pool, vocab, println)
    trainer.initFresh(0L)
    println(s"model: $cfg P=${model.parameterCount} workers=$workers batch=$batch")
    var stats = Vector.empty[StepStats]
    for i <- 1 to steps do
      val t0 = System.nanoTime()
      val st = trainer.trainStep()
      val wallMs = (System.nanoTime() - t0) / 1e6
      val tps = batch.toLong * cfg.context / (wallMs / 1000)
      println(f"step $i%2d  loss ${st.loss}%.4f |g| ${st.norm}%.3f  sample ${st.sampleMs}%.0f zero ${st.zeroMs}%.0f fb ${st.fbMs}%.0f reduce ${st.reduceMs}%.0f adam ${st.adamMs}%.0f  wall ${wallMs}%.0fms  $tps%.0f tok/s  rss ${Train.rssMiB}%.0fMiB")
      if i > 2 then stats = stats :+ st
    if stats.nonEmpty then
      val med = (f: StepStats => Double) => { val v = stats.map(f).sorted; v(v.length / 2) }
      val total = med(s => s.sampleMs + s.zeroMs + s.fbMs + s.reduceMs + s.adamMs)
      println(f"median (steps 3..$steps): fb ${med(_.fbMs)}%.0fms reduce ${med(_.reduceMs)}%.0fms adam ${med(_.adamMs)}%.0fms zero ${med(_.zeroMs)}%.0fms  total ${total}%.0fms  ${batch.toLong * cfg.context / (total / 1000)}%.0f tok/s")
    val t0 = System.nanoTime()
    val q = trainer.evaluate(trainer.quickWindows)
    println(f"eval quick64: ${q.getOrElse(Double.NaN)}%.4f in ${(System.nanoTime() - t0) / 1e9}%.1fs")
    pool.shutdown()
