package slm

import java.util.stream.IntStream
import scala.util.Random

/** 速度計測: `runMain slm.Bench [threads=16] [batch=32] [rounds=3] [vocab=3613 d=128 heads=4 layers=3 context=128 ff=512]`。 */
object Bench:

  def main(args: Array[String]): Unit =
    val opt = args.map(_.split("=", 2)).collect { case Array(k, v) => k -> v }.toMap
    def int(k: String, d: Int) = opt.get(k).map(_.toInt).getOrElse(d)
    val threads = int("threads", Runtime.getRuntime.availableProcessors())
    val batch = int("batch", 32)
    val rounds = int("rounds", 3)
    val cfg = Config(vocab = int("vocab", 3613), d = int("d", 128), heads = int("heads", 4), layers = int("layers", 3), context = int("context", 128), ff = int("ff", 512))
    val model = new Model(cfg)
    val rng = new Random(0)
    val p = model.layout.init(rng)
    println(s"model: $cfg, パラメータ ${model.parameterCount} 個, lanes=${Simd.lanes}")
    val windows = Array.fill(batch)(Array.fill(cfg.context + 1)(rng.nextInt(cfg.vocab)))
    val tokens = batch * cfg.context
    val flops = 6.0 * model.parameterCount * tokens

    val g1 = new Array[Float](p.length)
    val ws1 = model.workspace()
    for r <- 0 to rounds do
      val t0 = System.nanoTime()
      var b = 0
      while b < batch do { model.lossAndGrad(p, g1, windows(b), ws1); b += 1 }
      val sec = (System.nanoTime() - t0) / 1e9
      if r > 0 then println(f"1 thread : ${tokens / sec}%.0f tok/s  ${flops / sec / 1e9}%.1f GFLOPS  (${sec}%.2f s / $batch seqs)")

    val buffers = Array.fill(threads)(new Array[Float](p.length))
    val spaces = Array.fill(threads)(model.workspace())
    for r <- 0 to rounds do
      val t0 = System.nanoTime()
      IntStream.range(0, threads).parallel().forEach { th =>
        var b = th
        while b < batch do { model.lossAndGrad(p, buffers(th), windows(b), spaces(th)); b += threads }
      }
      val sec = (System.nanoTime() - t0) / 1e9
      if r > 0 then println(f"$threads threads: ${tokens / sec}%.0f tok/s  ${flops / sec / 1e9}%.1f GFLOPS  (${sec}%.2f s / $batch seqs)")
