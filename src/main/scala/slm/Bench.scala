package slm

import java.util.stream.IntStream
import scala.util.Random

/** 速度計測: `runMain slm.Bench [threads=16] [batch=32] [rounds=3]`。本番と同じ大きさのモデルで lossAndGrad を回す。 */
object Bench:

  def main(args: Array[String]): Unit =
    val opt = args.map(_.split("=", 2)).collect { case Array(k, v) => k -> v }.toMap
    val threads = opt.get("threads").map(_.toInt).getOrElse(Runtime.getRuntime.availableProcessors())
    val batch = opt.get("batch").map(_.toInt).getOrElse(32)
    val rounds = opt.get("rounds").map(_.toInt).getOrElse(3)
    val cfg = Config(vocab = 3613, d = 128, heads = 4, layers = 3, context = 128, ff = 512)
    val model = new Model(cfg)
    val rng = new Random(0)
    val p = model.layout.init(rng)
    val windows = Array.fill(batch)(Array.fill(cfg.context + 1)(rng.nextInt(cfg.vocab)))
    val tokens = batch * cfg.context

    // 1 スレッド
    val g1 = new Array[Double](p.length)
    val ws1 = model.workspace()
    for r <- 0 to rounds do
      val t0 = System.nanoTime()
      var b = 0
      while b < batch do { model.lossAndGrad(p, g1, windows(b), ws1); b += 1 }
      val sec = (System.nanoTime() - t0) / 1e9
      if r > 0 then println(f"1 thread : ${tokens / sec}%.0f tok/s  (${sec}%.2f s / $batch seqs)")

    // 並列
    val buffers = Array.fill(threads)(new Array[Double](p.length))
    val spaces = Array.fill(threads)(model.workspace())
    for r <- 0 to rounds do
      val t0 = System.nanoTime()
      IntStream.range(0, threads).parallel().forEach { th =>
        var b = th
        while b < batch do { model.lossAndGrad(p, buffers(th), windows(b), spaces(th)); b += threads }
      }
      val sec = (System.nanoTime() - t0) / 1e9
      if r > 0 then println(f"$threads threads: ${tokens / sec}%.0f tok/s  (${sec}%.2f s / $batch seqs)")
