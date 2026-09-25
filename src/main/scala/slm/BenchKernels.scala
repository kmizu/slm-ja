package slm

import scala.util.Random

/** 密な層のカーネルの効率を、順伝播と逆伝播に分けて 1 スレッドで測る。
  *
  *   runMain slm.BenchKernels [d=768] [ff=3072] [vocab=4134] [heads=12] [layers=2] [context=256] [attention=linear] [rounds=5]
  *
  * 100M と同じ幅で層を減らした模型を使う。GFLOPS は密な層（注意の射影 4 つ、FF 2 つ、出力ヘッド）の積和だけを数える。
  * 逆伝播の時間は `lossAndGrad`（順伝播 + 逆伝播）から `lossOnly`（順伝播）を引いたもの。
  */
object BenchKernels:

  def main(args: Array[String]): Unit =
    val opt = args.map(_.split("=", 2)).collect { case Array(k, v) => k -> v }.toMap
    def int(k: String, d: Int) = opt.get(k).map(_.toInt).getOrElse(d)
    val cfg = Config(int("vocab", 4134), int("d", 768), int("heads", 12), int("layers", 2), int("context", 256), int("ff", 3072),
      opt.getOrElse("attention", "linear"))
    val rounds = int("rounds", 5)
    val model = new Model(cfg)
    val p = model.layout.init(new Random(1))
    val g = new Array[Float](p.length)
    val ws = model.workspace()
    val r = new Random(2)
    val T = cfg.context
    val tokens = Array.fill(T + 1)(r.nextInt(cfg.vocab))
    val d = cfg.d.toDouble
    val denseFlopsPerToken = 2.0 * (cfg.layers * (4 * d * d + 2 * d * cfg.ff) + cfg.vocab * d)
    val fwdFlops = denseFlopsPerToken * T
    val bwdFlops = 2 * fwdFlops
    println(s"config=$cfg lanes=${Simd.lanes} rounds=$rounds dense GFLOP/seq: fwd ${fwdFlops / 1e9} bwd ${bwdFlops / 1e9}")
    // JIT の立ち上がり
    for _ <- 1 to 2 do { model.lossOnly(p, tokens, 0, T + 1, ws); java.util.Arrays.fill(g, 0f); model.lossAndGrad(p, g, tokens, 0, T + 1, ws) }
    val fwd = new Array[Double](rounds)
    val both = new Array[Double](rounds)
    for k <- 0 until rounds do
      val t0 = System.nanoTime()
      model.lossOnly(p, tokens, 0, T + 1, ws)
      val t1 = System.nanoTime()
      java.util.Arrays.fill(g, 0f)
      val t2 = System.nanoTime()
      model.lossAndGrad(p, g, tokens, 0, T + 1, ws)
      val t3 = System.nanoTime()
      fwd(k) = (t1 - t0) / 1e9
      both(k) = (t3 - t2) / 1e9
    def median(a: Array[Double]) = a.sorted.apply(a.length / 2)
    val f = median(fwd)
    val b = median(both) - f
    println(f"forward  ${f * 1000}%8.1f ms  ${fwdFlops / f / 1e9}%6.1f GFLOPS")
    println(f"backward ${b * 1000}%8.1f ms  ${bwdFlops / b / 1e9}%6.1f GFLOPS")
    println(f"checksum ${g.iterator.map(_.toDouble).sum}%.6e")
