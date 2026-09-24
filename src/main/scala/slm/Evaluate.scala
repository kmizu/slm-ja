package slm

import java.nio.file.{Files, Path}
import scala.util.Random

/** 固定した評価窓と固定プロンプトで checkpoint を評価・生成する。10M と 100M を同じ条件で比べるために使う。
  *
  *   runMain slm.Evaluate checkpoint=checkpoints/ja10m corpus=data/corpus.txt [split=0.97] [threads=2] [count=200]
  *   runMain slm.Evaluate state=runs/ja100m/state/step-00000750 export=runs/ja100m/export-750 corpus=data/corpus.txt
  *
  * `state=` は full-state の世代から重みを読み、`export=` があれば推論用の旧形式（params.bin/config.txt/vocab.txt）へ書き出す。
  * 損失は quick64（`even64-v1`）と final1024（`even1024-v1`）の両方。窓は検証領域（split 以降）に均等配置。
  */
object Evaluate:

  val prompts: Vector[String] = Vector("　私はその時、", "「おい、", "　夜になると、")

  def main(args: Array[String]): Unit =
    val opt = args.map { a => val i = a.indexOf('='); a.substring(0, i) -> a.substring(i + 1) }.toMap
    val threads = opt.get("threads").map(_.toInt).getOrElse(2)
    val count = opt.get("count").map(_.toInt).getOrElse(200)
    val corpusPath = Path.of(opt.getOrElse("corpus", "data/corpus.txt"))

    val (cfg, params, tokenizer, label) = opt.get("state") match
      case Some(dir) =>
        val (st, tok) = Checkpoint.loadState(Path.of(dir))
        opt.get("export").foreach(e => Checkpoint.save(Path.of(e), st.run.cfg, st.params, tok))
        (st.run.cfg, st.params, tok, s"$dir (step ${st.optimizerStep}, tokensSeen ${st.tokensSeen})")
      case None =>
        val dir = Path.of(opt.getOrElse("checkpoint", "checkpoints/ja10m"))
        val (c, p, tok) = Checkpoint.load(dir)
        (c, p, tok, dir.toString)
    val model = new Model(cfg)
    println(s"model: $label")
    println(s"config: $cfg params=${model.parameterCount} vocabHash=${tokenizer.hash}")

    val text = Files.readString(corpusPath)
    val tokens = tokenizer.encode(text)
    val split = opt.get("splitIndex").map(_.toInt).getOrElse((tokens.length * opt.get("split").map(_.toDouble).getOrElse(0.97)).toInt)
    println(s"corpus: ${text.length} chars, sha256=${Checkpoint.sha256(corpusPath).take(16)}…, split=$split, unkRate=${f"${tokens.count(_ == tokenizer.unk).toDouble / tokens.length}%.5f"}")

    val pool = new WorkerPool(threads)
    val spaces = Array.fill(threads)(model.workspace())
    def evaluate(windows: Array[Int]): Double =
      val sums = new Array[Double](threads)
      val length = cfg.context + 1
      pool.run(threads) { th =>
        var s = 0.0
        var i = th
        while i < windows.length do { s += model.lossOnly(params, tokens, windows(i), length, spaces(th)); i += threads }
        sums(th) = s
      }
      sums.sum / windows.length
    for (name, n) <- Seq("quick64" -> 64, "final1024" -> 1024) do
      val windows = EvalWindows.starts(split, tokens.length, cfg.context, n)
      val t0 = System.nanoTime()
      val loss = evaluate(windows)
      println(f"$name (${EvalWindows.identity(n)}, windows=${windows.length}): loss $loss%.4f  ${(System.nanoTime() - t0) / 1e9}%.1fs")
    pool.shutdown()

    for p <- prompts do
      val text = Generate.sample(model, params, tokenizer, p, count, 0.7, 40, new Random(3))
      println(s"=== $p")
      println(text)
