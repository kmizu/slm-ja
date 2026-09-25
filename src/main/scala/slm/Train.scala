package slm

import java.nio.file.{Files, Path}
import scala.util.Random

/** 1 update の内訳。 */
final case class StepStats(loss: Double, norm: Double, clipped: Boolean, lr: Double,
                           sampleMs: Double, zeroMs: Double, fbMs: Double, reduceMs: Double, adamMs: Double)

/** 学習の状態機械。データ・pool・作業領域を持ち、1 update ずつ進める。 */
final class Trainer(val run: RunConfig, val model: Model, val tokens: Array[Int], val trainLength: Int,
                    val validStart: Int, val validEnd: Int, val pool: WorkerPool, val tokenizer: Tokenizer,
                    val log: String => Unit):
  val cfg: Config = run.cfg
  val P: Int = model.parameterCount
  val workers: Int = pool.size
  val params: Array[Float] = new Array[Float](P)
  val adam = new AdamW(P, run.beta1, run.beta2, run.eps)
  val decay: DecayRanges = DecayRanges.of(model.layout)
  private val gradBuffers = Array.fill(workers)(new Array[Float](P))
  private val grad = new Array[Float](P)
  private val spaces = Array.fill(workers)(model.workspace())
  private val partial = new Array[Double](workers)
  private val losses = new Array[Double](run.batch)
  var tokensSeen: Long = 0L
  var bestQuick: Double = Double.MaxValue
  var bestGeneration: String = ""
  var elapsedBefore: Double = 0.0
  var meta: Map[String, String] = Map.empty
  def stepsDone: Long = adam.step

  val quickWindows: Array[Int] = EvalWindows.starts(validStart, validEnd, cfg.context, 64)
  val finalWindows: Array[Int] = EvalWindows.starts(validStart, validEnd, cfg.context, 1024)

  def initFresh(seed: Long): Unit =
    System.arraycopy(model.layout.init(new Random(seed)), 0, params, 0, P)

  def initFrom(weights: Array[Float]): Unit =
    require(weights.length == P, s"重みの数が違う: ${weights.length} vs $P")
    System.arraycopy(weights, 0, params, 0, P)

  def restore(st: Checkpoint.TrainState): Unit =
    require(st.params.length == P)
    System.arraycopy(st.params, 0, params, 0, P)
    System.arraycopy(st.m, 0, adam.m, 0, P)
    System.arraycopy(st.v, 0, adam.v, 0, P)
    adam.step = st.optimizerStep
    tokensSeen = st.tokensSeen
    bestQuick = st.bestQuickValid
    bestGeneration = st.bestGeneration
    elapsedBefore = st.elapsedSeconds

  private def ms(t0: Long): Double = (System.nanoTime() - t0) / 1e6

  /** 1 update。窓の抽出 → worker ごとの forward/backward → 区間並列の集約と norm → clip を織り込んだ AdamW。 */
  def trainStep(): StepStats =
    val step = stepsDone
    var t0 = System.nanoTime()
    val starts = Sampler.windowStarts(run.seed, step, run.batch, trainLength, cfg.context)
    val sampleMs = ms(t0)
    t0 = System.nanoTime()
    pool.run(workers)(th => java.util.Arrays.fill(gradBuffers(th), 0f))
    val zeroMs = ms(t0)
    t0 = System.nanoTime()
    val length = cfg.context + 1
    pool.run(workers) { th =>
      var b = th
      while b < run.batch do
        losses(b) = model.lossAndGrad(params, gradBuffers(th), tokens, starts(b), length, spaces(th))
        b += workers
    }
    val fbMs = ms(t0)
    t0 = System.nanoTime()
    val invBatch = (1.0 / run.batch).toFloat
    pool.ranges(P) { (k, s, e) => partial(k) = Optimizer.reduceRange(gradBuffers, grad, s, e, invBatch) }
    var sumSq = 0.0
    var k = 0
    while k < workers do { sumSq += partial(k); k += 1 } // 固定順
    val norm = math.sqrt(sumSq)
    val reduceMs = ms(t0)
    t0 = System.nanoTime()
    val lr = run.lr(step)
    val scale = Optimizer.clipScale(norm, run.clip)
    val coeff = adam.beginStep(lr, scale, run.weightDecay)
    pool.ranges(P) { (_, s, e) => adam.updateRange(params, grad, s, e, coeff, decay) }
    val adamMs = ms(t0)
    tokensSeen += run.batch.toLong * cfg.context
    var lossSum = 0.0
    var b = 0
    while b < run.batch do { lossSum += losses(b); b += 1 }
    StepStats(lossSum / run.batch, norm, scale < 1.0, lr, sampleMs, zeroMs, fbMs, reduceMs, adamMs)

  /** 固定窓集合の平均損失（順伝播のみ、勾配配列を使わない）。窓が無ければ None。 */
  def evaluate(windows: Array[Int]): Option[Double] =
    if windows.isEmpty then None
    else
      val sums = new Array[Double](workers)
      val length = cfg.context + 1
      pool.run(workers) { th =>
        var s = 0.0
        var i = th
        while i < windows.length do { s += model.lossOnly(params, tokens, windows(i), length, spaces(th)); i += workers }
        sums(th) = s
      }
      Some(sums.sum / windows.length)

  def sample(prompt: String, count: Int, seed: Long): String =
    Generate.sample(model, params, tokenizer, prompt, count, 0.8, 40, new Random(seed))

  def state(elapsedNow: Double): Checkpoint.TrainState =
    Checkpoint.TrainState(run, params, adam.m, adam.v, adam.step, tokensSeen, bestQuick, bestGeneration, elapsedBefore + elapsedNow, meta)

object Train:

  val knownKeys: Set[String] = Set("mode", "corpus", "vocabFile", "minCount", "d", "heads", "layers", "context", "ff", "batch", "threads",
    "steps", "stopAfterSteps", "lr", "floorLr", "warmup", "wd", "clip", "seed", "evalEvery", "saveEvery", "saveSeconds", "sampleEvery",
    "prompt", "out", "resume", "initFrom", "stopFile", "keepGenerations", "split", "attention", "extendTo", "restartWarmup", "restartLr")

  def parseArgs(args: Array[String]): Map[String, String] =
    val kv = args.map { a =>
      val i = a.indexOf('=')
      require(i > 0, s"引数は key=value の形: $a")
      a.substring(0, i) -> a.substring(i + 1)
    }.toMap
    val unknown = kv.keySet -- knownKeys
    require(unknown.isEmpty, s"不明な引数: ${unknown.mkString(", ")}。使えるのは ${knownKeys.toSeq.sorted.mkString(", ")}")
    kv

  def rssMiB: Double =
    try
      Files.readAllLines(Path.of("/proc/self/status")).stream().filter(_.startsWith("VmRSS:")).findFirst()
        .map(l => l.split("\\s+")(1).toDouble / 1024).orElse(0.0)
    catch case _: Exception => 0.0

  def gitCommit: String =
    try
      val p = new ProcessBuilder("git", "rev-parse", "HEAD").redirectErrorStream(true).start()
      val out = new String(p.getInputStream.readAllBytes()).trim
      if p.waitFor() == 0 then out else "unknown"
    catch case _: Exception => "unknown"

  def main(args: Array[String]): Unit =
    val opt = parseArgs(args)
    def str(k: String, d: String) = opt.getOrElse(k, d)
    def int(k: String, d: Int) = opt.get(k).map(_.toInt).getOrElse(d)
    def dbl(k: String, d: Double) = opt.get(k).map(_.toDouble).getOrElse(d)
    val mode = str("mode", "train")
    val outDir = Path.of(str("out", "runs/run"))
    val resume = opt.get("resume").filter(_.nonEmpty)

    // ---- 再開なら保存済みの run 設定を使う ----
    val resumed: Option[(Checkpoint.TrainState, Tokenizer, Path)] = resume.map { dir =>
      val stateDir = Path.of(dir).resolve("state")
      val gen = Checkpoint.resolveLatest(stateDir).getOrElse(throw new IllegalArgumentException(
        s"$stateDir に完全な世代がない。重みだけの旧形式は resume= ではなく initFrom= で読み込む"))
      val (st, tok) = Checkpoint.loadState(gen)
      (st, tok, gen)
    }

    // ---- データ ----
    val corpusPath = Path.of(str("corpus", "data/corpus.txt"))
    require(Files.exists(corpusPath), s"コーパスが無い: $corpusPath")
    val text = Files.readString(corpusPath)
    val corpusHash = Checkpoint.sha256(corpusPath)
    val tokenizer = resumed.map(_._2).orElse(opt.get("vocabFile").map(f => Tokenizer.load(Path.of(f))))
      .getOrElse(Tokenizer.fromText(text, int("minCount", 10)))
    val tokens = tokenizer.encode(text)
    val splitRatio = dbl("split", 0.97)
    val split = resumed.flatMap(_._1.meta.get("split")).map(_.toInt).getOrElse((tokens.length * splitRatio).toInt)
    val unkRate = tokens.count(_ == tokenizer.unk).toDouble / tokens.length

    val run0 = resumed.map(_._1.run).getOrElse {
      val cfg = Config(tokenizer.vocabSize, int("d", 128), int("heads", 4), int("layers", 3), int("context", 128), int("ff", 512), str("attention", "softmax"))
      val peak = dbl("lr", 1e-3)
      RunConfig(cfg, int("batch", 32), int("steps", 4096), int("warmup", 256), peak, dbl("floorLr", peak * 0.1), dbl("wd", 0.1), dbl("clip", 1.0), int("seed", 0).toLong)
    }
    // ---- 延長（warm restart）: 予算を使い切った run を extendTo update まで伸ばす。元の区間の学習率は変えない ----
    val run = opt.get("extendTo").map(_.toInt) match
      case Some(to) =>
        require(resumed.isDefined, "extendTo= は resume= と一緒に使う")
        RunConfig.extend(run0, to,
          int("restartWarmup", if run0.isExtended then run0.restartWarmup else 128),
          dbl("restartLr", if run0.isExtended then run0.restartPeakLr else run0.peakLr * 0.5))
      case None =>
        require(!opt.contains("restartWarmup") && !opt.contains("restartLr"), "restartWarmup= と restartLr= は extendTo= と一緒に使う")
        run0
    for (k, v) <- Seq("d" -> run.cfg.d, "heads" -> run.cfg.heads, "layers" -> run.cfg.layers, "context" -> run.cfg.context, "ff" -> run.cfg.ff, "batch" -> run.batch, "steps" -> run.steps) do
      opt.get(k).foreach(given_ => require(given_.toInt == v, s"resume と CLI の $k が違う: 保存 $v, 指定 $given_"))
    opt.get("attention").foreach(a => require(a == run.cfg.attention, s"resume と CLI の attention が違う: 保存 ${run.cfg.attention}, 指定 $a"))
    resumed.foreach { (st, _, _) =>
      require(st.meta("corpusHash") == corpusHash, "コーパスが保存時と違う")
      require(st.meta("vocabHash") == tokenizer.hash, "語彙が保存時と違う")
    }
    val cfg = run.cfg
    val workers = int("threads", math.min(run.batch, Runtime.getRuntime.availableProcessors()))
    Preflight.validate(cfg, run.batch, workers)
    require(split > cfg.context + 1 && tokens.length - split >= cfg.context + 1, s"学習/検証データが短すぎる: split=$split total=${tokens.length}")
    val est = Preflight.estimate(cfg)
    val report = new StringBuilder
    report.append(s"corpus=$corpusPath chars=${text.length} tokens=${tokens.length} vocab=${tokenizer.vocabSize} unkRate=${f"$unkRate%.5f"} split=$split corpusSha256=$corpusHash vocabHash=${tokenizer.hash}\n")
    report.append(s"config=$cfg batch=${run.batch} steps=${run.steps} workers=$workers lanes=${Simd.lanes}\n")
    report.append(est.toText(Seq(1, 2, 4, 8, 16).filter(_ <= run.batch), Preflight.heapBytes))
    println(report)
    if mode == "preflight" then
      val quick = EvalWindows.starts(split, tokens.length, cfg.context, 64).length
      println(s"quick eval windows=$quick final eval windows=${EvalWindows.starts(split, tokens.length, cfg.context, 1024).length}")
      return
    require(mode == "train", s"mode は preflight か train: $mode")
    require(est.trainingBytes(workers) < Preflight.heapBytes * 0.8, f"メモリ不足の見込み: 主要配列 ${Preflight.gib(est.trainingBytes(workers))}%.1f GiB に対して heap ${Preflight.gib(Preflight.heapBytes)}%.1f GiB。threads を減らすか heap を増やす")

    // ---- run directory と lock ----
    Files.createDirectories(outDir)
    val lock = outDir.resolve("LOCK")
    if Files.exists(lock) then
      val pid = Files.readString(lock).trim
      require(!Files.exists(Path.of("/proc").resolve(pid)), s"同じ run が動いている (pid $pid): $lock")
    Files.writeString(lock, ProcessHandle.current().pid().toString)
    val logFile = Files.newBufferedWriter(outDir.resolve("train.log"), java.nio.charset.StandardCharsets.UTF_8,
      java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND)
    def out(s: String): Unit = { println(s); logFile.write(s); logFile.newLine(); logFile.flush() }
    Files.writeString(outDir.resolve("run.txt"), run.toText + s"workers=$workers\ncommand=${args.mkString(" ")}\n" + report)

    val pool = new WorkerPool(workers)
    val model = new Model(cfg)
    val trainer = new Trainer(run, model, tokens, split, split, tokens.length, pool, tokenizer, out)
    trainer.meta = Map(
      "corpusHash" -> corpusHash, "vocabHash" -> tokenizer.hash, "split" -> split.toString, "tokens" -> tokens.length.toString,
      "samplerVersion" -> Sampler.version.toString, "reduction" -> "worker-order-v1", "kernel" -> "dot4x2-v1",
      "quickEval" -> EvalWindows.identity(64), "finalEval" -> EvalWindows.identity(1024), "commit" -> gitCommit,
      "platform" -> s"${System.getProperty("os.name")} ${System.getProperty("os.arch")} jdk${System.getProperty("java.version")} lanes=${Simd.lanes}",
      "workers" -> workers.toString)
    resumed match
      case Some((st, _, gen)) =>
        trainer.restore(st)
        out(s"resume from $gen: step=${st.optimizerStep} tokensSeen=${st.tokensSeen} bestQuick=${st.bestQuickValid}")
      case None =>
        opt.get("initFrom") match
          case Some(dir) =>
            val (c, w, _) = Checkpoint.load(Path.of(dir))
            require(c == cfg, s"initFrom の形状が違う: $c vs $cfg")
            trainer.initFrom(w)
            out(s"initFrom $dir (重みだけ。optimizer と step は新規)")
          case None => trainer.initFresh(run.seed)
    out(report.toString.trim)
    out(s"model: $cfg, パラメータ ${model.parameterCount} 個, workers=$workers, batch=${run.batch}, steps=${run.steps} (done ${trainer.stepsDone})")

    val evalEvery = int("evalEvery", 250)
    val saveEvery = int("saveEvery", 250)
    val saveSeconds = int("saveSeconds", 600)
    val sampleEvery = int("sampleEvery", 0)
    val stopAfter = int("stopAfterSteps", Int.MaxValue)
    val stopFile = opt.get("stopFile").map(Path.of(_))
    val keep = int("keepGenerations", 2)
    val prompt = str("prompt", "　私はその時、")
    val stateDir = outDir.resolve("state")
    val start = System.nanoTime()
    val startStep = trainer.stepsDone
    val startTokens = trainer.tokensSeen
    def elapsed = (System.nanoTime() - start) / 1e9
    var lastSave = System.nanoTime()
    var status = "running"
    var clipCount = 0L
    var stepCount = 0L
    var recentLoss = 0.0; var recentN = 0
    var recentFb = 0.0; var recentReduce = 0.0; var recentAdam = 0.0; var recentZero = 0.0; var recentSample = 0.0
    var windowStart = System.nanoTime()
    var totalSaveSeconds = 0.0
    var totalEvalSeconds = 0.0
    var lastSavedStep = -1L

    def writeStatus(): Unit =
      Files.writeString(outDir.resolve("status.txt"),
        f"status=$status\nstep=${trainer.stepsDone}\ntokensSeen=${trainer.tokensSeen}\nbestQuickValid=${trainer.bestQuick}\nbestGeneration=${trainer.bestGeneration}\nelapsedSeconds=${trainer.elapsedBefore + elapsed}%.1f\nlast=${Checkpoint.readPointer(stateDir.resolve("last")).getOrElse("")}\n")

    def save(tag: String): Unit =
      val t0 = System.nanoTime()
      val gen = Checkpoint.saveState(stateDir, trainer.state(elapsed), tokenizer)
      Checkpoint.prune(stateDir, keep, trainer.bestGeneration)
      val sec = (System.nanoTime() - t0) / 1e9
      totalSaveSeconds += sec
      lastSave = System.nanoTime()
      lastSavedStep = trainer.stepsDone
      out(f"save  ${trainer.stepsDone}%6d  $tag -> ${gen.getFileName}  ${sec}%.1fs")
      writeStatus()

    def quickEval(): Unit =
      val t0 = System.nanoTime()
      trainer.evaluate(trainer.quickWindows) match
        case Some(v) =>
          val improved = v < trainer.bestQuick
          if improved then trainer.bestQuick = v
          totalEvalSeconds += (System.nanoTime() - t0) / 1e9
          out(f"eval  ${trainer.stepsDone}%6d  quick64 ${v}%.4f  (best ${trainer.bestQuick}%.4f)  ${(System.nanoTime() - t0) / 1e9}%.1fs")
          if improved then
            save("best")
            trainer.bestGeneration = Checkpoint.readPointer(stateDir.resolve("last")).getOrElse("")
            Checkpoint.writePointer(stateDir.resolve("best"), trainer.bestGeneration)
        case None => out("eval: 検証窓が無い")

    try
      while trainer.stepsDone < run.steps && status == "running" do
        val st = trainer.trainStep()
        stepCount += 1
        if st.clipped then clipCount += 1
        recentLoss += st.loss; recentN += 1
        recentFb += st.fbMs; recentReduce += st.reduceMs; recentAdam += st.adamMs; recentZero += st.zeroMs; recentSample += st.sampleMs
        val step = trainer.stepsDone
        if step % 20 == 0 || step == run.steps then
          val windowSec = (System.nanoTime() - windowStart) / 1e9
          val rolling = recentN.toLong * run.batch * cfg.context / windowSec
          val wall = (trainer.tokensSeen - startTokens) / elapsed
          val remaining = (run.steps - step).toDouble * run.batch * cfg.context
          val eta = if wall > 0 then remaining / wall / 3600 else Double.NaN
          out(f"step $step%6d  loss ${recentLoss / recentN}%.4f  lr ${st.lr}%.2e  |g| ${st.norm}%.2f  clip ${100.0 * clipCount / stepCount}%.0f%%  " +
            f"fb ${recentFb / recentN}%.0fms red ${recentReduce / recentN}%.0fms adam ${recentAdam / recentN}%.0fms zero ${recentZero / recentN}%.0fms  " +
            f"rolling $rolling%.0f tok/s wall $wall%.0f tok/s  rss ${rssMiB}%.0fMiB  tokens ${trainer.tokensSeen}  eta ${eta}%.1fh")
          recentLoss = 0; recentN = 0; recentFb = 0; recentReduce = 0; recentAdam = 0; recentZero = 0; recentSample = 0
          windowStart = System.nanoTime()
        if evalEvery > 0 && step % evalEvery == 0 && step < run.steps then quickEval()
        if sampleEvery > 0 && step % sampleEvery == 0 then out("sample: " + trainer.sample(prompt, 80, step).replace("\n", "⏎"))
        val wantSave = (saveEvery > 0 && step % saveEvery == 0) || (saveSeconds > 0 && (System.nanoTime() - lastSave) / 1e9 >= saveSeconds)
        val stopRequested = stopFile.exists(Files.exists(_))
        if step - startStep >= stopAfter || stopRequested then
          status = "interrupted"
          if lastSavedStep != step then save(if stopRequested then "stop-file" else "stopAfterSteps")
        else if wantSave && step < run.steps && lastSavedStep != step then save("periodic")
      if status == "running" then
        // 予算に到達: 最終評価・最終保存・推論用 export
        val t0 = System.nanoTime()
        val fin = trainer.evaluate(trainer.finalWindows)
        totalEvalSeconds += (System.nanoTime() - t0) / 1e9
        out(f"final ${trainer.stepsDone}%6d  final1024 ${fin.getOrElse(Double.NaN)}%.4f  windows=${trainer.finalWindows.length}  ${(System.nanoTime() - t0) / 1e9}%.1fs")
        quickEval()
        save("final")
        Checkpoint.save(outDir.resolve("export-final"), cfg, trainer.params, tokenizer)
        Checkpoint.completeGenerations(stateDir).find(_.getFileName.toString == trainer.bestGeneration).foreach { best =>
          val (bs, _) = Checkpoint.loadState(best)
          Checkpoint.save(outDir.resolve("export-best"), cfg, bs.params, tokenizer)
        }
        status = "completed"
        writeStatus()
      out(f"done: status=$status step=${trainer.stepsDone} tokensSeen=${trainer.tokensSeen} elapsed=${elapsed}%.0fs save=${totalSaveSeconds}%.0fs eval=${totalEvalSeconds}%.0fs")
    catch
      case e: Throwable =>
        status = "failed"
        out(s"failed: ${e}")
        writeStatus()
        throw e
    finally
      writeStatus()
      Files.deleteIfExists(lock)
      logFile.close()
      pool.shutdown()
