package slm

import java.util.SplittableRandom

/** 学習 run の設定。checkpoint に保存し、resume 時に照合する。 */
final case class RunConfig(
    cfg: Config,
    batch: Int,
    steps: Int,            // run の累計目標 update 数（schedule の全長）
    warmup: Int,
    peakLr: Double,
    floorLr: Double,
    weightDecay: Double,
    clip: Double,
    seed: Long,
    beta1: Double = 0.9,
    beta2: Double = 0.95,
    eps: Double = 1e-8
):
  /** 0 起点の update index に対する学習率（予熱つき cosine、最終 update で floor に一致）。 */
  def lr(updateIndex: Long): Double =
    if updateIndex < warmup then peakLr * (updateIndex + 1).toDouble / warmup
    else
      val span = math.max(1L, steps.toLong - 1 - warmup)
      val progress = math.min(1.0, (updateIndex - warmup).toDouble / span)
      floorLr + (peakLr - floorLr) * 0.5 * (1 + math.cos(math.Pi * progress))

  def toText: String =
    s"""vocab=${cfg.vocab}
       |d=${cfg.d}
       |heads=${cfg.heads}
       |layers=${cfg.layers}
       |context=${cfg.context}
       |ff=${cfg.ff}
       |attention=${cfg.attention}
       |batch=$batch
       |steps=$steps
       |warmup=$warmup
       |peakLr=$peakLr
       |floorLr=$floorLr
       |weightDecay=$weightDecay
       |clip=$clip
       |seed=$seed
       |beta1=$beta1
       |beta2=$beta2
       |eps=$eps
       |""".stripMargin

object RunConfig:
  def fromMap(kv: Map[String, String]): RunConfig =
    RunConfig(
      Config(kv("vocab").toInt, kv("d").toInt, kv("heads").toInt, kv("layers").toInt, kv("context").toInt, kv("ff").toInt, kv.getOrElse("attention", "softmax")),
      kv("batch").toInt, kv("steps").toInt, kv("warmup").toInt, kv("peakLr").toDouble, kv("floorLr").toDouble,
      kv("weightDecay").toDouble, kv("clip").toDouble, kv("seed").toLong, kv("beta1").toDouble, kv("beta2").toDouble, kv("eps").toDouble)

/** データ抽出。step と seed から決定的に窓の開始位置を作る（初期化の乱数消費とは独立）。 */
object Sampler:
  val version: Int = 1

  /** splitmix64 風の混ぜ合わせ。 */
  def mix64(a: Long, b: Long): Long =
    var z = a * 0x9E3779B97F4A7C15L + b + 0x9E3779B97F4A7C15L
    z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L
    z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL
    z ^ (z >>> 31)

  /** 長さ N の列から長さ T+1 の窓を切る開始位置は 0 <= start < N - T。 */
  def numberOfStarts(trainLength: Int, context: Int): Int = trainLength - context

  /** step 番目（0 起点）の update で使う batch 個の開始位置（系列 index 順）。 */
  def windowStarts(seed: Long, step: Long, batch: Int, trainLength: Int, context: Int): Array[Int] =
    val n = numberOfStarts(trainLength, context)
    require(n >= 1, s"学習データが短すぎる: length=$trainLength context=$context")
    val r = new SplittableRandom(mix64(seed, step))
    Array.fill(batch)(r.nextInt(n))

/** 評価窓。valid 領域に均等配置した固定の窓集合。 */
object EvalWindows:
  /** 領域 [start, end) から長さ T+1 の窓を最大 count 個、均等に取る。足りなければ有効な全窓。 */
  def starts(start: Int, end: Int, context: Int, count: Int): Array[Int] =
    val length = end - start
    val available = length - context // 有効な開始位置の数（0 <= s < length - T）
    if available <= 0 then Array.empty
    else if available <= count then Array.tabulate(available)(i => start + i)
    else Array.tabulate(count)(i => start + ((i.toLong * (available - 1)) / (count - 1)).toInt)

  def identity(count: Int): String = s"even$count-v1"
