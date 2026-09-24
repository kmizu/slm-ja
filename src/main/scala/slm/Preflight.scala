package slm

/** 形状・メモリ・データの事前確認。配列を確保する前に Long で見積もる。 */
object Preflight:

  final case class Estimate(params: Long, paramBytes: Long, workspaceBytes: Long, decayRangeCount: Int):
    /** 学習時の主要配列: params, grad, m, v + worker ごとの grad と Workspace。 */
    def trainingBytes(workers: Int): Long = 4L * paramBytes + workers.toLong * (paramBytes + workspaceBytes)
    def toText(workers: Seq[Int], heapBytes: Long): String =
      val sb = new StringBuilder
      sb.append(f"params=$params%,d  Float[P]=${mib(paramBytes)}%.1f MiB  workspace/worker=${mib(workspaceBytes)}%.1f MiB\n")
      for w <- workers do
        val b = trainingBytes(w)
        sb.append(f"  workers=$w%2d: 主要配列 ${gib(b)}%.2f GiB  (heap ${gib(heapBytes)}%.1f GiB の ${100.0 * b / heapBytes}%.0f%%)\n")
      sb.append(f"checkpoint 1 世代 (params+m+v) = ${gib(3 * paramBytes)}%.2f GiB\n")
      sb.toString

  def mib(b: Long): Double = b / 1048576.0
  def gib(b: Long): Double = b / 1073741824.0

  /** P = V*d + T*d + L*(4*d*d + 2*d*F + 9*d + F) + 2*d + V */
  def parameterCount(c: Config): Long =
    val V = c.vocab.toLong; val d = c.d.toLong; val T = c.context.toLong; val L = c.layers.toLong; val F = c.ff.toLong
    V * d + T * d + L * (4 * d * d + 2 * d * F + 9 * d + F) + 2 * d + V

  /** S = L*(8*T*d + T*F + H*T*T + 4*T) + 12*T*d + 2*T*F + T*V + 3*T + 8 */
  def workspaceFloats(c: Config): Long =
    val V = c.vocab.toLong; val d = c.d.toLong; val T = c.context.toLong; val L = c.layers.toLong; val F = c.ff.toLong; val H = c.heads.toLong
    val attn = if c.isLinear then 2 * T * d + H * T else H * T * T
    val linearExtra = if c.isLinear then (d / H).toLong * (d / H) + 3 * (d / H) + T * d + H * T else 0L
    L * (8 * T * d + T * F + attn + 4 * T) + 12 * T * d + 2 * T * F + T * V + 3 * T + 8 + linearExtra

  def estimate(c: Config): Estimate =
    val p = parameterCount(c)
    require(p <= Int.MaxValue, s"パラメータ数 $p が Int を超える")
    require(workspaceFloats(c) <= Int.MaxValue && c.context.toLong * c.vocab <= Int.MaxValue && c.heads.toLong * c.context * c.context <= Int.MaxValue,
      "作業領域の配列長が Int を超える")
    Estimate(p, 4L * p, 4L * workspaceFloats(c), 6 * c.layers)

  def validate(c: Config, batch: Int, workers: Int): Unit =
    require(c.vocab > 0 && c.d > 0 && c.heads > 0 && c.layers > 0 && c.context > 0 && c.ff > 0, "次元は正の整数")
    require(c.d % c.heads == 0, s"d=${c.d} は heads=${c.heads} で割り切れない")
    require(batch >= 1, "batch は 1 以上")
    require(workers >= 1 && workers <= batch, s"workers=$workers は 1 以上 batch=$batch 以下")

  def heapBytes: Long = Runtime.getRuntime.maxMemory()
