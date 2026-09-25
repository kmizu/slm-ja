package slm

import java.nio.file.Path
import scala.util.Random

/** 生成: `runMain slm.Generate [checkpoint=checkpoints/ja1m] [prompt=...] [count=200] [temperature=0.8] [topK=40] [seed=0] [threads=1] [method=decoder|recompute] [time=1]` */
object Generate:

  /** 1 文字ずつの推論（`Decoder`）で count 文字を生成する。
    *
    * 状態を持ち回すので、線形注意では 1 文字あたりの計算が文脈長に依らない（softmax 注意は KV キャッシュで、今の q との内積だけ）。
    * プロンプトの読み込みと窓の詰め直しは一括の順伝播（`Decoder.prefill`）で行う。
    * 位置埋め込みが context 個しかないので、位置が context に達したら直近 context/2 文字で窓を詰め直して続ける。
    * 全体が context 文字以内なら、毎回計算し直す `sampleRecompute` と同じ文字列になる。
    */
  def sample(model: Model, params: Array[Float], tokenizer: Tokenizer, prompt: String, count: Int,
             temperature: Double, topK: Int, rng: Random, pool: Option[WorkerPool] = None): String =
    var ids = tokenizer.encode(prompt).toVector
    require(ids.nonEmpty, "プロンプトが空")
    val context = model.cfg.context
    val dec = new Decoder(model, params, pool)
    def prefill(window: Vector[Int]): Array[Float] = dec.prefill(window.toArray)
    var logits = prefill(ids.takeRight(context))
    var n = 1
    while n <= count do
      val next = pick(logits.map(_.toDouble), temperature, topK, rng)
      ids = ids :+ next
      if n < count then
        logits = if dec.position < context then dec.step(next) else prefill(ids.takeRight(math.max(1, context / 2)))
      n += 1
    tokenizer.decode(ids)

  /** 旧来の生成: 1 文字ごとに直近 context 文字の順伝播をやり直す（1 文字あたり context 倍の計算）。検証と速度比較用。 */
  def sampleRecompute(model: Model, params: Array[Float], tokenizer: Tokenizer, prompt: String, count: Int,
                      temperature: Double, topK: Int, rng: Random): String =
    var ids = tokenizer.encode(prompt).toVector
    val ws = model.workspace()
    for _ <- 1 to count do
      val window = ids.takeRight(model.cfg.context).toArray
      val logits = model.lastLogits(params, window, ws).map(_.toDouble)
      ids = ids :+ pick(logits, temperature, topK, rng)
    tokenizer.decode(ids)

  /** 温度で割ってから、上位 topK 個だけを softmax で抽選する。 */
  def pick(logits: Array[Double], temperature: Double, topK: Int, rng: Random): Int =
    if temperature <= 0.0 then logits.indices.maxBy(logits)
    else
      val top = logits.indices.sortBy(i => -logits(i)).take(math.max(1, topK))
      val scaled = top.map(i => logits(i) / temperature)
      val shift = scaled.max
      val weights = scaled.map(s => math.exp(s - shift))
      val total = weights.sum
      var r = rng.nextDouble() * total
      var k = 0
      while k < top.length - 1 && r >= weights(k) do { r -= weights(k); k += 1 }
      top(k)

  def main(args: Array[String]): Unit =
    val opt = args.map(_.split("=", 2)).collect { case Array(k, v) => k -> v }.toMap
    val (cfg, params, tokenizer) = Checkpoint.load(Path.of(opt.getOrElse("checkpoint", "checkpoints/ja1m")))
    val model = new Model(cfg)
    val prompt = opt.getOrElse("prompt", "　吾輩は")
    val count = opt.get("count").map(_.toInt).getOrElse(200)
    val temperature = opt.get("temperature").map(_.toDouble).getOrElse(0.8)
    val topK = opt.get("topK").map(_.toInt).getOrElse(40)
    val rng = new Random(opt.get("seed").map(_.toLong).getOrElse(0L))
    val threads = opt.get("threads").map(_.toInt).getOrElse(1)
    val pool = if threads > 1 then Some(new WorkerPool(threads)) else None
    val t0 = System.nanoTime()
    val text = opt.getOrElse("method", "decoder") match
      case "decoder" => sample(model, params, tokenizer, prompt, count, temperature, topK, rng, pool)
      case "recompute" => sampleRecompute(model, params, tokenizer, prompt, count, temperature, topK, rng)
      case m => throw new IllegalArgumentException(s"method は decoder か recompute: $m")
    val sec = (System.nanoTime() - t0) / 1e9
    pool.foreach(_.shutdown())
    println(text)
    if opt.contains("time") then System.err.println(f"generated $count chars in $sec%.2f s (${sec * 1000 / count}%.1f ms/char, method=${opt.getOrElse("method", "decoder")}, threads=$threads)")
