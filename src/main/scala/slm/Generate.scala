package slm

import java.nio.file.Path
import scala.util.Random

/** 生成: `runMain slm.Generate [checkpoint=checkpoints/ja1m] [prompt=...] [count=200] [temperature=0.8] [topK=40] [seed=0]` */
object Generate:

  def sample(model: Model, params: Array[Float], tokenizer: Tokenizer, prompt: String, count: Int,
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
    val text = sample(model, params, tokenizer, opt.getOrElse("prompt", "　吾輩は"), opt.get("count").map(_.toInt).getOrElse(200),
      opt.get("temperature").map(_.toDouble).getOrElse(0.8), opt.get("topK").map(_.toInt).getOrElse(40),
      new Random(opt.get("seed").map(_.toLong).getOrElse(0L)))
    println(text)
