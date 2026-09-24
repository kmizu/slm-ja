package slm

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path}

/** パラメータと設定の保存・読み込み。 */
object Checkpoint:

  def save(dir: Path, cfg: Config, params: Array[Double], tokenizer: Tokenizer): Unit =
    Files.createDirectories(dir)
    Files.writeString(dir.resolve("config.txt"),
      s"vocab=${cfg.vocab}\nd=${cfg.d}\nheads=${cfg.heads}\nlayers=${cfg.layers}\ncontext=${cfg.context}\nff=${cfg.ff}\n")
    tokenizer.save(dir.resolve("vocab.txt"))
    val buf = ByteBuffer.allocate(params.length * 8).order(ByteOrder.LITTLE_ENDIAN)
    params.foreach(buf.putDouble)
    Files.write(dir.resolve("params.bin"), buf.array())

  def load(dir: Path): (Config, Array[Double], Tokenizer) =
    val kv = Files.readString(dir.resolve("config.txt")).linesIterator.filter(_.contains("=")).map { l =>
      val Array(k, v) = l.split("=", 2); k -> v.toInt
    }.toMap
    val cfg = Config(kv("vocab"), kv("d"), kv("heads"), kv("layers"), kv("context"), kv("ff"))
    val bytes = Files.readAllBytes(dir.resolve("params.bin"))
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    val params = Array.fill(bytes.length / 8)(buf.getDouble)
    (cfg, params, Tokenizer.load(dir.resolve("vocab.txt")))
