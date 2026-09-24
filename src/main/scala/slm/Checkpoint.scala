package slm

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path}

/** パラメータと設定の保存・読み込み。`params.bin` は little-endian の float32（旧版の float64 も読める）。 */
object Checkpoint:

  def save(dir: Path, cfg: Config, params: Array[Float], tokenizer: Tokenizer): Unit =
    Files.createDirectories(dir)
    Files.writeString(dir.resolve("config.txt"),
      s"vocab=${cfg.vocab}\nd=${cfg.d}\nheads=${cfg.heads}\nlayers=${cfg.layers}\ncontext=${cfg.context}\nff=${cfg.ff}\n")
    tokenizer.save(dir.resolve("vocab.txt"))
    val buf = ByteBuffer.allocate(params.length * 4).order(ByteOrder.LITTLE_ENDIAN)
    params.foreach(buf.putFloat)
    Files.write(dir.resolve("params.bin"), buf.array())

  def load(dir: Path): (Config, Array[Float], Tokenizer) =
    val kv = Files.readString(dir.resolve("config.txt")).linesIterator.filter(_.contains("=")).map { l =>
      val Array(k, v) = l.split("=", 2); k -> v.toInt
    }.toMap
    val cfg = Config(kv("vocab"), kv("d"), kv("heads"), kv("layers"), kv("context"), kv("ff"))
    val size = new Layout(cfg).size
    val bytes = Files.readAllBytes(dir.resolve("params.bin"))
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    val params =
      if bytes.length == size * 4L then Array.fill(size)(buf.getFloat)
      else if bytes.length == size * 8L then Array.fill(size)(buf.getDouble.toFloat)
      else throw new IllegalArgumentException(s"params.bin の大きさが合わない: ${bytes.length} bytes, $size params")
    (cfg, params, Tokenizer.load(dir.resolve("vocab.txt")))
