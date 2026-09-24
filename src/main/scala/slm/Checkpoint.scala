package slm

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.channels.FileChannel
import java.nio.file.{Files, Path, StandardCopyOption, StandardOpenOption}
import java.security.MessageDigest
import scala.jdk.CollectionConverters.*

/** 推論用の重みだけの保存形式（旧版と互換）と、学習を再開するための full-state 形式。 */
object Checkpoint:

  private val CHUNK = 4 * 1024 * 1024

  // ---- 重みだけ（推論・initFrom 用） ----

  def save(dir: Path, cfg: Config, params: Array[Float], tokenizer: Tokenizer): Unit =
    Files.createDirectories(dir)
    Files.writeString(dir.resolve("config.txt"),
      s"vocab=${cfg.vocab}\nd=${cfg.d}\nheads=${cfg.heads}\nlayers=${cfg.layers}\ncontext=${cfg.context}\nff=${cfg.ff}\nattention=${cfg.attention}\n")
    tokenizer.save(dir.resolve("vocab.txt"))
    writeFloats(dir.resolve("params.bin"), params)

  def load(dir: Path): (Config, Array[Float], Tokenizer) =
    val kv = readKv(dir.resolve("config.txt"))
    val cfg = Config(kv("vocab").toInt, kv("d").toInt, kv("heads").toInt, kv("layers").toInt, kv("context").toInt, kv("ff").toInt, kv.getOrElse("attention", "softmax"))
    val size = new Layout(cfg).size
    val file = dir.resolve("params.bin")
    val bytes = Files.size(file)
    val params =
      if bytes == size * 4L then readFloats(file, size)
      else if bytes == size * 8L then readDoublesAsFloats(file, size)
      else throw new IllegalArgumentException(s"params.bin の大きさが合わない: $bytes bytes, $size params")
    (cfg, params, Tokenizer.load(dir.resolve("vocab.txt")))

  def readKv(file: Path): Map[String, String] =
    Files.readString(file).linesIterator.filter(_.contains("=")).map { l => val Array(k, v) = l.split("=", 2); k -> v }.toMap

  // ---- 分割 I/O ----

  /** Float 配列を little-endian で逐次書き、SHA-256 を返す。 */
  def writeFloats(file: Path, a: Array[Float]): String =
    val digest = MessageDigest.getInstance("SHA-256")
    val ch = FileChannel.open(file, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)
    try
      val buf = ByteBuffer.allocate(CHUNK).order(ByteOrder.LITTLE_ENDIAN)
      var i = 0
      while i < a.length do
        buf.clear()
        val n = math.min(CHUNK / 4, a.length - i)
        var j = 0
        while j < n do { buf.putFloat(a(i + j)); j += 1 }
        buf.flip()
        digest.update(buf.duplicate())
        while buf.hasRemaining do ch.write(buf)
        i += n
      ch.force(true)
    finally ch.close()
    hex(digest.digest())

  def readFloats(file: Path, count: Int): Array[Float] =
    val out = new Array[Float](count)
    val ch = FileChannel.open(file, StandardOpenOption.READ)
    try
      val buf = ByteBuffer.allocate(CHUNK).order(ByteOrder.LITTLE_ENDIAN)
      var i = 0
      while i < count do
        buf.clear()
        buf.limit(math.min(CHUNK, (count - i) * 4))
        while buf.hasRemaining do
          if ch.read(buf) < 0 then throw new java.io.EOFException(s"$file が短い")
        buf.flip()
        while buf.hasRemaining do { out(i) = buf.getFloat; i += 1 }
    finally ch.close()
    out

  private def readDoublesAsFloats(file: Path, count: Int): Array[Float] =
    val out = new Array[Float](count)
    val ch = FileChannel.open(file, StandardOpenOption.READ)
    try
      val buf = ByteBuffer.allocate(CHUNK).order(ByteOrder.LITTLE_ENDIAN)
      var i = 0
      while i < count do
        buf.clear()
        buf.limit(math.min(CHUNK, (count - i) * 8))
        while buf.hasRemaining do
          if ch.read(buf) < 0 then throw new java.io.EOFException(s"$file が短い")
        buf.flip()
        while buf.hasRemaining do { out(i) = buf.getDouble.toFloat; i += 1 }
    finally ch.close()
    out

  def sha256(file: Path): String =
    val digest = MessageDigest.getInstance("SHA-256")
    val ch = FileChannel.open(file, StandardOpenOption.READ)
    try
      val buf = ByteBuffer.allocate(CHUNK)
      while ch.read(buf) >= 0 do { buf.flip(); digest.update(buf); buf.clear() }
    finally ch.close()
    hex(digest.digest())

  def hex(b: Array[Byte]): String = b.map(x => f"$x%02x").mkString

  // ---- full-state（学習の再開用） ----

  /** 学習を再開するのに要る全部。 */
  final case class TrainState(
      run: RunConfig,
      params: Array[Float],
      m: Array[Float],
      v: Array[Float],
      optimizerStep: Long,     // 完了した update 数
      tokensSeen: Long,
      bestQuickValid: Double,
      bestGeneration: String,
      elapsedSeconds: Double,
      meta: Map[String, String] // corpus/vocab hash, split, eval identity, workers, kernel, commit, platform ...
  )

  val formatVersion = 1

  private def genName(step: Long): String = f"step-$step%08d"

  /** 新しい世代を `<stateDir>/step-XXXXXXXX.partial-<unique>` に書き、検証してから `step-XXXXXXXX` として公開し、`last` を更新する。 */
  def saveState(stateDir: Path, st: TrainState, tokenizer: Tokenizer): Path =
    Files.createDirectories(stateDir)
    val name = genName(st.optimizerStep)
    val partial = stateDir.resolve(s"$name.partial-${System.nanoTime()}")
    Files.createDirectories(partial)
    val hashes = Map(
      "params.bin" -> writeFloats(partial.resolve("params.bin"), st.params),
      "m.bin" -> writeFloats(partial.resolve("m.bin"), st.m),
      "v.bin" -> writeFloats(partial.resolve("v.bin"), st.v))
    tokenizer.save(partial.resolve("vocab.txt"))
    val manifest = new StringBuilder
    manifest.append(s"formatVersion=$formatVersion\n")
    manifest.append(st.run.toText)
    manifest.append(s"optimizerStep=${st.optimizerStep}\ntokensSeen=${st.tokensSeen}\nbestQuickValid=${st.bestQuickValid}\n")
    manifest.append(s"bestGeneration=${st.bestGeneration}\nelapsedSeconds=${st.elapsedSeconds}\n")
    for (k, v) <- st.meta.toSeq.sortBy(_._1) do manifest.append(s"meta.$k=$v\n")
    for (f, h) <- hashes.toSeq.sortBy(_._1) do manifest.append(s"sha256.$f=$h\nlength.$f=${Files.size(partial.resolve(f))}\n")
    manifest.append(s"sha256.vocab.txt=${sha256(partial.resolve("vocab.txt"))}\n")
    Files.writeString(partial.resolve("manifest.txt"), manifest.toString)
    // 書いたものを読み直して検証してから公開
    for (f, h) <- hashes do
      val actual = sha256(partial.resolve(f))
      require(actual == h, s"保存直後の検証に失敗: $f")
    Files.writeString(partial.resolve("COMPLETE"), "ok\n")
    val target = stateDir.resolve(name)
    if Files.exists(target) then deleteRecursively(target)
    try Files.move(partial, target, StandardCopyOption.ATOMIC_MOVE)
    catch case _: java.nio.file.AtomicMoveNotSupportedException => Files.move(partial, target)
    writePointer(stateDir.resolve("last"), name)
    target

  def writePointer(file: Path, name: String): Unit =
    val tmp = file.resolveSibling(file.getFileName.toString + ".tmp")
    Files.writeString(tmp, name + "\n")
    try Files.move(tmp, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
    catch case _: java.nio.file.AtomicMoveNotSupportedException => Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING)

  def readPointer(file: Path): Option[String] =
    if Files.exists(file) then Some(Files.readString(file).trim).filter(_.nonEmpty) else None

  /** 世代ディレクトリが完全で、hash が一致するか。 */
  def isValidGeneration(dir: Path): Boolean =
    try
      if !Files.isDirectory(dir) || !Files.exists(dir.resolve("COMPLETE")) || !Files.exists(dir.resolve("manifest.txt")) then false
      else
        val kv = readKv(dir.resolve("manifest.txt"))
        Seq("params.bin", "m.bin", "v.bin").forall { f =>
          val file = dir.resolve(f)
          Files.exists(file) && Files.size(file) == kv(s"length.$f").toLong && sha256(file) == kv(s"sha256.$f")
        }
    catch case _: Exception => false

  /** 完全な世代を新しい順に列挙する。 */
  def completeGenerations(stateDir: Path): Vector[Path] =
    if !Files.isDirectory(stateDir) then Vector.empty
    else
      Files.list(stateDir).iterator().asScala.toVector
        .filter(p => p.getFileName.toString.matches("step-\\d{8}"))
        .sortBy(_.getFileName.toString)(using Ordering[String].reverse)
        .filter(isValidGeneration)

  /** `last` が指す世代が有効ならそれ、壊れていれば直前の完全な世代を選ぶ。 */
  def resolveLatest(stateDir: Path): Option[Path] =
    val pointed = readPointer(stateDir.resolve("last")).map(stateDir.resolve).filter(isValidGeneration)
    pointed.orElse(completeGenerations(stateDir).headOption)

  def loadState(dir: Path): (TrainState, Tokenizer) =
    require(isValidGeneration(dir), s"世代が不完全か破損している: $dir")
    val kv = readKv(dir.resolve("manifest.txt"))
    require(kv("formatVersion").toInt == formatVersion, s"formatVersion が違う: ${kv("formatVersion")}")
    val run = RunConfig.fromMap(kv)
    val size = new Layout(run.cfg).size
    val st = TrainState(run,
      readFloats(dir.resolve("params.bin"), size), readFloats(dir.resolve("m.bin"), size), readFloats(dir.resolve("v.bin"), size),
      kv("optimizerStep").toLong, kv("tokensSeen").toLong, kv("bestQuickValid").toDouble, kv("bestGeneration"), kv("elapsedSeconds").toDouble,
      kv.collect { case (k, v) if k.startsWith("meta.") => k.drop(5) -> v })
    (st, Tokenizer.load(dir.resolve("vocab.txt")))

  /** 直近 keep 世代と best 以外を消す。 */
  def prune(stateDir: Path, keep: Int, best: String): Unit =
    val gens = completeGenerations(stateDir)
    for g <- gens.drop(keep) if g.getFileName.toString != best do deleteRecursively(g)
    // 書きかけの残骸も消す
    if Files.isDirectory(stateDir) then
      Files.list(stateDir).iterator().asScala.filter(_.getFileName.toString.contains(".partial-")).foreach(deleteRecursively)

  def deleteRecursively(p: Path): Unit =
    if Files.isDirectory(p) then Files.list(p).iterator().asScala.foreach(deleteRecursively)
    Files.deleteIfExists(p)
