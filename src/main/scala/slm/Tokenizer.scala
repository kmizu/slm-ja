package slm

import java.nio.file.{Files, Path}

/** 文字単位のトークナイザ。出現回数が少ない文字はまとめて UNK（id 0）。文字は Scala の `Char` 単位。 */
final class Tokenizer(val chars: Vector[Char]):
  private val toId: Array[Int] =
    val table = new Array[Int](65536)
    chars.zipWithIndex.foreach((c, i) => table(c.toInt) = i + 1)
    table
  val vocabSize: Int = chars.length + 1
  val unk: Int = 0

  def encode(text: String): Array[Int] =
    val out = new Array[Int](text.length)
    var i = 0
    while i < text.length do { out(i) = toId(text.charAt(i).toInt); i += 1 }
    out

  def decode(ids: Iterable[Int]): String = ids.map(id => if id == unk then '�' else chars(id - 1)).mkString

  def save(path: Path): Unit = Files.writeString(path, chars.mkString)

  /** 語彙の同一性（順序込み）を表す短い識別子。 */
  def hash: String = Checkpoint.hex(java.security.MessageDigest.getInstance("SHA-256").digest(chars.mkString.getBytes("UTF-8"))).take(16)

object Tokenizer:

  /** 出現回数 `(-count, char)` 順。同じテキスト・同じ minCount なら同じ語彙。 */
  def fromText(text: String, minCount: Int): Tokenizer =
    val counts = new Array[Long](65536)
    var i = 0
    while i < text.length do { counts(text.charAt(i).toInt) += 1; i += 1 }
    val kept = (0 until 65536).filter(counts(_) >= minCount).map(c => (c.toChar, counts(c))).sortBy((c, n) => (-n, c)).map(_._1)
    new Tokenizer(kept.toVector)

  def load(path: Path): Tokenizer = new Tokenizer(Files.readString(path).toVector)
