package slm

import java.nio.file.{Files, Path}

/** 文字単位のトークナイザ。出現回数が少ない文字はまとめて UNK（id 0）。 */
final class Tokenizer(val chars: Vector[Char]):
  private val toId: Map[Char, Int] = chars.zipWithIndex.map((c, i) => c -> (i + 1)).toMap
  val vocabSize: Int = chars.length + 1
  val unk: Int = 0

  def encode(text: String): Array[Int] = text.map(c => toId.getOrElse(c, unk)).toArray

  def decode(ids: Iterable[Int]): String = ids.map(id => if id == unk then '�' else chars(id - 1)).mkString

  def save(path: Path): Unit = Files.writeString(path, chars.mkString)

object Tokenizer:

  def fromText(text: String, minCount: Int): Tokenizer =
    val counts = text.groupMapReduce(identity)(_ => 1L)(_ + _)
    val kept = counts.toVector.filter(_._2 >= minCount).sortBy((c, n) => (-n, c)).map(_._1)
    new Tokenizer(kept)

  def load(path: Path): Tokenizer = new Tokenizer(Files.readString(path).toVector)
