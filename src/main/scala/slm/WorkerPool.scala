package slm

import java.util.concurrent.{Callable, Executors, ExecutorService, TimeUnit}
import scala.jdk.CollectionConverters.*

/** 固定数の worker スレッド。学習・集約・optimizer の各 phase で共用し、phase の間に barrier を置く。
  * worker の例外は main に伝播する。pool の中から同じ pool へ blocking task を投入しない。
  */
final class WorkerPool(val size: Int):
  require(size >= 1)
  private val exec: ExecutorService = Executors.newFixedThreadPool(size, r => {
    val t = new Thread(r); t.setDaemon(true); t.setName(s"slm-worker-${t.threadId()}"); t
  })

  /** tasks 個の仕事を並列に実行し、全部終わるまで待つ。 */
  def run(tasks: Int)(f: Int => Unit): Unit =
    if size == 1 || tasks == 1 then
      var i = 0
      while i < tasks do { f(i); i += 1 }
    else
      val callables = (0 until tasks).map(i => new Callable[Unit] { def call(): Unit = f(i) }).asJava
      val futures = exec.invokeAll(callables)
      var first: Throwable = null
      futures.forEach { fu =>
        try fu.get()
        catch case e: java.util.concurrent.ExecutionException => if first == null then first = e.getCause
      }
      if first != null then throw first

  /** [0, n) を size 個の連続区間に分けて並列処理する（境界は align の倍数）。 */
  def ranges(n: Int, align: Int = 16)(f: (Int, Int, Int) => Unit): Unit =
    val chunks = WorkerPool.split(n, size, align)
    run(chunks.length)(k => f(k, chunks(k)._1, chunks(k)._2))

  def shutdown(): Unit =
    exec.shutdown()
    exec.awaitTermination(10, TimeUnit.SECONDS)

object WorkerPool:
  def split(n: Int, parts: Int, align: Int): Array[(Int, Int)] =
    val per = math.max(align, ((n + parts - 1) / parts + align - 1) / align * align)
    val out = (0 until parts).map(k => (math.min(n, k * per), math.min(n, (k + 1) * per))).filter((s, e) => e > s)
    if out.isEmpty then Array((0, n)) else out.toArray
