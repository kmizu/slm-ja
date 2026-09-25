package slm

import scala.util.Random

/** C のカーネル（native/libslmkern.so）が Scala のカーネルと bit 単位で同じ logits と勾配を返すことの検証。
  *
  * ライブラリが無い環境（`scripts/build-native.sh` を実行していない、AVX-512 が無い）では飛ばす。
  */
class NativeKernelTest extends munit.FunSuite:

  private def compute(cfg: Config, p: Array[Float], tokens: Array[Int], length: Int, native: Boolean): (Vector[Float], Vector[Float]) =
    val saved = NativeKernels.enabled()
    NativeKernels.setEnabled(native)
    try
      val model = new Model(cfg)
      val ws = model.workspace()
      val g = new Array[Float](p.length)
      model.lossAndGrad(p, g, tokens, 0, length, ws)
      (ws.logits.take((length - 1) * cfg.vocab).toVector, g.toVector)
    finally NativeKernels.setEnabled(saved)

  private val configs = Seq(
    "端数だらけ（幅 39、FF 37、語彙 23、文脈 11）" -> Config(vocab = 23, d = 39, heads = 3, layers = 2, context = 11, ff = 37, attention = "linear"),
    "タイル 2 枚（文脈 70）" -> Config(vocab = 70, d = 64, heads = 4, layers = 1, context = 70, ff = 96, attention = "softmax"))

  for (name, cfg) <- configs do
    test(s"$name: C のカーネルは Scala のカーネルと logits・勾配が bit 一致する") {
      assume(NativeKernels.AVAILABLE && Simd.lanes == 16, s"C のカーネルが使えない: ${NativeKernels.STATUS}")
      val p = new Layout(cfg).init(new Random(21))
      val r = new Random(22)
      var i = 0
      while i < p.length do { p(i) += (r.nextGaussian() * 0.2).toFloat; i += 1 }
      val tokens = Array.fill(cfg.context + 1)(r.nextInt(cfg.vocab))
      for length <- Seq(2, 6, cfg.context + 1) do
        val (scalaLogits, scalaGrad) = compute(cfg, p, tokens, length, native = false)
        val (nativeLogits, nativeGrad) = compute(cfg, p, tokens, length, native = true)
        assertEquals(nativeLogits, scalaLogits, s"logits が不一致（length=$length）")
        assertEquals(nativeGrad, scalaGrad, s"勾配が不一致（length=$length）")
    }

  test("C のカーネルは、ライブラリがあって AVX-512 の環境なら読み込まれている") {
    assume(java.nio.file.Files.exists(java.nio.file.Path.of("native/libslmkern.so")), "ライブラリがまだ作られていない")
    assume(Simd.lanes == 16, "AVX-512 が無い")
    assert(NativeKernels.AVAILABLE, NativeKernels.STATUS)
  }
