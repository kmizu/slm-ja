package slm;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * 密な層の C カーネル（native/libslmkern.so）の呼び出し口。FFM API で、Java の float[] をコピーせずに渡す。
 *
 * <p>ライブラリは {@code -Dslm.native.lib=...}（既定は native/libslmkern.so）から読み込み、{@code -Dslm.native=false} で使わない。
 * 読み込めなければ {@link #AVAILABLE} は false で、Model は Scala のカーネルを使う。
 * 呼び出しは {@code Linker.Option.critical(true)} なので、呼び出しの間は GC が待たされる（1 回は数十ミリ秒以下）。
 */
public final class NativeKernels {
    private NativeKernels() {}

    public static final boolean AVAILABLE;
    public static final String STATUS;
    private static final MethodHandle FORWARD;
    private static final MethodHandle BACKWARD;
    private static volatile boolean enabled;

    static {
        MethodHandle forward = null;
        MethodHandle backward = null;
        boolean ok = false;
        String status;
        try {
            Path lib = Path.of(System.getProperty("slm.native.lib", "native/libslmkern.so"));
            if ("false".equals(System.getProperty("slm.native"))) {
                status = "-Dslm.native=false で無効";
            } else if (!Files.exists(lib)) {
                status = "ライブラリが無い: " + lib.toAbsolutePath() + "（scripts/build-native.sh で作る）";
            } else {
                Linker linker = Linker.nativeLinker();
                SymbolLookup lookup = SymbolLookup.libraryLookup(lib, Arena.global());
                Linker.Option critical = Linker.Option.critical(true);
                forward = linker.downcallHandle(lookup.find("slm_dense_forward").orElseThrow(),
                        FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, JAVA_INT), critical);
                backward = linker.downcallHandle(lookup.find("slm_dense_backward").orElseThrow(),
                        FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT),
                        critical);
                ok = true;
                status = "読み込み済み: " + lib.toAbsolutePath();
            }
        } catch (Throwable e) {
            status = "読み込みに失敗: " + e;
        }
        FORWARD = forward;
        BACKWARD = backward;
        AVAILABLE = ok;
        STATUS = status;
        enabled = ok;
    }

    /** C のカーネルを使うか（ライブラリがあるときだけ true にできる）。 */
    public static boolean enabled() {
        return enabled;
    }

    /** テストで C 版と Scala 版を切り替える。ライブラリが無ければ常に false。 */
    public static void setEnabled(boolean on) {
        enabled = on && AVAILABLE;
    }

    /** y[t][o] = Σ_i W[o][i] x[t][i] + b[o]。W は p[w + o * in + i]、b は p[b + o]。 */
    public static void denseForward(float[] p, int w, int b, int in, int out, float[] x, float[] y, int T) {
        try {
            FORWARD.invokeExact(MemorySegment.ofArray(p), (long) w, (long) b, in, out, MemorySegment.ofArray(x), MemorySegment.ofArray(y), T);
        } catch (Throwable e) {
            throw new IllegalStateException("slm_dense_forward の呼び出しに失敗", e);
        }
    }

    /** dW += Σ_t dy[t] x[t]ᵀ（g[w..]）、db += Σ_t dy[t]（g[b..]）、dx[t] += Wᵀ dy[t]。 */
    public static void denseBackward(float[] p, float[] g, int w, int b, int in, int out, float[] x, float[] dy, float[] dx, int T) {
        try {
            BACKWARD.invokeExact(MemorySegment.ofArray(p), MemorySegment.ofArray(g), (long) w, (long) b, in, out,
                    MemorySegment.ofArray(x), MemorySegment.ofArray(dy), MemorySegment.ofArray(dx), T);
        } catch (Throwable e) {
            throw new IllegalStateException("slm_dense_backward の呼び出しに失敗", e);
        }
    }
}
