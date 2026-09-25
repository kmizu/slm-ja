/*
 * slm-ja の密な層のカーネル（AVX-512、単精度）。
 *
 * Scala の Model.denseAll / denseBackwardAll（Simd.dot / dot4x2 / axpy / axpy4 / axpy4x2）と
 * 各要素の計算を完全に同じにしてある。積和の順序、16 レーンの合計の畳み方（orderedSum）、
 * 端数の処理、ゼロの勾配の読み飛ばしがすべて同じなので、結果は bit 単位で一致する。
 * 違うのは読み込みと書き戻しの回数だけ:
 *   - 順伝播: 重み 4 行 × トークン 4 本の内積 16 個を同時に（Vector API では JIT の節点予算を超えて遅かった形）
 *   - 重みの勾配: タイルの x を 64 要素の切れ端ごとに詰め替え、4 行 × 64 要素ぶんをレジスタに置いたまま全トークンを足し込む。
 *     行を 64 本ずつの塊に分けて塊の中で切れ端を回すので、重みの勾配はタイルごとに 1 回だけ読み書きする
 *   - 入力の勾配: 重みを 32 行 × 64 要素の切れ端に詰め替え、4 トークン × 64 要素ぶんをレジスタに置いたまま足し込む
 * 詰め替えは写すだけなので、足し込みの順序は変わらない（行列積ライブラリの BLIS と同じ考え方）。
 * 途中でメモリに書き戻すかどうかは丸めに影響しないので、足し込みの順序が同じなら結果は変わらない。
 *
 * ビルド: scripts/build-native.sh（-ffp-contract=off で積和の自動融合を止める。Scala 側の a * b + c は融合されないため）
 */
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define LANES 16
/* トークンのタイル。重みを 1 回読むあいだに何トークンぶん使い回すか。Scala 版（64）より大きくしても、
 * 4 の倍数なら端数のトークンも足し込みの順序も変わらないので結果は bit 一致する。
 * 100M・8 workers の実測: 64 で 351〜374 tok/s、128 で 486〜519、256 で 443〜466。 */
#ifdef TILE_OVERRIDE
#define TILE TILE_OVERRIDE
#else
#define TILE 128
#endif
#define OCHUNK 32

static inline __m512i xor_index(int h) {
    return _mm512_xor_si512(_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0), _mm512_set1_epi32(h));
}

/* Simd.orderedSum と同じ: 半分ずつ入れ替えて足す（16 → 8 → 4 → 2 → 1）。 */
static inline float osum(__m512 v) {
    v = _mm512_add_ps(v, _mm512_permutexvar_ps(xor_index(8), v));
    v = _mm512_add_ps(v, _mm512_permutexvar_ps(xor_index(4), v));
    v = _mm512_add_ps(v, _mm512_permutexvar_ps(xor_index(2), v));
    v = _mm512_add_ps(v, _mm512_permutexvar_ps(xor_index(1), v));
    return _mm512_cvtss_f32(v);
}

/* Simd.dot と同じ（積算 2 本を 32 要素ずつ、残りの 16 要素ずつは 1 本目へ、最後にスカラーで端数）。 */
static inline float dot(const float *a, const float *b, int n) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    int i = 0;
    int bound2 = n - n % (2 * LANES);
    for (; i < bound2; i += 2 * LANES) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + LANES), _mm512_loadu_ps(b + i + LANES), acc1);
    }
    int bound = n - n % LANES;
    for (; i < bound; i += LANES) acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
    float s = osum(_mm512_add_ps(acc0, acc1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

/* 重み R 行（w から n おき）とトークン 4 本（x から n おき）の内積。out[4 * r + t]。各内積は Simd.dot4x2 と同じ計算。 */
static inline __attribute__((always_inline)) void kernel_rx4(const float *w, const float *x, int n, int R, float *out) {
    __m512 acc[4][4];
#pragma GCC unroll 4
    for (int r = 0; r < 4; r++)
#pragma GCC unroll 4
        for (int t = 0; t < 4; t++) acc[r][t] = _mm512_setzero_ps();
    int i = 0;
    int bound = n - n % LANES;
    for (; i < bound; i += LANES) {
        __m512 wv[4];
#pragma GCC unroll 4
        for (int r = 0; r < 4; r++) if (r < R) wv[r] = _mm512_loadu_ps(w + (int64_t)r * n + i);
#pragma GCC unroll 4
        for (int t = 0; t < 4; t++) {
            __m512 xv = _mm512_loadu_ps(x + (int64_t)t * n + i);
#pragma GCC unroll 4
            for (int r = 0; r < 4; r++) if (r < R) acc[r][t] = _mm512_fmadd_ps(xv, wv[r], acc[r][t]);
        }
    }
#pragma GCC unroll 4
    for (int r = 0; r < 4; r++)
#pragma GCC unroll 4
        for (int t = 0; t < 4; t++) if (r < R) out[4 * r + t] = osum(acc[r][t]);
    for (; i < n; i++) {
        for (int r = 0; r < R; r++) {
            float wr = w[(int64_t)r * n + i];
            for (int t = 0; t < 4; t++) out[4 * r + t] += wr * x[(int64_t)t * n + i];
        }
    }
}

static void kernel_4x4(const float *w, const float *x, int n, float *out) { kernel_rx4(w, x, n, 4, out); }
static void kernel_2x4(const float *w, const float *x, int n, float *out) { kernel_rx4(w, x, n, 2, out); }

/* y[t][o] = Σ_i W[o][i] x[t][i] + b[o]（Model.denseAll と同じ結果）。p は重み全体、w と b はその中の位置。 */
void slm_dense_forward(const float *p, int64_t w, int64_t b, int in, int out, const float *x, float *y, int T) {
    float o16[16];
    for (int t0 = 0; t0 < T; t0 += TILE) {
        int tEnd = t0 + TILE < T ? t0 + TILE : T;
        int o = 0;
        for (; o + 3 < out; o += 4) {
            const float *row0 = p + w + (int64_t)o * in;
            int t = t0;
            for (; t + 3 < tEnd; t += 4) {
                kernel_4x4(row0, x + (int64_t)t * in, in, o16);
                for (int r = 0; r < 4; r++) {
                    float bias = p[b + o + r];
                    for (int tt = 0; tt < 4; tt++) y[(int64_t)(t + tt) * out + o + r] = bias + o16[4 * r + tt];
                }
            }
            for (; t < tEnd; t++)
                for (int r = 0; r < 4; r++)
                    y[(int64_t)t * out + o + r] = p[b + o + r] + dot(row0 + (int64_t)r * in, x + (int64_t)t * in, in);
        }
        for (; o + 1 < out; o += 2) {
            const float *row0 = p + w + (int64_t)o * in;
            int t = t0;
            for (; t + 3 < tEnd; t += 4) {
                kernel_2x4(row0, x + (int64_t)t * in, in, o16);
                for (int r = 0; r < 2; r++) {
                    float bias = p[b + o + r];
                    for (int tt = 0; tt < 4; tt++) y[(int64_t)(t + tt) * out + o + r] = bias + o16[4 * r + tt];
                }
            }
            for (; t < tEnd; t++)
                for (int r = 0; r < 2; r++)
                    y[(int64_t)t * out + o + r] = p[b + o + r] + dot(row0 + (int64_t)r * in, x + (int64_t)t * in, in);
        }
        for (; o < out; o++) {
            const float *row = p + w + (int64_t)o * in;
            for (int t = t0; t < tEnd; t++) y[(int64_t)t * out + o] = p[b + o] + dot(row, x + (int64_t)t * in, in);
        }
    }
}

/* スレッドごとの作業域（勾配の印、詰め替えた x、詰め替えた重み）。足りなければ確保し直して使い回す。 */
typedef struct { void *ptr; size_t cap; } scratch_t;
static __thread scratch_t s_nz, s_xp, s_wp;

static void *scratch(scratch_t *s, size_t bytes) {
    if (bytes > s->cap) {
        free(s->ptr);
        size_t rounded = (bytes + 63) & ~(size_t)63;
        s->ptr = aligned_alloc(64, rounded);
        s->cap = s->ptr ? rounded : 0;
    }
    return s->ptr;
}

#define NQMAX (TILE / 4)
#define MC 64
#define GROUP (4 * LANES)

/*
 * 重みの勾配: 行 o..o+R-1（R <= 4）の要素 [i, i + 16K)（K <= 4）に、タイル内のトークンを順に足す。
 * トークン t の x は xb + (t - t0) * xs（詰め替えた切れ端なら xs = 64、元の配列なら xs = in）。
 * 行ごとに、トークン 4 本の組（印が立っていれば）を axpy4 と同じ積和の鎖で、残りのトークンは axpy と同じく 1 本ずつ。
 */
static inline __attribute__((always_inline)) void dw_block(float *g, int64_t w, int in, int out, const float *xb, int64_t xs,
                                                          const float *dy, int t0, int tEnd, int nq, const unsigned char *nz,
                                                          int o, int R, int i, int K) {
    int tq = t0 + 4 * nq;
    float *grow[4];
    for (int r = 0; r < 4; r++) grow[r] = g + w + (int64_t)(o + (r < R ? r : 0)) * in + i;
    __m512 acc[4][4];
#pragma GCC unroll 4
    for (int r = 0; r < 4; r++)
#pragma GCC unroll 4
        for (int k = 0; k < 4; k++) if (r < R && k < K) acc[r][k] = _mm512_loadu_ps(grow[r] + k * LANES);
    for (int q = 0; q < nq; q++) {
        for (int tt = 0; tt < 4; tt++) {
            int t = t0 + 4 * q + tt;
            const float *xr = xb + (int64_t)(t - t0) * xs;
            __m512 xv[4];
#pragma GCC unroll 4
            for (int k = 0; k < 4; k++) if (k < K) xv[k] = _mm512_loadu_ps(xr + k * LANES);
#pragma GCC unroll 4
            for (int r = 0; r < 4; r++)
                if (r < R && nz[(o + r) * NQMAX + q]) {
                    __m512 gv = _mm512_set1_ps(dy[(int64_t)t * out + o + r]);
#pragma GCC unroll 4
                    for (int k = 0; k < 4; k++) if (k < K) acc[r][k] = _mm512_fmadd_ps(xv[k], gv, acc[r][k]);
                }
        }
    }
    for (int t = tq; t < tEnd; t++) {
        const float *xr = xb + (int64_t)(t - t0) * xs;
        __m512 xv[4];
#pragma GCC unroll 4
        for (int k = 0; k < 4; k++) if (k < K) xv[k] = _mm512_loadu_ps(xr + k * LANES);
#pragma GCC unroll 4
        for (int r = 0; r < 4; r++) {
            if (r >= R) continue;
            float gy = dy[(int64_t)t * out + o + r];
            if (gy != 0.0f) {
                __m512 gv = _mm512_set1_ps(gy);
#pragma GCC unroll 4
                for (int k = 0; k < 4; k++) if (k < K) acc[r][k] = _mm512_fmadd_ps(xv[k], gv, acc[r][k]);
            }
        }
    }
#pragma GCC unroll 4
    for (int r = 0; r < 4; r++)
#pragma GCC unroll 4
        for (int k = 0; k < 4; k++) if (r < R && k < K) _mm512_storeu_ps(grow[r] + k * LANES, acc[r][k]);
}

static void dw_block4x4(float *g, int64_t w, int in, int out, const float *xb, int64_t xs, const float *dy, int t0, int tEnd, int nq,
                        const unsigned char *nz, int o, int i) {
    dw_block(g, w, in, out, xb, xs, dy, t0, tEnd, nq, nz, o, 4, i, 4);
}
static void dw_block1x4(float *g, int64_t w, int in, int out, const float *xb, int64_t xs, const float *dy, int t0, int tEnd, int nq,
                        const unsigned char *nz, int o, int i) {
    dw_block(g, w, in, out, xb, xs, dy, t0, tEnd, nq, nz, o, 1, i, 4);
}
static void dw_block4x1(float *g, int64_t w, int in, int out, const float *xb, int64_t xs, const float *dy, int t0, int tEnd, int nq,
                        const unsigned char *nz, int o, int i) {
    dw_block(g, w, in, out, xb, xs, dy, t0, tEnd, nq, nz, o, 4, i, 1);
}
static void dw_block1x1(float *g, int64_t w, int in, int out, const float *xb, int64_t xs, const float *dy, int t0, int tEnd, int nq,
                        const unsigned char *nz, int o, int i) {
    dw_block(g, w, in, out, xb, xs, dy, t0, tEnd, nq, nz, o, 1, i, 1);
}

/*
 * 重みの勾配と bias の勾配（Model.denseBackwardAll の dW パスと同じ結果）。
 * タイルの x を 64 要素の切れ端ごとの連続した並びに詰め替え、行を 64 本ずつの塊に分けて、塊の中で切れ端を回す。
 * 重みの勾配の各要素はタイルごとに 1 回だけ読み書きし、x の切れ端（64 トークン × 64 要素 = 16 KB）は L1 に載る。
 */
static void dw_tile(float *g, int64_t w, int64_t b, int in, int out, const float *x, const float *dy, int t0, int tEnd) {
    int TT = tEnd - t0;
    int nq = TT / 4;
    int tq = t0 + 4 * nq;
    unsigned char *nz = (unsigned char *)scratch(&s_nz, (size_t)out * NQMAX);
    for (int o = 0; o < out; o++) {
        for (int q = 0; q < nq; q++) {
            int t = t0 + 4 * q;
            float g0 = dy[(int64_t)t * out + o], g1 = dy[(int64_t)(t + 1) * out + o];
            float g2 = dy[(int64_t)(t + 2) * out + o], g3 = dy[(int64_t)(t + 3) * out + o];
            unsigned char f = g0 != 0.0f || g1 != 0.0f || g2 != 0.0f || g3 != 0.0f;
            nz[o * NQMAX + q] = f;
            if (f) g[b + o] += g0 + g1 + g2 + g3;
        }
        for (int t = tq; t < tEnd; t++) {
            float gy = dy[(int64_t)t * out + o];
            if (gy != 0.0f) g[b + o] += gy;
        }
    }
    int quadEnd = out - out % 4;
    int bound = in - in % LANES;
    int n64 = bound - bound % GROUP;
    int ngroups = n64 / GROUP;
    float *xp = (float *)scratch(&s_xp, sizeof(float) * ((size_t)ngroups * TILE * GROUP + 1));
    for (int tt = 0; tt < TT; tt++)
        for (int ig = 0; ig < ngroups; ig++)
            memcpy(xp + ((size_t)ig * TILE + tt) * GROUP, x + (int64_t)(t0 + tt) * in + (int64_t)ig * GROUP, sizeof(float) * GROUP);
    const float *xt = x + (int64_t)t0 * in;
    for (int ob = 0; ob < quadEnd; ob += MC) {
        int obEnd = ob + MC < quadEnd ? ob + MC : quadEnd;
        for (int ig = 0; ig < ngroups; ig++) {
            const float *slice = xp + (size_t)ig * TILE * GROUP;
            for (int o = ob; o < obEnd; o += 4) dw_block4x4(g, w, in, out, slice, GROUP, dy, t0, tEnd, nq, nz, o, ig * GROUP);
        }
        for (int i = n64; i < bound; i += LANES)
            for (int o = ob; o < obEnd; o += 4) dw_block4x1(g, w, in, out, xt + i, in, dy, t0, tEnd, nq, nz, o, i);
    }
    for (int o = quadEnd; o < out; o++) {
        for (int ig = 0; ig < ngroups; ig++)
            dw_block1x4(g, w, in, out, xp + (size_t)ig * TILE * GROUP, GROUP, dy, t0, tEnd, nq, nz, o, ig * GROUP);
        for (int i = n64; i < bound; i += LANES) dw_block1x1(g, w, in, out, xt + i, in, dy, t0, tEnd, nq, nz, o, i);
    }
    /* スカラーの端数の要素（in % 16 個）: axpy4 / axpy の端数処理と同じ式 */
    for (int i = bound; i < in; i++) {
        for (int o = 0; o < out; o++) {
            float *gr = g + w + (int64_t)o * in;
            float v = gr[i];
            for (int q = 0; q < nq; q++) {
                if (!nz[o * NQMAX + q]) continue;
                int t = t0 + 4 * q;
                float a0 = dy[(int64_t)t * out + o], a1 = dy[(int64_t)(t + 1) * out + o];
                float a2 = dy[(int64_t)(t + 2) * out + o], a3 = dy[(int64_t)(t + 3) * out + o];
                v += a0 * x[(int64_t)t * in + i] + a1 * x[(int64_t)(t + 1) * in + i] + a2 * x[(int64_t)(t + 2) * in + i] +
                     a3 * x[(int64_t)(t + 3) * in + i];
            }
            for (int t = tq; t < tEnd; t++) {
                float gy = dy[(int64_t)t * out + o];
                if (gy != 0.0f) v += gy * x[(int64_t)t * in + i];
            }
            gr[i] = v;
        }
    }
}

/*
 * 入力の勾配: トークン t..t+TT-1（TT <= 4）の要素 [i, i + 16K) に、重み行 [oBegin, oEnd)（4 行の組のみ）を順に足す。
 * 行 o の重みは wb + (o - oBegin) * ws（詰め替えた塊なら ws = 64、元の配列なら ws = in）。
 * トークンごとに、4 行の組（どれかの勾配が 0 でなければ）を axpy4 と同じ積和の鎖で。
 */
static inline __attribute__((always_inline)) void dx_block(const float *wb, int64_t ws, int in, int out, const float *dy, float *dx,
                                                          int t, int TT, int oBegin, int oEnd, int i, int K) {
    float *drow[4];
    for (int tt = 0; tt < 4; tt++) drow[tt] = dx + (int64_t)(t + (tt < TT ? tt : 0)) * in + i;
    __m512 acc[4][4];
#pragma GCC unroll 4
    for (int tt = 0; tt < 4; tt++)
#pragma GCC unroll 4
        for (int k = 0; k < 4; k++) if (tt < TT && k < K) acc[tt][k] = _mm512_loadu_ps(drow[tt] + k * LANES);
    for (int o = oBegin; o < oEnd; o += 4) {
        float gq[4][4];
        unsigned char nzq[4];
#pragma GCC unroll 4
        for (int tt = 0; tt < 4; tt++) {
            if (tt >= TT) { nzq[tt] = 0; continue; }
            const float *gr = dy + (int64_t)(t + tt) * out + o;
            gq[tt][0] = gr[0]; gq[tt][1] = gr[1]; gq[tt][2] = gr[2]; gq[tt][3] = gr[3];
            nzq[tt] = gr[0] != 0.0f || gr[1] != 0.0f || gr[2] != 0.0f || gr[3] != 0.0f;
        }
#pragma GCC unroll 4
        for (int j = 0; j < 4; j++) {
            const float *wr = wb + (int64_t)(o + j - oBegin) * ws;
            __m512 wv[4];
#pragma GCC unroll 4
            for (int k = 0; k < 4; k++) if (k < K) wv[k] = _mm512_loadu_ps(wr + k * LANES);
#pragma GCC unroll 4
            for (int tt = 0; tt < 4; tt++)
                if (nzq[tt]) {
                    __m512 gv = _mm512_set1_ps(gq[tt][j]);
#pragma GCC unroll 4
                    for (int k = 0; k < 4; k++) if (k < K) acc[tt][k] = _mm512_fmadd_ps(wv[k], gv, acc[tt][k]);
                }
        }
    }
#pragma GCC unroll 4
    for (int tt = 0; tt < 4; tt++)
#pragma GCC unroll 4
        for (int k = 0; k < 4; k++) if (tt < TT && k < K) _mm512_storeu_ps(drow[tt] + k * LANES, acc[tt][k]);
}

static void dx_block_t4k4(const float *wb, int64_t ws, int in, int out, const float *dy, float *dx, int t, int oB, int oE, int i) {
    dx_block(wb, ws, in, out, dy, dx, t, 4, oB, oE, i, 4);
}
static void dx_block_tnk4(const float *wb, int64_t ws, int in, int out, const float *dy, float *dx, int t, int TT, int oB, int oE, int i) {
    dx_block(wb, ws, in, out, dy, dx, t, TT, oB, oE, i, 4);
}
static void dx_block_tnk1(const float *wb, int64_t ws, int in, int out, const float *dy, float *dx, int t, int TT, int oB, int oE, int i) {
    dx_block(wb, ws, in, out, dy, dx, t, TT, oB, oE, i, 1);
}

/*
 * 入力の勾配（Model.denseBackwardAll の dx パスと同じ結果）。要素ごとに、4 行の組を全部足してから端数の行を 1 行ずつ足す。
 * 重みを 32 行の塊ごとに、64 要素の切れ端ごとの連続した並び（32 行 × 64 要素 = 8 KB）に詰め替えてから使う。
 */
static void dx_tile(const float *p, int64_t w, int in, int out, const float *dy, float *dx, int t0, int tEnd) {
    int quadEnd = out - out % 4;
    int bound = in - in % LANES;
    int n64 = bound - bound % GROUP;
    int ngroups = n64 / GROUP;
    float *wp = (float *)scratch(&s_wp, sizeof(float) * ((size_t)ngroups * OCHUNK * GROUP + 1));
    for (int oc = 0; oc < quadEnd; oc += OCHUNK) {
        int ocEnd = oc + OCHUNK < quadEnd ? oc + OCHUNK : quadEnd;
        for (int j = 0; j < ocEnd - oc; j++)
            for (int ig = 0; ig < ngroups; ig++)
                memcpy(wp + ((size_t)ig * OCHUNK + j) * GROUP, p + w + (int64_t)(oc + j) * in + (int64_t)ig * GROUP, sizeof(float) * GROUP);
        for (int ig = 0; ig < ngroups; ig++) {
            const float *slice = wp + (size_t)ig * OCHUNK * GROUP;
            for (int t = t0; t < tEnd; t += 4) {
                int TT = tEnd - t < 4 ? tEnd - t : 4;
                if (TT == 4) dx_block_t4k4(slice, GROUP, in, out, dy, dx, t, oc, ocEnd, ig * GROUP);
                else dx_block_tnk4(slice, GROUP, in, out, dy, dx, t, TT, oc, ocEnd, ig * GROUP);
            }
        }
        for (int i = n64; i < bound; i += LANES)
            for (int t = t0; t < tEnd; t += 4)
                dx_block_tnk1(p + w + (int64_t)oc * in + i, in, in, out, dy, dx, t, tEnd - t < 4 ? tEnd - t : 4, oc, ocEnd, i);
    }
    /* 端数の行（out % 4 本）: 4 行の組のあとで、axpy と同じく 1 行ずつ足す */
    for (int o = quadEnd; o < out; o++) {
        const float *row = p + w + (int64_t)o * in;
        for (int t = t0; t < tEnd; t++) {
            float gy = dy[(int64_t)t * out + o];
            if (gy == 0.0f) continue;
            float *dr = dx + (int64_t)t * in;
            __m512 gv = _mm512_set1_ps(gy);
            for (int k = 0; k < bound; k += LANES)
                _mm512_storeu_ps(dr + k, _mm512_fmadd_ps(_mm512_loadu_ps(row + k), gv, _mm512_loadu_ps(dr + k)));
        }
    }
    /* スカラーの端数の要素（in % 16 個）: 4 行の組を順に、そのあと端数の行（axpy4 / axpy の端数処理と同じ式） */
    if (bound < in) {
        for (int t = t0; t < tEnd; t++) {
            float *dr = dx + (int64_t)t * in;
            const float *gr = dy + (int64_t)t * out;
            for (int k = bound; k < in; k++) {
                float v = dr[k];
                for (int o = 0; o < quadEnd; o += 4) {
                    float a0 = gr[o], a1 = gr[o + 1], a2 = gr[o + 2], a3 = gr[o + 3];
                    if (a0 != 0.0f || a1 != 0.0f || a2 != 0.0f || a3 != 0.0f) {
                        const float *r0 = p + w + (int64_t)o * in;
                        v += a0 * r0[k] + a1 * r0[in + k] + a2 * r0[2 * (int64_t)in + k] + a3 * r0[3 * (int64_t)in + k];
                    }
                }
                for (int o = quadEnd; o < out; o++) {
                    float gy = gr[o];
                    if (gy != 0.0f) v += gy * p[w + (int64_t)o * in + k];
                }
                dr[k] = v;
            }
        }
    }
}

/* Model.denseBackwardAll と同じ結果: dW += Σ_t dy[t] x[t]ᵀ、db += Σ_t dy[t]、dx[t] += Wᵀ dy[t]（タイルごとに dW → dx）。 */
void slm_dense_backward(const float *p, float *g, int64_t w, int64_t b, int in, int out, const float *x, const float *dy, float *dx,
                        int T) {
    for (int t0 = 0; t0 < T; t0 += TILE) {
        int tEnd = t0 + TILE < T ? t0 + TILE : T;
        dw_tile(g, w, b, in, out, x, dy, t0, tEnd);
        dx_tile(p, w, in, out, dy, dx, t0, tEnd);
    }
}
