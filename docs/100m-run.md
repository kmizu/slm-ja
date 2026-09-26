# 100M run の記録

指示書 `slm-ja-100m-claude-code-instructions.md` に沿って、約 100M パラメータのモデルを再開可能な状態で初回の学習予算（33.5M トークン）まで回した記録。
実装方針（手書き Scala・Float32・JDK Vector API・行列ライブラリなし）は維持している。

## 実行環境

| 項目 | 値 |
|---|---|
| CPU | AMD Ryzen 7 8700G（8 コア / 16 スレッド、AVX-512、L1d 32 KiB×8、L2 1 MiB×8、L3 16 MiB） |
| RAM | 30 GiB（空き 26 GiB） |
| OS / JDK / Scala | WSL2 Linux / OpenJDK 25.0.1 / Scala 3.3.6、`Simd.lanes = 16` |
| ディスク | 空き 12 GiB（checkpoint は 1 世代 1.15 GiB、直近 2 世代 + best を保持） |

## 入力の固定（manifest）

| 項目 | 値 |
|---|---|
| コーパス | `data/corpus.txt` 71,026,284 文字（青空文庫 3,911 作品）、SHA-256 `8007ba16a849e7e5f9ab29ba3b11e9ebdfc205eae794afcd875e50f66d759196` |
| 語彙 | 10M の `checkpoints/ja10m/vocab.txt` を固定（4,134 = 4,133 字 + UNK）、hash `5218c7d796ad94b0`、UNK 率 0.037% |
| split | 先頭 97% = 68,895,495 トークンが学習、末尾 2,130,789 が検証 |
| 評価窓 | quick: 検証領域に均等配置した 64 窓（`even64-v1`）、final: 1,024 窓（`even1024-v1`）。旧ログ（末尾 3% の先頭 64 窓）とは別の集合 |

## モデルと学習設定

| 項目 | 値 |
|---|---|
| shape | d=768 / heads=12 (headDim 64) / layers=14 / ff=3072 / context=256、埋め込みと出力ヘッド共有 |
| 注意 | **線形注意**（φ(x)=elu(x)+1、ヘッドごとの減衰 γ_h = 1 − 2^−(5+h)、状態 64×64、系列長に対して O(n)）。指示書の softmax 注意からの変更はユーザー指示による |
| パラメータ数 | 102,607,398（Preflight の式 `P = V·d + T·d + L·(4d² + 2dF + 9d + F) + 2d + V` と Layout が一致） |
| 予算 | 4,096 updates × 32 系列 × 256 文字 = 33,554,432 トークン |
| lr / warmup / floor | 3e-4 / 256 / 3e-5（cosine、最終 update で floor に一致） |
| AdamW | β=(0.9, 0.95)、eps 1e-8、weight decay 0.1（重み行列のみ）、global clip 1.0 |
| seed | 0（初期化と、step ごとの窓抽出 `SplittableRandom(mix64(seed, step))` を分離） |

## 実装の変更点（指示書の Task 対応）

- **Task 2** 評価は `Model.lossOnly`（forward のみ、勾配配列なし）。`lossOnly == lossAndGrad の loss` と params 不変をテスト。evalEvery / saveEvery / saveSeconds / sampleEvery を分離。
- **Task 3** full-state checkpoint（params, m, v を 4 MiB 分割 I/O で保存、SHA-256 と長さを manifest に記録、`COMPLETE` マーカー、`step-XXXXXXXX.partial-<unique>` → atomic move → `last` pointer 更新）。切り詰め・hash 不一致・pointer 先の欠落を注入して直前世代へ戻ることをテスト。7 step → 保存 → 別インスタンスで読込 → 3 step が、連続 10 step と **bit 一致**（`TrainingTest`）。
  - bit 一致のために SIMD の縮約を `reduceLanes(ADD)` から順序固定の木構造加算に変更（JIT の段階で丸めが変わるのを防ぐ）。1M 設定で 28,000 → 25,000 tok/s のコスト。
- **Task 4** 固定 worker pool（`WorkerPool`）、勾配集約を「パラメータ区間 × worker 順」の並列に、AdamW を `beginStep` + `updateRange` に分け、clip は読み込み時に掛ける。`Boolean[P]` の decay mask は重み区間のリストに置換。直列と並列の一致をテスト。
- **Task 5** 密なカーネルにトークンタイル（64）を追加。x / dx のタイルを L2 に置いたまま全ニューロンを回す。
- **Preflight** で P・Float[P]・Workspace・worker 数ごとの主要配列・heap 比を Long で計算し、未知 CLI キー等を拒否。

## 計測（100M 実形状、`BenchTrain`、同一 CPU・同一形状・batch 32、steps 3..5 の中央値）

| 条件 | fb (ms) | reduce | adam | zero | 1 step | tok/s | RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 線形注意、16 workers、旧カーネル、heap 14g | 77,306 | 539 | 203 | 879 | 78,854 | 104 | 12.0 GiB |
| 線形注意、8 workers、旧カーネル、heap 12g | 24,315 | 255 | 184 | 417 | 25,172 | 325 | 7.3 GiB |
| 線形注意、8 workers、**タイル化カーネル** | 22,042 | 257 | 191 | 408 | 22,899 | **358** | 7.3 GiB |
| 線形注意、4 workers、タイル化 | 32,506 | 158 | 187 | 205 | 33,049 | 248 | – |
| softmax 注意、16 workers、heap 14g | – | – | – | – | OOM | – | – |

- 16 workers は SMT の 2 スレッドが L2 を取り合ってキャッシュ律速になり、8 workers の 1/3 の速度。**採用は 8 workers**。
- 集約と AdamW は 1 step の 2% 未満で、律速は forward/backward。
- 358 tok/s × 0.63 GFLOP/token ≈ 225 GFLOPS（attention・softmax・LayerNorm・Adam を除く概算）。
- 初回予算の見積もり: 33.5M tokens / 358 tok/s ≈ 26 時間（eval・save を除く）。

## pilot（100 steps、`stopAfterSteps=100`、`runs/ja100m`）

| 項目 | 実測 |
|---|---|
| wall throughput | 359 tok/s（JIT 立ち上がりを含む 100 steps、eval・save 込み）、rolling 336〜375 tok/s |
| 1 step | fb 21.0〜21.8 s、reduce 0.26 s、adam 0.19 s、zero 0.42 s |
| loss | train 8.48 → 4.54（step 100）、quick64 5.218（step 50）→ 4.487（step 100） |
| grad norm / clip | 2.3〜4.9、clip 100%（warmup 中、lr 1.2e-4 以下） |
| save | 世代 1 つ 6.5〜16 s（同じ step で best と periodic を二重保存していたのを修正） |
| eval quick64 | 11.5 s |
| RSS | 9.8〜11.2 GiB（heap 12 GiB）、GC ログ `runs/ja100m/gc-*.log` |
| 停止 | step 100 で `interrupted`、`state/step-00000100`（best も同じ）、LOCK 解放 |
| resume | `scripts/resume-100m.sh run=runs/ja100m threads=8 stopAfterSteps=5` → step 100 → 105、tokensSeen 819,200 → 860,160、`state/step-00000105`、best pointer は step-100 のまま |

pilot と同じコード・kernel・workers・設定で続けるため、本番は step 105 の世代から resume する（別 run として初期化はしない）。

## 本番（`scripts/resume-100m.sh run=runs/ja100m threads=8 evalEvery=250 saveEvery=250 saveSeconds=600`）

pilot の step 105 の世代から再開し、初回予算の 4,096 updates まで完走した（`status=completed`）。

| 項目 | 実測 |
|---|---|
| 期間 | 2026-09-25 03:06 〜 2026-09-26 04:15 ごろ（JST） |
| 学習時間 | 95,810 s（26.6 時間）。うち save 2,377 s、eval 633 s |
| wall throughput | 343 tok/s（全体平均） |
| 1 step | fb 19〜21 s（単独時）、reduce 0.25 s、adam 0.18 s、zero 0.40 s |
| grad norm / clip | 終盤 0.7〜0.8、clip 6〜10% |
| RSS | 11.5 GiB（heap 12 GiB） |
| 障害・再開 | 本番中の異常終了なし。再開は pilot 後の 2 回（step 100、step 105）だけ |
| 他ジョブとの競合 | 2026-09-25 18:00 〜 翌 00:30 ごろ、同じマシンで別の CPU ジョブが走り（load 最大 27）、fb が 35〜45 s、rolling が 167〜203 tok/s に落ちた。平均 343 tok/s はこの区間を含む |

### 評価損失の推移（quick64、`even64-v1`）

| step | 50 | 250 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4096 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| loss | 5.218 | 4.014 | 3.639 | 3.323 | 3.154 | 3.049 | 2.974 | 2.899 | 2.864 | **2.8375** | 2.8379 |

一度も上がらずに下がり続け、lr が floor（3e-5）に着いた最後の 100 step でほぼ平らになった。

### 共通の評価条件での比較（`slm.Evaluate`、同じコーパス・語彙・split・窓集合）

| モデル | 消費トークン | quick64 (even64-v1) | final1024 (even1024-v1) |
|---|---:|---:|---:|
| 10M `checkpoints/ja10m`（softmax 注意、旧 run） | 33.5M | 2.880 | 2.872 |
| 100M step 750（線形注意、途中） | 6.1M | 3.433 | 3.431 |
| 100M best = step 4000 | 32.8M | **2.8375** | 2.8317 |
| **100M final = step 4096** | 33.5M | 2.8379 | **2.8305** |

同じトークン数（33.5M）で、100M は 10M を final1024 で 0.042 nat/文字下回った。quick64 で選んだ best（step 4000）は final1024 では final（step 4096）より 0.0012 悪く、64 窓での 0.0004 の差は窓の揺らぎの範囲だった。推論用には `export-final` を使う。10M を追い抜いたのは step 3,500 前後（約 28.7M トークン）。
10M の旧ログの 2.70 は「末尾 3% の先頭 64 窓」で、上の値とは窓集合が違う。

### 生成（`runs/ja100m/export-final`、温度 0.7、topK 40）

固定プロンプト 3 本 × seed 6 本（0, 1, 2, 3, 4, 5）の 18 本を引いた。seed 3 は途中の checkpoint と同じ条件。

```text
　私はその時、いつものように話をきいていた。――あの娘達の声がすこし低くなかった。
　私は、「えらいました。」と言った。
　彼は顔を見据えて、その前に首肯いた。
「ちくしょう。」と私は言った。
「ありがとう。」
　私は答えた。
```
（seed 0）

```text
「おい、どこで」
　と、いつか、小腰に、それを、どうかすると、小袖の柄になって、
「いや、そんなものではない」
　と、小次郎は、ひとりで、
「小次郎どの」
　と、いった。
　それからは、小次郎が、お綱のうしろにかくれて、いま、それを呼び、自分の小田原を出て行った。
```
（seed 4）

```text
　夜になると、女はその女を見ると、一人はその上で、
「これはあなたのお家には、どうしておきますか」と、いいつけるように訊いた。
「あなたのことは、どうせ、あなたは、あなたをお疑いすることが出来ないといったものか」
　女はしかし、ほんとうにお答えなかった。
```
（seed 5）

- 鉤括弧の開閉、字下げ、「と、いった。」で地の文に戻る形は 18 本すべてで崩れていない。
- 作家ごとの文体の塊が出る。seed 4 は小次郎・お綱・小田原で時代小説、step 3,941 の途中出力では「小林君」で少年探偵もの、seed 5 は「〜でございます」の敬語の会話。
- 同じ句を 3 回以上そのまま繰り返すループは 18 本中 0 本。step 3,941 の途中 checkpoint（seed 3）では「部屋の中の部屋だった」「部屋があって、部屋があった」のループが出ていた。代名詞（「おれは、おれの…」）が過剰に続くものは 18 本中 2 本。
- 意味の一貫性は文の単位までで、段落全体の筋はまだ通らない。

## 延長（1 エポックまで、warm restart、進行中）

初回予算の最終世代（step 4096）から、学習領域の 1 エポック（68,895,495 トークン ÷ 8,192 = 8,410 updates）まで延長している。完走した `runs/ja100m` は触らず、続きを `runs/ja100m-e1` に書く。

```bash
SLM_HEAP=12g scripts/run-100m.sh resume=runs/ja100m out=runs/ja100m-e1 corpus=data/corpus.txt threads=8 \
  extendTo=8410 restartWarmup=128 restartLr=1.5e-4 evalEvery=250 saveEvery=250 saveSeconds=1800 sampleEvery=0 \
  stopFile=runs/ja100m-e1/STOP
```

- 学習率: update 4096 までは元の cosine のまま（`RunConfig.lr` は延長前と同じ値を返す）。4096 から 128 updates で floor 3e-5 → 1.5e-4 に予熱し、もう一度 cosine で update 8409 の floor 3e-5 まで下ろす。ピークを元の半分にしたのは、再予熱で一時的に損失が跳ねる幅を抑えるため。
- 延長の設定（`restartAt` / `restartWarmup` / `restartPeakLr`）は checkpoint の manifest に保存される。止まったら `scripts/resume-100m.sh run=runs/ja100m-e1` で、保存済みの延長設定のまま続きから再開する。延長は 1 回だけ対応。
- 窓の抽出は `(seed, step)` から決まるので、延長区間の step 4096 以降は初回予算と重ならない新しい乱数列になる。
- テスト（`TrainingTest`）: 延長前の区間の学習率が元と完全に一致、延長区間の予熱と最終 floor、設定テキストの往復と延長なしの旧設定の読み込み、完走 state からの延長学習と世代への保存。
- ディスク: 延長 run は保存中に最大 4 世代（1 世代 1.15 GiB）を抱える。空き 5.4 GiB では足りないので、`runs/ja100m/state/step-00004087`（最終世代の 9 step 前の定期保存）を消した。best（4000）と final（4096）は残している。

## 生成の高速化（`Decoder`）

`Generate.sample` は、1 文字ごとに直近 256 文字の順伝播をやり直していた（1 文字あたり文脈長ぶんの計算）。
`Decoder` は層ごとの状態を持ち回して、足した 1 文字ぶんだけ計算する。

- 線形注意: ヘッドごとの s（64×64）と z（64）を減衰させてから今の文字を足す。1 文字あたりの注意の計算とメモリは文脈長に依らない（O(1)）。
- softmax 注意（1M・10M）: 過去の k と v を溜め（KV キャッシュ）、今の q とだけ内積をとる。
- プロンプトの読み込みと窓の詰め直しは `Decoder.prefill` で、学習用の一括の順伝播（64 文字ずつ重みを使い回す速い経路）から状態を組み立てる。
- 位置埋め込みが 256 個しかないので、位置が 256 に達したら直近 128 文字で窓を詰め直す。256 文字以内なら旧方式と同じ文字列になる（100M の実モデルで 203 文字が一致）。
- テスト（`DecoderTest`、線形・softmax の両方）: 1 文字ずつの出力が系列全体の順伝播の最後の位置と 1e-4 で一致、並列版と 1 スレッド版の bit 一致、`prefill` が順伝播と bit 一致、位置と語彙の範囲外を拒否、256 文字以内で旧方式と同じ文字列。z の減衰を外す変異を入れるとテストが落ちることも確かめた。

100M の最終モデル、温度 0.7、topK 40、seed 3、1 スレッドでの実測。延長学習（8 workers）と別の CPU ジョブが同時に走っていて load 25 の状態なので、絶対値は空いているときより遅い。比は同じ条件どうし。

| 条件 | 1 文字あたり |
|---|---:|
| 旧方式、プロンプト 4 文字から 200 文字 | 1,549 ms |
| 旧方式、窓がいっぱい（250 文字のプロンプトから 10 文字） | 2,052 ms |
| **新方式、200 文字** | **259〜289 ms** |
| **新方式、600 文字（窓の詰め直し 3 回込み）** | **278 ms** |

新方式は生成が長くなっても 1 文字あたりの時間が変わらない。窓がいっぱいの状態どうしで約 7.4 倍。
新方式は 1 文字ごとに重み全体（410 MB）を読むので、メモリの読み出しで速さが決まる。4 スレッドに分けても、混んだ状態では速くならなかった（274 ms）。

### Kimi Linear などの線形注意との関係

このリポジトリの線形注意は、固定サイズの状態を再帰的に更新する系統で、推論の 1 文字あたりが O(1) になる点は Kimi Linear（KDA）などと同じ。中身はそれより素朴な形:

| 項目 | このリポジトリ | Kimi Linear（KDA） |
|---|---|---|
| 状態の更新 | 足し込み（s ← γs + φ(k)vᵀ） | デルタ則（同じキーの古い値を消してから書く） |
| 忘れ方 | ヘッドごとに固定の減衰 γ_h = 1 − 2^−(5+h)（RetNet と同じ） | 入力に応じたゲートを次元ごとに掛ける |
| 特徴写像と正規化 | φ = elu + 1、分母 φ(q)·z（Katharopoulos ら 2020） | q, k を正規化、分母なし |
| 層の構成 | 全 14 層が線形注意 | 線形注意 3 層にフル注意 1 層を挟むハイブリッド |
| 位置 | 学習した絶対位置埋め込み（256 個） | 線形注意側は位置埋め込みなし |

絶対位置埋め込みがあるので、状態を無限に持ち回すことはできず、256 文字で窓を詰め直している。位置埋め込みを外すにはモデルの作り直しと再学習が要る。

## SIMD カーネルの見直し（2026-09-26）

学習の 1 step の約 96% は密な層で、8 コアで約 225 GFLOPS 相当（理論ピーク約 1,150 GFLOPS の 2 割）だった。
`slm.BenchKernels`（100M と同じ幅で 2 層、1 スレッド、密な層の積和だけを数える）で、順伝播と逆伝播を分けて測った。
延長学習と別の CPU ジョブが同時に走っていて load 24〜25 の状態なので、新旧を交互に測って比べている。

| カーネル | 積和 1 回あたりの読み込み | 結果 |
|---|---:|---|
| 順伝播 dot4x2（重み 2 行 × トークン 4 本、現行） | 0.75 | 14〜16 GFLOPS |
| 順伝播 dot4x4（重み 4 行 × トークン 4 本） | 0.5 | **5〜6 GFLOPS（約 3 倍遅い）。見送り** |
| 逆伝播 axpy4（出力 1 本 × 入力 4 本、旧） | 1.25 | 中央値 11.0 GFLOPS（9 回） |
| **逆伝播 axpy4x2（出力 2 本 × 入力 4 本）** | 0.75 | **中央値 13.7 GFLOPS（9 回）、約 1.25 倍。採用** |

- axpy4x2 は重みの勾配（重み 2 行 × トークン 4 本）と入力の勾配（トークン 2 本 × 重み 4 行）の両方に使う。各要素への積和の順序は 1 本ずつのときと同じで、ゼロの勾配を読み飛ばす分岐も行ごと・トークンごとに残した。0 を足すと -0 が +0 に変わりうるため。
- 結果は旧カーネルと bit 一致する。`KernelGoldenTest` が、端数の分岐を全部通る模型（幅 48、FF 80、語彙 37、文脈 11）で logits と全勾配の指紋を旧カーネルの値と比べる。
- dot4x4 が遅い直接の原因は、JIT が 1 メソッドに使える節点の予算（NodeCountInliningCutoff）を超えたこと。合計の関数が 16 回中 9 回展開されず、512 ビットのベクトルを毎回オブジェクトに詰めていた（`-XX:+PrintInlining` で確認）。合計を 4 段の畳み込みにしても（2.7〜3.0 GFLOPS）、作業域に書き出して別メソッドで合計しても（6〜8 GFLOPS）遅いままだった。本体のループが 21 本のベクトルを同時に持つのでレジスタに収まっていないと推測しているが、機械語は確かめていない（hsdis が無い）。
- トークンのタイル幅は 64 のままが最速だった。16 と 32 は、x のタイルが L2 に収まる代わりに重みを読み直す回数が増え、逆伝播が 7〜13 GFLOPS に落ちた。

## C のカーネル（FFM API、2026-09-26）

Vector API では、積算 16 本を持つ形が JIT の節点予算を超えて遅くなった（上の節）。そこで密な層の順伝播と逆伝播のループ全体を C（AVX-512 の組み込み関数）で書き、
Java 標準の FFM API（`Linker.Option.critical(true)` で float[] をコピーせずに渡す）から呼ぶようにした。

- `native/slmkern.c` と `src/main/java/slm/NativeKernels.java`。ライブラリは `scripts/build-native.sh` が作る（sbt のテスト前と `run-100m.sh` でも自動）。AVX-512 か C コンパイラが無い環境、または `-Dslm.native=false` では Scala のカーネルで動く。
- 結果は Scala のカーネルと **bit 一致**する。各要素の積和の順序、16 レーンの合計の畳み方、端数処理、ゼロの勾配の読み飛ばしを同じにし、`-ffp-contract=off` で積和の自動融合を止めた。`NativeKernelTest` が端数だらけの模型（幅 39・FF 37・語彙 23）とタイル 2 枚の模型で logits と全勾配を突き合わせ、`KernelGoldenTest` も C 版で通る。融合を許した版を作るとテストが落ちることも確かめた。
- 順伝播は重み 4 行 × トークン 4 本（C コンパイラは積算 16 本をレジスタに収める）。逆伝播は BLIS と同じく詰め替えを使う: タイルの x を 64 要素の切れ端ごとに連続に並べ、4 行 × 64 要素の重みの勾配をレジスタに置いたまま全トークンを足し込む。入力の勾配は重み 32 行 × 64 要素の切れ端に詰め替える。詰め替えは写すだけなので足し込みの順序は変わらない。
- トークンのタイルは C 版だけ 128 にした。タイルが 4 の倍数なら端数のトークンも足し込みの順序も変わらないので、bit 一致のまま（タイル 128・256 のライブラリでもテストが通る）。

1 スレッドのカーネル単体（`BenchKernels`、マシンが空いている状態）:

| 経路 | Scala | C |
|---|---:|---:|
| 順伝播 | 60 GFLOPS | 75〜76 GFLOPS |
| 逆伝播 | 58〜59 GFLOPS | 61.5 GFLOPS |

100M 実形状・8 workers・batch 32 の 1 step（`BenchTrain`、学習を止めて交互に 2 回）:

| カーネル | tok/s |
|---|---:|
| Scala | 328〜338 |
| C、タイル 64 | 345〜374 |
| **C、タイル 128** | **486〜519** |
| C、タイル 256 | 443〜466 |

1 スレッドでは 60〜76 GFLOPS 出ているのに、8 workers では 1 コアあたり約 28 GFLOPS だった。損の大半は、各 worker がタイルごとに重み 410 MB を読み直し、共有の L3 とメモリを取り合うことにあった。タイルを 128 にして重みの読み直しを半分にしたのが一番効いた。256 では x の塊が L2 に収まらず遅くなる。
延長学習は step 4,330 から C のカーネル（タイル 128）で再開した。結果は bit 一致なので、数値は Scala のカーネルのまま続けた場合と変わらない。

## 次の本番: 位置埋め込みなし・最初から 1 エポック（別 PC で実行予定、2026-09-26）

コウタの判断で、延長学習（`runs/ja100m-e1`）は step 4,525（37.1M トークン）で止めた（`STOP` で境界保存、`scripts/resume-100m.sh run=runs/ja100m-e1` で再開できる）。
このマシン（WSL）の CPU を空けるため、次の本番は別の PC（Ryzen 9 5900X、Zen 3、12 コア / 24 スレッド、AVX2、L3 32 MB × 2）で回す。

変更点:

- **位置埋め込みなし（`positions=none`）**: 学習する位置埋め込みを持たず、位置の情報は線形注意の減衰だけが担う（Kimi Linear の線形注意層と同じ考え方）。重みが文脈長に依らなくなり、生成は窓を詰め直さずに状態を持ち回し続けられる（1 文字あたり O(1)）。パラメータは 196,608 個減って 102,410,790。
- **最初から 1 エポックの予算**: 8,410 updates の cosine を 1 本で組む（warm restart の跳ねがない）。
- **BF16 は見送り**: `vdpbf16ps` は AVX-512 BF16（Zen 4 以降）の命令で、5900X には無い。
- 5900X には AVX-512 が無いので、C のカーネルは作られず Scala のカーネル（8 レーン）で動く。8 レーンでも全テストが通ることを `-XX:MaxVectorSize=32` で確かめた（C のカーネルのテスト 3 件は飛ばされる）。`KernelGoldenTest` の指紋はレーン数ごとに持つ。
- テスト（`PositionsTest`、`GradientCheckTest`）: 位置埋め込みなしの勾配検査（Double 参照・Float32、softmax・線形注意）、パラメータ数、文脈長の違う模型で同じ出力、線形注意で文脈長の 3 倍まで 1 文字ずつの推論が長い文脈の順伝播と一致、設定と checkpoint の往復、項目の無い古い設定は位置埋め込みありとして読む。

別 PC での手順:

```bash
# JDK 25（Vector API は incubator）と sbt を入れておく
git clone git@github.com:kmizu/slm-ja.git && cd slm-ja
# コーパスはこのマシンの data/corpus.txt（SHA-256 8007ba16…）をコピーする（fetch.py で作り直すと青空文庫の索引の更新で中身が変わりうる）
sha256sum data/corpus.txt
sbt test
# 1) 形とメモリの確認
SLM_HEAP=12g scripts/run-100m.sh mode=preflight corpus=data/corpus.txt vocabFile=checkpoints/ja10m/vocab.txt \
  d=768 heads=12 layers=14 ff=3072 context=256 attention=linear positions=none batch=32 threads=12 out=runs/ja100m-nope-preflight
# 2) 速さ（worker 数を 12 と 8 で比べる）
java -Xmx12g -XX:+UseParallelGC --add-modules=jdk.incubator.vector -cp "$(cat target/classpath.txt)" slm.BenchTrain \
  vocabFile=checkpoints/ja10m/vocab.txt attention=linear positions=none threads=12 batch=32 steps=5
# 3) 本番（pilot として stopAfterSteps=250 で止め、quick64 を初回 run の step 250（4.014）と比べてから resume で続ける）
SLM_HEAP=12g scripts/run-100m.sh corpus=data/corpus.txt vocabFile=checkpoints/ja10m/vocab.txt \
  d=768 heads=12 layers=14 ff=3072 context=256 attention=linear positions=none batch=32 threads=12 steps=8410 stopAfterSteps=250 \
  lr=3e-4 warmup=256 wd=0.1 seed=0 evalEvery=250 saveEvery=250 saveSeconds=1800 sampleEvery=0 stopFile=runs/ja100m-nope/STOP out=runs/ja100m-nope
SLM_HEAP=12g scripts/resume-100m.sh run=runs/ja100m-nope threads=12 evalEvery=250 saveEvery=250 saveSeconds=1800 sampleEvery=0
```

step 250 までは学習率が予熱だけなので、初回 run（4,096 steps の cosine）と同じ学習率で比べられる。見込みは 5900X の実測次第で、1 エポックに 26〜48 時間。

## 採用した最適化と見送ったもの

| 項目 | 結果 |
|---|---|
| 8 workers（16 論理コアのうち物理コア数） | 採用。16 workers は SMT の L2 競合で 1/3 の速度 |
| トークンタイル（64）の密なカーネル | 採用。325 → 358 tok/s |
| 勾配集約と AdamW のパラメータ区間並列 | 採用。合わせて 1 step の 2% 未満 |
| 逆伝播の axpy4x2（出力 2 本で入力 4 本の読み込みを共有） | 採用。逆伝播が約 1.25 倍、結果は bit 一致 |
| 順伝播の dot4x4（重み 4 行 × トークン 4 本） | 見送り。JIT の節点予算を超えて約 3 倍遅い |
| トークンのタイル幅 16 / 32 | 見送り。64 より遅い |
| 密な層を C（AVX-512、FFM API）で、タイル 128 | 採用。100M・8 workers で約 1.5 倍（328〜338 → 486〜519 tok/s）、結果は bit 一致 |
| 順序固定の SIMD 縮約 | 採用（再開の bit 一致のため）。1M 設定で約 10% のコスト |
| softmax 注意のまま 100M | 見送り。16 workers・heap 14g で OOM、ユーザー指示で線形注意に変更 |
| 指示書 Task 6 の一部（`fetch.py` の文字数キャッシュ修正、token ids の単一コピー化） | 未実施。余分なコピーは Int 7,100 万個で約 0.28 GB。heap 12g に収まったため後回しにした |

## 残るボトルネックと延長の見積もり

- **forward/backward が 1 step の約 96%。** 8 コアで約 225 GFLOPS 相当まで来ていて、これ以上はカーネルの演算強度（重み 1 行をさらに多くのトークンで共有する）か、コア数そのものが要る。
- **生成は 1 文字ずつの推論に切り替えた（下の「生成の高速化」）。** 1 文字あたりの計算は文脈長に依らなくなったが、1 文字ごとに重み 410 MB を読むので、メモリの読み出しで速さが決まる。
- **他の CPU ジョブとの同居に弱い。** 8 workers が L2 を占有する前提なので、同じマシンで重いジョブが走ると半速になる。
- **延長。** 最終 lr（floor）でも損失はわずかに下がり続けていたので、学習領域（68.9M トークン）の 1 エポックまで warm restart で延長している（下の「延長」）。1 エポックを超えるなら、同じ文字語彙で日本語 Wikipedia などのコーパスを足す。

## 再現コマンド

```bash
sbt test
SLM_HEAP=12g scripts/run-100m.sh mode=preflight corpus=data/corpus.txt vocabFile=checkpoints/ja10m/vocab.txt \
  d=768 heads=12 layers=14 ff=3072 context=256 attention=linear batch=32 threads=8 out=runs/ja100m-preflight
SLM_HEAP=12g scripts/run-100m.sh corpus=data/corpus.txt vocabFile=checkpoints/ja10m/vocab.txt \
  d=768 heads=12 layers=14 ff=3072 context=256 attention=linear batch=32 threads=8 steps=4096 stopAfterSteps=100 \
  lr=3e-4 warmup=256 wd=0.1 seed=0 evalEvery=50 saveEvery=25 saveSeconds=600 sampleEvery=0 stopFile=runs/ja100m/STOP out=runs/ja100m
SLM_HEAP=12g scripts/resume-100m.sh run=runs/ja100m threads=8 evalEvery=250 saveEvery=250 saveSeconds=600
```

停止は `touch runs/ja100m/STOP`（step 境界で一貫した状態を保存して `interrupted` で終了）。
