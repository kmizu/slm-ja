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

## 採用した最適化と見送ったもの

| 項目 | 結果 |
|---|---|
| 8 workers（16 論理コアのうち物理コア数） | 採用。16 workers は SMT の L2 競合で 1/3 の速度 |
| トークンタイル（64）の密なカーネル | 採用。325 → 358 tok/s |
| 勾配集約と AdamW のパラメータ区間並列 | 採用。合わせて 1 step の 2% 未満 |
| 順序固定の SIMD 縮約 | 採用（再開の bit 一致のため）。1M 設定で約 10% のコスト |
| softmax 注意のまま 100M | 見送り。16 workers・heap 14g で OOM、ユーザー指示で線形注意に変更 |
| 指示書 Task 6 の一部（`fetch.py` の文字数キャッシュ修正、token ids の単一コピー化） | 未実施。余分なコピーは Int 7,100 万個で約 0.28 GB。heap 12g に収まったため後回しにした |

## 残るボトルネックと延長の見積もり

- **forward/backward が 1 step の約 96%。** 8 コアで約 225 GFLOPS 相当まで来ていて、これ以上はカーネルの演算強度（重み 1 行をさらに多くのトークンで共有する）か、コア数そのものが要る。
- **生成が遅い。** `Generate.sample` は 1 文字ごとに直近 256 文字ぶんの順伝播を 1 スレッドでやり直している。線形注意はヘッドごとの状態（64×64 と 64）を持ち回せば 1 文字あたりの計算量が文脈長に依らなくなるので、次に直す候補。
- **他の CPU ジョブとの同居に弱い。** 8 workers が L2 を占有する前提なので、同じマシンで重いジョブが走ると半速になる。
- **延長の見積もり。** 最終 lr（floor）でも損失はわずかに下がり続けていた。学習領域（68.9M トークン）を 1 エポックまで使い切るには追加 35.3M トークンで、343 tok/s なら約 29 時間。新しい cosine を warm restart で組み直すのが素直。1 エポックを超えるなら、同じ文字語彙で日本語 Wikipedia などのコーパスを足す。

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
