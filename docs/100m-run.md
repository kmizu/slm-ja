# 100M run の記録

指示書 `slm-ja-100m-claude-code-instructions.md` に沿って、約 100M パラメータのモデルを再開可能な状態で学習予算まで回した記録。
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

（進行中。完了後に final1024 / best / 生成例 / 所要時間を追記）

### 共通の評価条件での比較（`slm.Evaluate`、同じコーパス・語彙・split・窓集合）

| モデル | 消費トークン | quick64 (even64-v1) | final1024 (even1024-v1) |
|---|---:|---:|---:|
| 10M `checkpoints/ja10m`（softmax 注意、旧 run） | 33.5M | 2.880 | 2.872 |
| 100M step 750（線形注意、途中） | 6.1M | 3.433 | 3.431 |

10M の旧ログの 2.70 は「末尾 3% の先頭 64 窓」で、上の値とは窓集合が違う。100M の quick64 は学習ログの eval と一致（3.4334）。
100M の途中経過の生成（温度 0.7、seed 3）は `runs/ja100m/export-step-00000750` から出せる。step 750 では文の形はあるが 10M より粗い。

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
