#!/usr/bin/env bash
# 100M run の起動: 引数はそのまま slm.Train へ渡す（eval で再解釈しない）。
#   SLM_HEAP=10g scripts/run-100m.sh corpus=... vocabFile=... d=768 ... out=runs/ja100m
# classpath は sbt から取得し、GC ログと起動コマンドを run ディレクトリに残す。二重起動は Train 側の LOCK で拒否する。
set -euo pipefail
cd "$(dirname "$0")/.."
HEAP="${SLM_HEAP:-10g}"
OUT=""
for a in "$@"; do case "$a" in out=*) OUT="${a#out=}";; esac; done
[ -n "$OUT" ] || { echo "out=... が要る" >&2; exit 2; }
mkdir -p "$OUT"
if [ ! -f target/classpath.txt ] || [ -n "${SLM_REBUILD:-}" ]; then
  sbt -batch compile "export Runtime/fullClasspath" 2>/dev/null | grep -E '^/|:/' | tail -1 > target/classpath.txt
fi
CP="$(cat target/classpath.txt)"
scripts/build-native.sh   # 密な層の C カーネル（無い環境では Scala のカーネルで動く）
[ -n "$CP" ] || { echo "classpath を取得できない" >&2; exit 2; }
java -version 2>&1 | head -1
STAMP="$(date +%Y%m%d-%H%M%S)"
CMD=(java "-Xmx$HEAP" -XX:+UseParallelGC "-Xlog:gc:file=$OUT/gc-$STAMP.log" --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED -cp "$CP" slm.Train "$@")
printf '%q ' "${CMD[@]}" > "$OUT/command-$STAMP.txt"; echo >> "$OUT/command-$STAMP.txt"
echo "launching: ${CMD[*]}"
nohup "${CMD[@]}" >> "$OUT/stdout-$STAMP.log" 2>&1 &
PID=$!
echo "$PID" > "$OUT/PID"
echo "pid=$PID log=$OUT/stdout-$STAMP.log"
