#!/usr/bin/env bash
# 検証済みの最新世代から再開: SLM_HEAP=10g scripts/resume-100m.sh run=runs/ja100m [threads=N] [stopAfterSteps=N] [saveEvery=N] ...
# run 設定（形状・steps・lr・seed など）は checkpoint から復元する。STOP ファイルが残っていれば止まる理由を表示して終了する。
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=""
EXTRA=()
for a in "$@"; do case "$a" in run=*) RUN="${a#run=}";; *) EXTRA+=("$a");; esac; done
[ -n "$RUN" ] || { echo "run=... が要る" >&2; exit 2; }
[ -d "$RUN/state" ] || { echo "$RUN/state が無い（旧形式の重みは initFrom= で）" >&2; exit 2; }
if [ -f "$RUN/STOP" ]; then
  echo "$RUN/STOP が残っている。再開してもすぐ停止する。停止目的でなければ手で消してから再実行:" >&2
  echo "  rm $RUN/STOP" >&2
  exit 3
fi
CORPUS="$(grep -oE 'corpus=[^ ]+' "$RUN/run.txt" | head -1 | cut -d= -f2)"
exec "$(dirname "$0")/run-100m.sh" "resume=$RUN" "corpus=${CORPUS:-data/corpus.txt}" "out=$RUN" "stopFile=$RUN/STOP" "${EXTRA[@]}"
