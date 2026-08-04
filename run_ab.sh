#!/usr/bin/env bash
# Compaction A/B: ayni kod, tek fark PIG_COMPACTION.
#
# Az oyun / cok pass tercih edildi: iki kol arasindaki farki gorunur kilan sey
# oyun cesitliligi degil, ayni oyunda tekrarlanan kosulardaki varyansin
# ortalanmasi. Duck'in kendi sayilari bunun ne kadar gurultulu oldugunu
# gosteriyor (ar25 pass skorlari: 0.0 ... 8.33).
set -uo pipefail
cd "$(dirname "$0")/ARC3-Inference"

export PYTHONPATH="$PWD:$PWD/../tufa-arc-agi-framework/src"
export CEREBRAS_API_KEY="$(grep CEREBRAS "/Users/gokturkakman/Desktop/Exposure AI/Brenchmark/.env" | cut -d= -f2)"
export OPENAI_BASE_URL=https://api.cerebras.ai/v1
export OPENAI_PROVIDER=cerebras
export OPERATION_MODE=offline
export MPLBACKEND=agg
export CONFIG_PATH=configs/inference.cerebras.json

GAMES="${GAMES:-ft09,vc33}"
PASSES="${PASSES:-3}"
MINUTES="${MINUTES:-10}"
ENVDIR="/Users/gokturkakman/Desktop/Exposure AI/Brenchmark/environment_files"
OUT="${OUT:-/tmp/pig-ab}"
mkdir -p "$OUT"

run_arm () {
  local name="$1" flag="$2"
  echo "=== KOL: $name (PIG_COMPACTION=$flag) ==="
  PIG_COMPACTION="$flag" .venv/bin/python -m inference.framework.run \
    --game "$GAMES" --model gpt-oss-120b \
    --n-passes "$PASSES" --max-runtime-minutes "$MINUTES" \
    --concurrent-jobs 3 \
    --experiment-dir "$OUT/$name" \
    --environments-dir "$ENVDIR" > "$OUT/$name.log" 2>&1
  echo "--- $name ozet ---"
  sed -n '/mean score/,/^$/p' "$OUT/$name.log" | head -8
  grep -E "^  [a-z0-9]+-" "$OUT/$name.log" | tail -5
}

run_arm "off" 0
run_arm "on"  1

echo
echo "================ A/B SONUC ================"
for arm in off on; do
  s=$(grep "mean score" "$OUT/$arm.log" | head -1 | awk '{print $3}')
  a=$(grep "total actions" "$OUT/$arm.log" | head -1 | awk '{print $3}')
  t=$(grep "total tokens" "$OUT/$arm.log" | head -1 | awk '{print $3}')
  printf "%-4s mean_score=%-8s actions=%-7s tokens=%s\n" "$arm" "$s" "$a" "$t"
done
