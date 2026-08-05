#!/usr/bin/env bash
# RunPod pod bootstrap: install deps, start vLLM, run parallel ARC3 benchmark.
set -euo pipefail

LOG=/workspace/bootstrap.log
exec > >(tee -a "$LOG") 2>&1

echo "=== runpod bootstrap $(date -Is) ==="

export PATH="/root/.local/bin:${PATH:-}"
export HF_HOME=/workspace/cache/huggingface
export SERVER_STATE_ROOT=/workspace/cache/arc3_runtime
export SERVER_CACHE_ROOT="${SERVER_STATE_ROOT}/cache"
mkdir -p "$HF_HOME" "$SERVER_STATE_ROOT" /workspace/runs

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git curl ca-certificates build-essential

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="/root/.local/bin:$PATH"
fi

SRC_TARBALL="${SRC_TARBALL:-}"
if [[ -n "$SRC_TARBALL" ]]; then
  echo "Downloading workspace tarball..."
  curl -fsSL "$SRC_TARBALL" -o /workspace/duck-harness.tgz
  tar xzf /workspace/duck-harness.tgz -C /workspace
else
  REPO_URL="${REPO_URL:-https://github.com/ToprakG/pig-harness.git}"
  echo "Cloning from ${REPO_URL} (no SRC_TARBALL set)..."
  cd /workspace
  git clone --depth 1 "$REPO_URL" duck-harness
fi

if [[ -d /workspace/ARC3-Inference ]]; then
  REPO=/workspace
elif [[ -d /workspace/duck-harness/ARC3-Inference ]]; then
  REPO=/workspace/duck-harness
else
  echo "Could not find ARC3-Inference after source setup"
  ls -la /workspace
  exit 1
fi

cd "$REPO/ARC3-Inference"
make install
make install-local-sources

echo "Starting vLLM server..."
nohup make server >> /workspace/vllm_server.log 2>&1 &

echo "Waiting for vLLM (up to 30 min)..."
ready=0
for i in $(seq 1 180); do
  if make check-server 2>/dev/null; then
    ready=1
    echo "vLLM ready after ${i}0s"
    break
  fi
  sleep 10
done
if [[ "$ready" != 1 ]]; then
  echo "vLLM failed to start; tail server log:"
  tail -n 80 /workspace/vllm_server.log || true
  exit 1
fi

GAMES="${GAMES:-tn36-ef4dde99,lf52-271a04aa,cn04-2fe56bfb,bp35-0a0ad940,wa30-ee6fef47,lp85-305b61c3,r11l-495a7899,tu93-0768757b,sp80-589a99af,m0r0-492f87ba}"
RUN_NAME="${RUN_NAME:-runpod-10game-45min}"
CONCURRENT_JOBS="${CONCURRENT_JOBS:-10}"
MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-45}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-/workspace/runs/runpod}"

echo "Starting benchmark: games=$GAMES concurrent=$CONCURRENT_JOBS runtime=${MAX_RUNTIME_MINUTES}m"

make interactive \
  CONFIG_PATH=configs/inference.json \
  GAME="$GAMES" \
  GAME_TAGS= \
  EXCLUDE_GAME_TAGS= \
  N_PASSES=1 \
  CONCURRENT_JOBS="$CONCURRENT_JOBS" \
  MAX_RUNTIME_MINUTES="$MAX_RUNTIME_MINUTES" \
  RUN_NAME="$RUN_NAME" \
  EXPERIMENTS_DIR="$EXPERIMENTS_DIR" \
  2>&1 | tee /workspace/benchmark.log

touch /workspace/benchmark_complete.flag
echo "=== benchmark finished $(date -Is) ==="
