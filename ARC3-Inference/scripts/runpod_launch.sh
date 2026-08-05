#!/usr/bin/env bash
# Create a RunPod GPU pod and auto-run the 10-game / 45min parallel benchmark.
# Usage:
#   export RUNPOD_API_KEY='rpa_...'
#   ./scripts/runpod_launch.sh
#
# Optional overrides:
#   SRC_TARBALL_URL   - wget tarball of local workspace (includes uncommitted changes)
#   GPU_TYPE_IDS      - comma-separated RunPod gpu type ids
#   CLOUD_TYPE        - SECURE or COMMUNITY (default SECURE)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOOTSTRAP="$ROOT/scripts/runpod_bootstrap.sh"

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "Set RUNPOD_API_KEY first."
  exit 1
fi

GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA A100-SXM4-80GB}"
CLOUD_TYPE="${CLOUD_TYPE:-SECURE}"
BOOTSTRAP_B64="$(base64 < "$BOOTSTRAP" | tr -d '\n')"

python3 - "$BOOTSTRAP_B64" "$GPU_TYPE_IDS" "$CLOUD_TYPE" <<'PY' > /tmp/runpod_launch_payload.json
import json, os, sys
bootstrap_b64, gpu_csv, cloud = sys.argv[1], sys.argv[2], sys.argv[3]
start_cmd = (
    f"echo {bootstrap_b64} | base64 -d > /workspace/bootstrap.sh && "
    "chmod +x /workspace/bootstrap.sh && bash /workspace/bootstrap.sh"
)
if os.environ.get("SRC_TARBALL_URL"):
    start_cmd = (
        f"export SRC_TARBALL='{os.environ['SRC_TARBALL_URL']}' && "
        + start_cmd
    )
payload = {
    "name": os.environ.get("POD_NAME", "arc3-10game-45min-parallel"),
    "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    "gpuTypeIds": [g.strip() for g in gpu_csv.split(",") if g.strip()],
    "gpuCount": int(os.environ.get("GPU_COUNT", "1")),
    "cloudType": cloud,
    "containerDiskInGb": int(os.environ.get("CONTAINER_DISK_GB", "120")),
    "volumeInGb": int(os.environ.get("VOLUME_GB", "200")),
    "volumeMountPath": "/workspace",
    "ports": ["8888/http", "22/tcp"],
    "dockerStartCmd": ["bash", "-lc", start_cmd],
    "env": {
        "CONCURRENT_JOBS": os.environ.get("CONCURRENT_JOBS", "10"),
        "MAX_RUNTIME_MINUTES": os.environ.get("MAX_RUNTIME_MINUTES", "45"),
        "RUN_NAME": os.environ.get("RUN_NAME", "runpod-10game-45min"),
        "EXPERIMENTS_DIR": os.environ.get("EXPERIMENTS_DIR", "/workspace/runs/runpod"),
        "GAMES": os.environ.get(
            "GAMES",
            "tn36-ef4dde99,lf52-271a04aa,cn04-2fe56bfb,bp35-0a0ad940,wa30-ee6fef47,"
            "lp85-305b61c3,r11l-495a7899,tu93-0768757b,sp80-589a99af,m0r0-492f87ba",
        ),
    },
}
print(json.dumps(payload))
PY

echo "Creating pod (cloud=$CLOUD_TYPE)..."
response="$(curl -sS -w "\n%{http_code}" --request POST \
  --url https://rest.runpod.io/v1/pods \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data @/tmp/runpod_launch_payload.json)"
body="${response%$'\n'*}"
code="${response##*$'\n'}"
echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
echo "HTTP $code"
if [[ "$code" != "201" && "$code" != "200" ]]; then
  exit 1
fi
