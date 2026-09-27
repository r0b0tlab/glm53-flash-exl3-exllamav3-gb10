#!/usr/bin/env bash
# Launch only the checked, owned single-GB10 instance; never replace another container.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${TARGET:?set TARGET to the target EXL3 pack directory}"
DRAFT="${DRAFT:?set DRAFT to the DFlash2 EXL3 draft directory}"
PROFILE="${PROFILE:-$ROOT/config/config.yml}"
IMAGE="${IMAGE:-glm53-exl3-exllamav3:local}"
GUARD_LOG="${GUARD_LOG:-$ROOT/evidence/guard.jsonl}"
GUARD_TMUX_SOCKET="${GUARD_TMUX_SOCKET:-glm53-exl3-guard}"
if [[ -z ${CACHE_RETIRE_RECEIPT:-} ]]; then
  CACHE_RETIRE_RECEIPT="$ROOT/evidence/cache-retire-$(date -u +%Y%m%dT%H%M%SZ)-${BASHPID}.json"
fi
NAME=glm53-exl3

if [[ $(uname -m) != aarch64 ]]; then
  printf 'aarch64 host required\n' >&2; exit 1
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ne 1 ]]; then
  printf 'exactly one GPU required for --gpus all\n' >&2; exit 1
fi
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; then
  printf 'GPU has an active compute process; refusing to disturb it\n' >&2; exit 1
fi
# Separate tmux socket makes guard ownership inspectable. It must have a fresh
# sample, not merely a stale "armed" line. Keep checking it after launch too.
if ! tmux -L "$GUARD_TMUX_SOCKET" has-session -t '=guard' 2>/dev/null; then
  printf 'independent host guard is not running\n' >&2; exit 1
fi
python3 "$ROOT/scripts/guard_admission.py" "$GUARD_LOG" >&2
if [[ ! -f "$PROFILE" || -L "$PROFILE" ]]; then
  printf 'profile file missing or symlinked\n' >&2; exit 1
fi
python3 "$ROOT/scripts/check_artifacts.py" --target "$TARGET" --draft "$DRAFT" >&2
if [[ -n $(ss -H -ltn '( sport = :5013 )') ]]; then
  printf 'port 5013 occupied\n' >&2; exit 1
fi
if docker container inspect "$NAME" >/dev/null 2>&1; then
  printf 'container name glm53-exl3 already exists; inspect it before cleanup\n' >&2; exit 1
fi
if [[ $(docker image inspect "$IMAGE" --format '{{.Architecture}}') != arm64 ||
      $(docker image inspect "$IMAGE" --format '{{ index .Config.Labels "org.opencontainers.image.exllamav3-revision" }}') != 12414d0af7b3beeabdda5990f6b554b996fa1416 ]]; then
  printf 'image architecture/source identity mismatch\n' >&2; exit 1
fi
# Retire ONLY the already-validated read-only model shard cache. On GB10,
# MemAvailable includes clean file pages the GPU allocator may not reclaim;
# admission is based on mincore readback and physical MemFree, not a global
# drop_caches or a weakened guardian.
python3 "$ROOT/scripts/retire_page_cache.py" --target "$TARGET" --draft "$DRAFT" \
  --out "$CACHE_RETIRE_RECEIPT" >&2
# Final guard check closes the time spent validating/retiring model pages.
python3 "$ROOT/scripts/guard_admission.py" "$GUARD_LOG" >&2
exec docker run -d --name "$NAME" --gpus all \
  --cpus=14 --memory=112g --memory-swap=112g --shm-size=8g \
  --security-opt=no-new-privileges --cap-drop=ALL \
  -p 127.0.0.1:5013:5000 \
  -v "$TARGET:/models/target:ro" -v "$DRAFT:/models/draft:ro" \
  -v "$PROFILE:/opt/tabbyAPI/config.yml:ro" -v glm53-exl3-cache:/data \
  "$IMAGE"
