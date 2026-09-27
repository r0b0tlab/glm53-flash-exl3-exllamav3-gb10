#!/usr/bin/env bash
# Single-owner 256K K5 serve with guarded load and exact-image receipt.
# Usage: bash scripts/serve_256k.sh /absolute/evidence/build-<passing-attempt>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
BUILDDIR="${1:?exact successful build receipt required}"
PROFILE="$ROOT/config/gb10-256k-k5.yml"
TARGET="${TARGET:?set TARGET to the target EXL3 pack directory}"
DRAFT="${DRAFT:?set DRAFT to the DFlash2 EXL3 draft directory}"
export TARGET DRAFT
IMAGE=glm53-exl3-exllamav3:local
NAME=glm53-exl3
GUARD_LEASE_SECONDS=86400
cd "$ROOT"
case "$BUILDDIR" in "$ROOT"/evidence/build-*) ;; *) printf 'non-project build receipt\n' >&2; exit 2;; esac
ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)-$$"
OUT="$ROOT/evidence/serve-256k-$ATTEMPT"
mkdir -p "$OUT"
CONTAINER_ID=""
GUARD_PID=""
GUARD_OWNED=0
GUARD_LOG="$OUT/guard.jsonl"
STEWARD_PID=""
STEWARD_OWNED=0
STEWARD_LOG="$OUT/cache-steward.jsonl"
STEWARD_STOP="$OUT/cache-steward.stop.request"
START_EPOCH="$(date +%s)"
finish() {
  local rc=$? observed="" current_guard_pid="" current_steward_pid=""
  trap '' INT TERM HUP
  trap - EXIT
  if [[ $STEWARD_OWNED == 1 ]]; then
    printf '%s\n' 'owner cleanup' >"$STEWARD_STOP"
    current_steward_pid="$(tmux -L glm53-exl3-cache list-panes -t steward -F '#{pane_pid}' 2>/dev/null || true)"
    if [[ -n "$current_steward_pid" && "$current_steward_pid" == "$STEWARD_PID" ]]; then
      tmux -L glm53-exl3-cache kill-session -t steward 2>/dev/null || rc=4
    elif [[ -n "$current_steward_pid" ]]; then
      printf 'steward pane identity changed; no foreign cleanup\n' >"$OUT/steward-cleanup-error.txt"
      rc=4
    fi
  fi
  if [[ -n "$CONTAINER_ID" ]]; then
    observed="$(docker inspect "$NAME" --format '{{.Id}}' 2>/dev/null || true)"
    if [[ "$observed" == "$CONTAINER_ID" ]]; then
      docker logs "$CONTAINER_ID" >"$OUT/server.log" 2>&1 || true
      docker inspect "$CONTAINER_ID" >"$OUT/container-before-stop.json" || rc=4
      if [[ $(docker inspect "$CONTAINER_ID" --format '{{.State.Running}}' 2>/dev/null || true) == true ]]; then
        docker stop --timeout 3 "$CONTAINER_ID" >"$OUT/docker-stop.txt" 2>&1 || rc=4
      fi
      docker inspect "$CONTAINER_ID" >"$OUT/container-after-stop.json" || rc=4
      if [[ $(docker inspect "$CONTAINER_ID" --format '{{.State.Running}}' 2>/dev/null || true) != false ]]; then rc=4; fi
    else
      printf 'named container identity changed; no foreign cleanup\n' >"$OUT/cleanup-error.txt"
      rc=4
    fi
  fi
  if [[ $GUARD_OWNED == 1 ]]; then
    current_guard_pid="$(tmux -L glm53-exl3-guard list-panes -t guard -F '#{pane_pid}' 2>/dev/null || true)"
    if [[ -n "$GUARD_PID" && "$GUARD_PID" == "$current_guard_pid" ]]; then
      tmux -L glm53-exl3-guard kill-session -t guard 2>/dev/null || rc=4
    else
      printf 'owned guardian lost or changed identity\n' >"$OUT/guard-cleanup-error.txt"
      rc=4
    fi
  fi
  date -Is >"$OUT/end.timestamp"
  printf '%s\n' "$(( $(date +%s)-START_EPOCH ))" >"$OUT/duration-seconds.txt"
  printf '%s\n' "$rc" >"$OUT/serve.rc"
  if [[ $rc == 0 ]]; then date -Is >"$OUT/serve-pass.timestamp";
  else date -Is >"$OUT/serve-fail.timestamp"; fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# Exact source, image, profile, host, and ownership admission before a GPU process.
test -z "$(git status --porcelain=v1)"
test "$(git rev-parse HEAD)" = "$(<"$BUILDDIR/source-revision.txt")"
test "$(<"$BUILDDIR/build.rc")" = 0
test -f "$BUILDDIR/build-pass.timestamp"
test -z "$(docker ps --format '{{.Names}}')"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
if docker container inspect "$NAME" >/dev/null 2>&1; then printf 'owned name already exists\n' >&2; exit 2; fi
if tmux -L glm53-exl3-guard has-session -t guard 2>/dev/null; then printf 'guardian socket already owned\n' >&2; exit 2; fi
if tmux -L glm53-exl3-cache has-session -t steward 2>/dev/null; then printf 'steward socket already owned\n' >&2; exit 2; fi
test ! -e "$STEWARD_STOP"
.venv/bin/python - "$PROFILE" "$TARGET" <<'PY'
import json,sys
from pathlib import Path
from ruamel.yaml import YAML
p=YAML(typ='safe').load(Path(sys.argv[1]).read_text())
assert p['network']['host']=='0.0.0.0' and p['network']['port']==5000
assert p['network']['allowed_origins']==[] and p['network']['disable_fetch_requests']
assert p['model']['max_seq_len']==p['model']['cache_size']==262144
assert p['model']['max_batch_size']==1 and p['model']['cache_mode']=='FP16'
assert p['model']['vision'] and p['model']['tensor_parallel'] is False
assert p['draft_model']['draft_mode']=='model' and p['draft_model']['draft_num_tokens']==5
assert p['memory']['sysmem_recurrent_cache']==512
model=json.loads(Path(sys.argv[2]).joinpath('config.json').read_text())
assert model['text_config']['max_position_embeddings']>=262144
PY
python3 - "$BUILDDIR" "$IMAGE" "$OUT" <<'PY'
import json,pathlib,subprocess,sys
receipt,image,out=pathlib.Path(sys.argv[1]),sys.argv[2],pathlib.Path(sys.argv[3])
expected=json.loads((receipt/'image-inspect.json').read_text())['Id']
x=json.loads(subprocess.check_output(['docker','image','inspect',image],text=True))[0]
assert x['Id']==expected and x['Architecture']=='arm64' and x['Config']['User']=='runner'
assert x['Config']['Labels']['org.opencontainers.image.revision']==(receipt/'source-revision.txt').read_text().strip()
out.joinpath('image-id.txt').write_text(expected+'\n')
PY
sha256sum "$0" "$PROFILE" scripts/serve.sh scripts/cache_steward.py scripts/retire_page_cache.py scripts/guard.py scripts/guard_admission.py scripts/smoke.py scripts/exact_niah.py >"$OUT/input-hashes.txt"
date -Is >"$OUT/start.timestamp"
tmux -L glm53-exl3-guard new-session -d -s guard -c "$ROOT" "python3 scripts/guard.py --duration-seconds '$GUARD_LEASE_SECONDS' --image-id-file '$OUT/image-id.txt' --out '$GUARD_LOG'"
GUARD_OWNED=1
GUARD_PID="$(tmux -L glm53-exl3-guard list-panes -t guard -F '#{pane_pid}')"
test "$GUARD_PID" -gt 0
printf '%s\n' "$GUARD_PID" >"$OUT/guard-pid.txt"
DEADLINE=$(( $(date +%s)+30 ))
until python3 scripts/guard_admission.py "$GUARD_LOG" >"$OUT/guard-ready.txt" 2>"$OUT/guard-ready.err"; do
  if ! tmux -L glm53-exl3-guard has-session -t guard 2>/dev/null || [[ $(date +%s) -ge $DEADLINE ]]; then
    printf 'guardian failed admission\n' >&2; exit 3
  fi
  sleep 1
done
# Audit the native SM121 extension on the actual GPU BEFORE the giant weight
# load. A second CUDA context after 256K READY would consume scarce physical
# memory and could manufacture an otherwise avoidable NVRM failure.
docker run --rm --gpus all --memory=2g --network none --entrypoint python "$IMAGE" -c \
  "import torch,exllamav3_ext; assert torch.__version__=='2.13.0+cu130'; assert torch.cuda.get_device_capability(0)==(12,1); print('SM121_NATIVE_GPU_OK',exllamav3_ext.__file__)" \
  >"$OUT/native-gpu-audit.txt" 2>"$OUT/native-gpu-audit.err"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
python3 scripts/guard_admission.py "$GUARD_LOG" >"$OUT/guard-post-audit.txt"
tmux -L glm53-exl3-cache new-session -d -s steward -c "$ROOT" \
  "python3 scripts/cache_steward.py --target '$TARGET' --draft '$DRAFT' --out '$STEWARD_LOG' --stop-file '$STEWARD_STOP' --max-seconds 300 --interval 1 --cache-limit-mib 512"
STEWARD_OWNED=1
STEWARD_PID="$(tmux -L glm53-exl3-cache list-panes -t steward -F '#{pane_pid}')"
test "$STEWARD_PID" -gt 0
printf '%s\n' "$STEWARD_PID" >"$OUT/steward-pid.txt"
DEADLINE=$(( $(date +%s)+30 ))
until python3 - "$STEWARD_LOG" <<'PY' >"$OUT/steward-ready.txt" 2>"$OUT/steward-ready.err"
import json,sys
from pathlib import Path
rows=[json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
assert rows[0]['event']=='armed' and rows[0]['files']==32
assert not any(r.get('event')=='failed' for r in rows)
samples=[r for r in rows if r.get('event')=='sample']
assert samples and samples[-1]['cached_after_mib']<=512
print('STEWARD_READY',samples[-1]['cached_after_mib'])
PY
do
  if [[ $(tmux -L glm53-exl3-cache list-panes -t steward -F '#{pane_pid}' 2>/dev/null || true) != "$STEWARD_PID" ||
        $(date +%s) -ge $DEADLINE ]]; then
    printf 'pinned model cache steward failed admission\n' >&2; exit 5
  fi
  sleep 1
done
set +e
GUARD_LOG="$GUARD_LOG" PROFILE="$PROFILE" CACHE_RETIRE_RECEIPT="$OUT/cache-retire.json" \
  bash scripts/serve.sh >"$OUT/container-id.txt" 2>"$OUT/launch.err"
LAUNCH_RC=$?
set -e
CONTAINER_ID="$(<"$OUT/container-id.txt")"
if [[ $LAUNCH_RC -ne 0 || ! "$CONTAINER_ID" =~ ^[0-9a-f]{64}$ ]]; then
  candidate="$(docker inspect "$NAME" --format '{{.Id}}' 2>/dev/null || true)"
  if [[ "$candidate" =~ ^[0-9a-f]{64}$ &&
        $(docker inspect "$candidate" --format '{{.Image}}') == "$(<"$OUT/image-id.txt")" ]]; then
    CONTAINER_ID="$candidate"
    printf 'recovered exact owned ID after uncertain launch: %s\n' "$candidate" >"$OUT/receipt-recovery.txt"
  fi
  printf 'launcher rc=%s or malformed ID\n' "$LAUNCH_RC" >&2; exit 4
fi
if [[ $(docker inspect "$NAME" --format '{{.Id}}') != "$CONTAINER_ID" ||
      $(docker inspect "$CONTAINER_ID" --format '{{.Image}}') != "$(<"$OUT/image-id.txt")" ]]; then
  printf 'live name/image identity mismatch\n' >&2; exit 4
fi
python3 - "$OUT/cache-retire.json" <<'PY'
import json,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text())
assert x['status']=='PASS' and x['budget']['remaining_cached_mib']<=64
assert x['memfree_after_mib']>=x['budget']['required_memfree_mib']
PY
DEADLINE=$(( $(date +%s)+1800 ))
while true; do
  if ! tmux -L glm53-exl3-guard has-session -t guard 2>/dev/null ||
     ! python3 scripts/guard_admission.py "$GUARD_LOG" >"$OUT/guard-current.txt" 2>"$OUT/guard-current.err"; then
    printf 'guardian died or unsafe during load\n' >&2; exit 5
  fi
  if [[ $(tmux -L glm53-exl3-cache list-panes -t steward -F '#{pane_pid}' 2>/dev/null || true) != "$STEWARD_PID" ]] ||
     ! python3 - "$STEWARD_LOG" <<'PY' >/dev/null 2>&1
import json,sys
from pathlib import Path
rows=[json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
assert rows and not any(r.get('event') in ('failed','stopped') for r in rows)
assert any(r.get('event')=='sample' for r in rows)
PY
  then
    printf 'cache steward failed during model load\n' >&2; exit 5
  fi
  if [[ $(docker inspect "$CONTAINER_ID" --format '{{.State.Running}}' 2>/dev/null || true) != true ]]; then
    printf 'container exited before READY\n' >&2; exit 6
  fi
  if python3 - "$OUT/models.json" <<'PY' >/dev/null 2>&1
import json,pathlib,sys,urllib.request
with urllib.request.urlopen('http://127.0.0.1:5013/v1/models',timeout=3) as r: x=json.load(r)
assert 'target' in [m['id'] for m in x['data']]
pathlib.Path(sys.argv[1]).write_text(json.dumps(x,indent=2)+'\n')
PY
  then break; fi
  if [[ $(date +%s) -ge $DEADLINE ]]; then printf '256K readiness timeout\n' >&2; exit 7; fi
  sleep 3
done
python3 - "$OUT/model-card.json" <<'PY'
import json,sys,urllib.request
from pathlib import Path
# /v1/models is a list projection (id + meta only); /v1/model carries
# the loaded model's effective cache/context parameters.
with urllib.request.urlopen('http://127.0.0.1:5013/v1/model',timeout=5) as r:
    card=json.load(r)
Path(sys.argv[1]).write_text(json.dumps(card,indent=2)+'\n')
if card.get('id')!='target':
    raise ValueError(f'wrong loaded model: {card.get("id")}')
p=card.get('parameters') or {}
if (p.get('max_seq_len')!=262144 or p.get('cache_size')!=262144 or
        p.get('max_batch_size')!=1 or p.get('cache_mode')!='FP16' or
        p.get('use_vision') is not True):
    raise ValueError(f'live served profile not 256K/C1: {p}')
print('LIVE_256K_PROFILE_OK',p['max_seq_len'],p['cache_size'],p['max_batch_size'])
PY
printf '%s\n' 'TARGET_256K_READY' >"$OUT/ready.txt"
printf '%s\n' 'model READY; stop loading-only cache steward' >"$STEWARD_STOP"
DEADLINE=$(( $(date +%s)+20 ))
while tmux -L glm53-exl3-cache has-session -t steward 2>/dev/null; do
  if [[ $(tmux -L glm53-exl3-cache list-panes -t steward -F '#{pane_pid}' 2>/dev/null || true) != "$STEWARD_PID" ||
        $(date +%s) -ge $DEADLINE ]]; then
    printf 'cache steward did not exit cleanly after READY\n' >&2; exit 5
  fi
  sleep 1
done
python3 - "$STEWARD_LOG" <<'PY' >"$OUT/steward-stop-proof.txt"
import json,sys
from pathlib import Path
rows=[json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
assert rows[-1].get('event')=='stopped' and rows[-1].get('reason')=='stop.request'
assert not any(r.get('event')=='failed' for r in rows)
samples=[r for r in rows if r.get('event')=='sample']
assert any(r.get('container_running') is True for r in samples)
assert samples[-1]['cached_after_mib']<=512
print('STEWARD_STOP_OK',len(samples),samples[-1]['cached_after_mib'])
PY
STEWARD_OWNED=0
docker exec "$CONTAINER_ID" python /opt/smoke.py >"$OUT/smoke.txt" 2>"$OUT/smoke.err"
python3 scripts/guard_admission.py "$GUARD_LOG" >"$OUT/guard-post-smoke.txt"
python3 - "$GUARD_LOG" <<'PY'
import json,sys
from pathlib import Path
from scripts.guard import new_oom
p=Path(sys.argv[1]); start=json.loads(p.read_text().splitlines()[0])['since']
assert not new_oom(start),'new NVIDIA/kernel OOM since guardian start'
PY
date -Is >"$OUT/short-smoke-pass.timestamp"
printf '%s\n' "SERVER_256K_LIVE id=$CONTAINER_ID port=127.0.0.1:5013 guard_pid=$GUARD_PID" >"$OUT/live.txt"
# Remain the sole owner while guarded local clients use the server.
# An exact stop.request under this attempt yields clean cleanup. The guardian
# expires at 24h; the owner closes before then rather than promising permanence.
while true; do
  if [[ -f "$OUT/stop.request" ]]; then break; fi
  if [[ $(date +%s) -ge $((START_EPOCH+GUARD_LEASE_SECONDS-100)) ]]; then
    printf 'guardian 24h lease nearly expired; ending this bounded epoch\n' >"$OUT/lease-expired.txt"
    exit 3
  fi
  if ! tmux -L glm53-exl3-guard has-session -t guard 2>/dev/null ||
     ! python3 scripts/guard_admission.py "$GUARD_LOG" >"$OUT/guard-live.txt" 2>"$OUT/guard-live.err"; then
    printf 'guardian died or unsafe after smoke\n' >&2; exit 5
  fi
  if [[ $(docker inspect "$CONTAINER_ID" --format '{{.State.Running}}' 2>/dev/null || true) != true ]]; then
    printf 'server died after 256K READY\n' >&2; exit 6
  fi
  sleep 5
done
