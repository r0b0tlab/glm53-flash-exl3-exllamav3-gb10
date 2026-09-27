#!/usr/bin/env bash
# Canonical, fail-closed native ARM64 SM121 build owner with auditable receipt.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
IMAGE=glm53-exl3-exllamav3:local
BASE='nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6'
cd "$ROOT"
ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)-$$"
OUT="$ROOT/evidence/build-$ATTEMPT"
mkdir -p "$OUT"
START_EPOCH="$(date +%s)"
FINALIZED=0
finalize() {
  local rc=$?
  trap - EXIT
  if [[ $FINALIZED == 0 ]]; then
    printf '%s\n' "$rc" >"$OUT/build.rc"
    date -Is >"$OUT/build-end.timestamp"
    printf '%s\n' "$(( $(date +%s) - START_EPOCH ))" >"$OUT/duration-seconds.txt"
    if [[ $rc == 0 ]]; then
      date -Is >"$OUT/build-pass.timestamp"
    else
      date -Is >"$OUT/build-fail.timestamp"
    fi
  fi
  exit "$rc"
}
trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Refuse a second owner before either can target the shared image tag.
exec 9>"$ROOT/evidence/.build.lock"
flock -n 9 || { printf 'a native build owner already holds the lock\n' >&2; exit 75; }

# Admit only clean tracked source and the exact pinned local base. Never prune caches.
test -z "$(git status --porcelain=v1)"
REV="$(git rev-parse HEAD)"
TREE="$(git rev-parse 'HEAD^{tree}')"
test "$(git branch --show-current)" = main
test "$(docker image inspect "$BASE" --format '{{.Architecture}}')" = arm64
test "$(uname -m)" = aarch64
test -z "$(docker ps -q)"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test "$(df -BG --output=avail "$ROOT" | tr -dc '0-9')" -ge 300
test "$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)" -ge 80000
sudo -n journalctl -b -k --no-pager -n 1 | tee "$OUT/kernel-preflight.txt" >/dev/null

printf '%s\n' "$REV" >"$OUT/source-revision.txt"
printf '%s\n' "$TREE" >"$OUT/source-tree.txt"
date -Is >"$OUT/build-start.timestamp"
sha256sum "$0" docker/Dockerfile runtime.lock.json config/config.yml config/gb10-256k-k5.yml scripts/smoke.py scripts/repair_cusparselt_wheel.py scripts/serve.sh scripts/serve_256k.sh scripts/cache_steward.py scripts/retire_page_cache.py scripts/guard.py scripts/guard_admission.py scripts/exact_niah.py >"$OUT/input-hashes.txt"
git archive --format=tar "$REV" | sha256sum >"$OUT/archive-1.sha256"
git archive --format=tar "$REV" | sha256sum >"$OUT/archive-2.sha256"
cmp "$OUT/archive-1.sha256" "$OUT/archive-2.sha256"
git archive --format=tar "$REV" | tar -tf - | wc -l >"$OUT/archive-entries.txt"
docker image inspect "$BASE" --format '{{.Id}} {{.Architecture}} {{.Size}}' >"$OUT/base-image.txt"
printf '%s\n' "docker buildx build --load --platform linux/arm64 --progress=plain --build-arg PROJECT_REVISION=$REV -f docker/Dockerfile -t $IMAGE ." >"$OUT/build-command.txt"

set +e
docker buildx build --load --platform linux/arm64 --progress=plain \
  --build-arg "PROJECT_REVISION=$REV" \
  -f docker/Dockerfile -t "$IMAGE" . 2>&1 | tee "$OUT/build.log"
BUILD_RC=${PIPESTATUS[0]}
set -e
if [[ $BUILD_RC -ne 0 ]]; then
  exit "$BUILD_RC"
fi
# These checks run after compile, as a separate audit. No GPU/model load yet.
python3 - "$IMAGE" "$REV" "$OUT" <<'PY'
import json,subprocess,sys,pathlib
image,expected,where=sys.argv[1],sys.argv[2],pathlib.Path(sys.argv[3])
x=json.loads(subprocess.check_output(['docker','image','inspect',image],text=True))[0]
where.joinpath('image-inspect.json').write_text(json.dumps(x,indent=2)+'\n')
assert x['Architecture']=='arm64',x['Architecture']
assert x['Config']['User']=='runner',x['Config']['User']
assert x['Config']['Labels']['org.opencontainers.image.revision']==expected
print('IMAGE_AUDIT_OK',x['Id'],x['Architecture'],x['Config']['User'])
PY
docker run --rm --runtime runc -e NVIDIA_VISIBLE_DEVICES=void --entrypoint python "$IMAGE" -c \
  "import torch,exllamav3_ext; from exllamav3.version import __version__; assert torch.__version__=='2.13.0+cu130'; assert __version__=='1.5.2'; assert exllamav3_ext.__file__.endswith('.so'); print('DEFAULT_USER_EXTENSION_OK',torch.__version__,exllamav3_ext.__file__)" \
  >"$OUT/default-user-audit.txt"
# EXIT finalizer is the sole writer of rc and PASS/FAIL.
