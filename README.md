# GLM-5.3-Flash EXL3 on one GB10 — ExLlamaV3 runtime, guarded 256K serve

Native ARM64/SM121 [ExLlamaV3](https://github.com/turboderp-org/exllamav3) v1.5.2 +
[TabbyAPI](https://github.com/theroyallab/tabbyAPI) runtime that serves the
GLM-5.3-Flash EXL3 pack with its DFlash2 EXL3 draft at a **262,144-token
context on a single NVIDIA GB10 / DGX Spark** — with a fail-closed load path
and an independently verified near-window retrieval result.

- Target pack (ours, MIT): [r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121](https://huggingface.co/r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121) — 98.5 GB, 31 shards
- DFlash2 draft (ours): [r0b0tlab/GLM-5.3-Flash-DFlash2-EXL3-3.00bpw](https://huggingface.co/r0b0tlab/GLM-5.3-Flash-DFlash2-EXL3-3.00bpw) — 610 MiB
- Draft upstream: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (CC BY-NC-ND 4.0, research/evaluation)

This is the standalone ExLlamaV3 runtime. It is independent of our earlier
vLLM EXL3 fork ([r0b0tlab/glm53-flash-exl3-dflash2-sm121](https://github.com/r0b0tlab/glm53-flash-exl3-dflash2-sm121)),
whose published DFlash2 K=5 profile is 32,768 tokens configured. Results do
not transfer between the two engines.

## Verified result — 256K near-window single-key retrieval

On a fresh serve epoch of this exact runtime (2026-09-27), a token-exact
needle-in-a-haystack client:

1. proves the server-inserted BOS token against `/v1/token/encode`
   (local chat-template IDs + 1 BOS == server IDs, token-for-token),
2. fits the haystack with the real local tokenizer (never a chars/token
   guess),
3. requires **exact equality** between local token count and the API's
   `usage.prompt_tokens`,
4. accepts only `finish_reason=stop` with the needle present.

Result: **259,993 actual prompt tokens** into a 262,144-token window
(2,048-token output reserve, 103 tokens to spare) — needle retrieved, clean
answer, exact token parity. Guard minimum during the request: 6,945 MiB
host `MemAvailable` against a 6,144-MiB fail-closed floor. No new NVRM
allocation failure in the privileged kernel journal.

Sanitized evidence: [`evidence/glm53-exl3-256k-20260927/`](evidence/glm53-exl3-256k-20260927/)
(`niah-single-near.public.json`, `guard-summary.public.json`, `bench-8k-matched.public.json`, `MANIFEST.sha256`).

**Not qualified on this profile:** multi-key near-window output formatting,
concurrency > 1, throughput at 256K, and long-run stability. Matched 8K
throughput (AR → DFlash2 K=5): structured 22.91 → 52.90 tok/s, code
22.61 → 49.96 tok/s, prose 22.54 → 20.56 tok/s — prose regressed; K=5 is not
a blanket speedup. Do not extrapolate 8K throughput to 256K.

## Why this is nontrivial on GB10

GB10 is a single SM121 GPU with 128 GB **unified** memory. A KV-cache estimate
alone (~4.30 GiB at 262,144 tokens with FP16 KV) says nothing about
loadability: the ~94 GB of model files, load transients, the file cache, and
native driver allocations share the same physical pool, and NVIDIA device
allocation can fail while `MemAvailable` still looks healthy.

This repo encodes the working recipe:

- `scripts/retire_page_cache.py` — prelaunch retirement of **only** the pinned
  31+1 read-only shard pages (`posix_fadvise(DONTNEED)` + `mincore` readback),
  with a physical-`MemFree` admission budget. Never `drop_caches`, never a
  global flush.
- `scripts/cache_steward.py` — bounded steward that keeps those shard pages
  retired **while weights load**, then stops at READY (proved by receipt).
- `scripts/guard.py` — independent host-memory/kernel-OOM guard: 6,144-MiB
  `MemAvailable` floor sampled every 2 s, fail-closed on new privileged-journal
  `NV_ERR_NO_MEMORY`/OOM, stops only the exact expected image+container ID,
  bounded 24-hour lease.
- `scripts/exact_niah.py` — the token-exact retrieval client described above
  (BOS proof, exact `usage.prompt_tokens` parity, explicit `single_near` /
  `multi_near` cases, `clean_completion` recorded separately from retrieval).
- `scripts/serve_256k.sh` — sole-owner launcher: exact source/image receipt
  admission, native GPU audit **before** the weight load, guard + steward in
  separate tmux sockets, authoritative `/v1/model` profile verification
  (262,144 max/cache, FP16, C1, vision), text/tool/vision smokes.

## Build and serve

Requires: aarch64 GB10 host, Docker with NVIDIA runtime, the model packs from
the HF links above, `tmux`, `ruamel.yaml` in a project `.venv`, and
noninteractive privileged kernel-journal reads (`sudo -n journalctl`).

```bash
# 1) offline suite (76 tests) + shell gates
.venv/bin/python -m unittest discover -s tests -q
bash -n scripts/build_image.sh scripts/serve_256k.sh
shellcheck -x scripts/build_image.sh scripts/serve_256k.sh

# 2) build the revision-labelled native image (durable tmux)
bash scripts/build_image.sh        # writes evidence/build-<ts>-<pid>/

# 3) guarded serve (single owner; model paths are required env)
TARGET=/path/to/glm-5.3-flash-exl3-2.25bpw-pack \
DRAFT=/path/to/glm53-flash-dflash2-exl3-3.00bpw \
  bash scripts/serve_256k.sh evidence/build-<ts>-<pid>
```

The serve is loopback-only (`127.0.0.1:5013`) and bounded to a 24-hour guard
lease by design. Served profile: FP16 KV, `max_seq_len = cache_size = 262144`,
batch size 1, DFlash2 K=5, 512-MiB host recurrent-cache cap
(`config/gb10-256k-k5.yml`).

A second profile, `config/config.yml`, is the original 8,192-token FP16-KV
K=5 comparison profile used for the matched AR-vs-K5 throughput table.

## Container image (click-run)

The qualification image is published — **ARM64/SM121 only** (aarch64 host
required; there is no x86-64 build):

```bash
docker pull ghcr.io/r0b0tlab/glm53-flash-exl3-exllamav3-gb10:b25c467
# :latest tracks the same build
```

- Registry digest: `sha256:e321670c0964d2613fb52aa33b84bd7783724eda315e1c6502480d9638e88c02`
- Image config ID: `sha256:1eccd2c0b7888a4a1f049ebc6953649ed1d931f4fbdb2aef0c8e98a3ca92f08b`
  (matches the local build byte-for-byte; the tag name is the source SHA)
- Built from this repository at commit `b25c467` by `scripts/build_image.sh`
  (revision label `org.opencontainers.image.revision=b25c4672f…`, non-root
  `runner` user). Contains **no model weights** — mount the HF packs from the
  links above. Model paths are bind-mounted read-only at
  `/models/target` and `/models/draft` by `scripts/serve.sh`.

If you prefer to build it yourself: `bash scripts/build_image.sh` from a clean
checkout of the same commit reproduces the image (native compile; needs
an aarch64 CUDA 13 host).

## Q200v2 quality kit (2026-09-27, same profile, serial, effort=low)

Full 200-row Q200v2 kit (text-180 + BFCL-hard20) on this exact runtime:

| Lane | Score |
|---|---|
| GSM8K | 79/80 (98.75%) |
| HumanEval | 39/40 (97.5%) — Docker-sandbox graded, 0 grader errors |
| IFEval | 31/40 (77.5%) |
| hard_reasoning | 19/20 (95.0%, independent manual review bound to content hashes) |
| **text-180** | **168/180 (93.33%)** |
| BFCL v4 multi_turn_base hard-20 | 6/20 (30.0%) — zero infra errors; all 14 failures are genuine model errors (state mismatches / missing exec results), not decode/harness failures |
| **Full 200** | **174/200 (87.0%)** |

All 180 text rows `finish_reason=stop`, zero transport errors, workers=1/threads=1
per concurrency discipline. Evidence:
[`evidence/glm53-exl3-q200v2-20260927/`](evidence/glm53-exl3-q200v2-20260927/).
For calibration, the vLLM EXL3 profile (different engine, 32K) scored 170/180
text and 10/20 BFCL on 2026-09-21 — text comparable, BFCL multi-turn weaker
here.

## Layout

- `docker/` — native ARM64 CUDA 13 image (ExLlamaV3 v1.5.2, TabbyAPI, torch 2.13.0+cu130)
- `scripts/` — build/serve/guard/steward/retrieval clients
- `config/` — pinned profiles
- `runtime.lock.json` — exact upstream pins
- `tests/` — 76 offline tests
- `evidence/` — sanitized public evidence bundles
- `docs/RESULTS.md` — full measured record with limits
- `docs/QUALIFICATION.md` — qualification narrative with limits

## License

- Our code: **MIT** (see `LICENSE`).
- TabbyAPI is **AGPL-3.0** (pinned at `816c321`, source manifest in
  `runtime.lock.json`); we use it unmodified inside the container. The AGPL
  source is available at its upstream repository; this repo's scripts build it
  from the pinned revision.
- ExLlamaV3: MIT.
- Model licenses flow through the HF links above (target MIT; upstream draft
  CC BY-NC-ND 4.0 — our draft quantization is published for
  research/evaluation on that basis).
