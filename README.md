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

## Layout

- `docker/` — native ARM64 CUDA 13 image (ExLlamaV3 v1.5.2, TabbyAPI, torch 2.13.0+cu130)
- `scripts/` — build/serve/guard/steward/retrieval clients
- `config/` — pinned profiles
- `runtime.lock.json` — exact upstream pins
- `tests/` — 76 offline tests
- `evidence/` — sanitized public evidence bundles
- `docs/QUALIFICATION.md` — full measured record with limits

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
