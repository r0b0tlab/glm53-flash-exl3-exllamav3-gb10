# Local qualification evidence — GB10 / EXL3 / ExLlamaV3

Scope: single GB10, native ARM64 SM121 ExLlamaV3 v1.5.2, pinned TabbyAPI, GLM-5.3-Flash EXL3 target and DFlash2 draft. All measurements below are local diagnostic epochs from 2026-09-27, before the standalone tracked 256K owner was admitted. Each row refers to an existing receipt, not an extrapolation from KV arithmetic. A configured 256K context alone is not a retrieval pass.

## Matched 8K throughput, concurrency 1

Source: `evidence/bench-ar-20260927T155617Z-1261860/bench.json` and `evidence/bench-spec-20260927T161055Z-1266143/bench.json`; both `bench.rc=0`. Five repetitions per structured/code/prose class. Median generated tok/s:

| Class | Autoregressive | DFlash2 K=5 | Difference |
| --- | ---: | ---: | ---: |
| Structured | 22.91 | 52.90 | +130.9% |
| Code | 22.61 | 49.96 | +121.0% |
| Prose | 22.54 | 20.56 | -8.8% |

The separate single-sample C1 row was 22.33 vs 50.03 tok/s. The K5 run recorded 4,832 accepted and 3,673 rejected draft tokens. Prose regressed; do not call K5 globally faster. These 8K measurements do not qualify performance at 256K.

## Guarded 256K retrieval

Local template IDs lacked the BOS token TabbyAPI inserted. After independently proving server token IDs against `/v1/token/encode`, each successful row below matched local BOS-corrected IDs to API `usage.prompt_tokens` exactly and returned the requested SKU(s) with `finish_reason=stop`. An output reserve of 2,048 tokens was used. `R1024`/`R512` name the host recurrent-cache caps in MiB; other effective parameters remained 262,144 max/cache, FP16 KV, batch size 1, and K5.

| Case | Profile | Prompt tokens | Result | Minimum guard MemAvailable during request |
| --- | --- | ---: | --- | ---: |
| Single key | R1024 | 31,733 | correct, rc 0 | not isolated per request |
| Single key | R1024 | 65,013 | correct, rc 0 | not isolated per request |
| Single key | R1024 | 130,040 | correct, rc 0 | 6,315 MiB |
| Single key, fresh epoch | R1024 | 234,080 | correct, rc 0 | 6,278 MiB |
| Single key, fresh epoch | R1024 | 259,988 | correct, rc 0 | 6,212 MiB |
| Two keys at 33/66, fresh epoch | R1024 | — | guard stopped at 6,119 MiB; no retrieval verdict | 6,119 MiB |
| Two keys at 33/66, fresh epoch | R512 | 259,994 | both correct SKUs, rc 0; output-format leak after them | 6,381 MiB |

The successful R512 two-key row consumed 393 output tokens, completed in 414 s, and had 102 tokens left after `259,994 + 2,048` in the configured window. Both requested SKUs appeared first, but the returned content continued with stray `<|assistant|>` text; retrieval succeeded, strict final-answer cleanliness did not. This is not permission to count that output as a clean quality pass. Its memory margin above the immutable 6,144-MiB guard floor was only 237 MiB. The R1024 attempt was killed by the independent guard, not a new NVRM error. Do not run extra host jobs on this narrow-memory profile or claim C>1/production uptime.

R1024 receipts: `evidence/serve-256k-20260927T172118Z-1304398/`, `evidence/serve-256k-20260927T174731Z-1350826/`, `evidence/serve-256k-20260927T180014Z-1360398/`, and failed two-key `evidence/serve-256k-20260927T181349Z-1370131/`. R512 two-key receipt: `evidence/serve-256k-20260927T182647Z-1378092/niah-multi260-bos.json` with `.rc=0` and `guard.jsonl`; the container was live and healthy immediately after the case. All model answers shown here are synthetic inventory SKUs, not application quality evidence.

## Load and memory failure boundary

The first 256K load exited before readiness. Live-loading shard cache filled tens of GiB on GB10 unified memory. A separate cache-steward trial reached 46/50 modules before an NVIDIA allocation failure. After targeted page-cache retirement left model-shard residency near zero and `MemFree` near 113 GiB, the guarded retry reached READY and passed text/tool/vision smokes. The steward used read-only `O_NOFOLLOW` descriptors and `posix_fadvise(DONTNEED)` only on the 31+1 pinned shard files, not a host-wide cache drop. The tracked startup steward (`scripts/cache_steward.py`) must be independently exercised with its owned `scripts/serve_256k.sh` before its integrated launch is called qualified.

Historical vLLM EXL3 context results are separate and do not validate this standalone runtime or imply TP=2. The KV-only estimate at 262,144 tokens was 4.30 GiB; model weights, recurrent state, staging, and file-cache/driver allocations dominate the actual safety boundary.
