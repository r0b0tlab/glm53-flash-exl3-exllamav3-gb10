# Results — GLM-5.3-Flash EXL3 on one GB10 (ExLlamaV3 runtime)

All results on a single NVIDIA GB10 / DGX Spark (SM121, aarch64, CUDA 13),
image `ghcr.io/r0b0tlab/glm53-flash-exl3-exllamav3-gb10:b25c467`
(registry digest `sha256:e321670c…88c02`), FP16 KV, 262,144-token C1 profile,
DFlash2 EXL3 draft K=5.

## Long-context (token-exact)

| Gate | Result |
|---|---|
| Near-window single-key NIAH | **PASS at 259,993 actual prompt tokens** (window 262,144; 2,048-token reserve; clean answer; exact client/server token parity; BOS proven via `/v1/token/encode`) |
| Guard minimum during request | 6,945 MiB `MemAvailable` vs 6,144-MiB fail-closed floor |
| New NVRM/OOM journal events | 0 |

Evidence: `evidence/glm53-exl3-256k-20260927/`.

## Matched 8K throughput (AR → DFlash2 K=5, median of 5, C1)

| Class | AR tok/s | K=5 tok/s |
|---|---:|---:|
| structured | 22.91 | 52.90 |
| code | 22.61 | 49.96 |
| prose | 22.54 | 20.56 (regressed) |

K5 counters: 4,832 accepted / 3,673 rejected draft tokens.

## Q200v2 kit (2026-09-27, serial, effort=low)

| Lane | Score |
|---|---|
| GSM8K | 79/80 (98.75%) |
| HumanEval | 39/40 (97.5%) — Docker-sandbox graded, 0 grader errors |
| IFEval | 31/40 (77.5%) |
| hard_reasoning | 19/20 (95.0%, independent manual review) |
| **text-180 total** | **168/180 (93.33%)** |
| BFCL v4 multi_turn_base hard-20 | 6/20 (30.0%) |
| **Full 200** | **174/200 (87.0%)** |

Zero transport errors; all 180 text rows `finish_reason=stop`; BFCL audit:
116 "Failed to decode" lines ≈ the normal text-only end-of-turn path
(cases × turns, kit-documented); all 14 BFCL failures carry explicit official
error types (instance-state mismatch / missing exec results) — genuine model
failures, not harness faults.

Evidence: `evidence/glm53-exl3-q200v2-20260927/`.

## Cross-engine calibration (do not transfer)

| | ExLlamaV3 (this repo, 256K) | vLLM fork (32K, 2026-09-21) |
|---|---|---|
| Q200 text-180 | 168/180 | 170/180 |
| BFCL-hard20 | 6/20 | 10/20 |
| Context verified | 259,993 tok single-key (token-exact) | 16,939 actual max (estimated-depth lane) |

## Not qualified

Multi-key near-window output formatting, concurrency > 1, 256K throughput,
long-run soak. Prose regresses under K=5 at 8K. 8K throughput is not
extrapolable to 256K.
