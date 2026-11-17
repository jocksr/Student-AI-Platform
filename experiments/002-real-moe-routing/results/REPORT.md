# Experiment 002 — primary comparison results

Measured on 2026-11-17. This report preserves the first completed series. The subsequent sweet-spot search requested by the user is a separate phase and must not be merged into this held-out comparison.

## Main result

GPT-OSS-20B MXFP4 runs locally on the RTX 5060 / i5-14400F / 32 GiB Windows system. In this bounded workload, a 4 GiB expert cache combined with CPU execution of 95% of cache misses improved decode throughput by **25.8% over ordinary static offload** and **48.4% over the CPU-expert baseline**. Caching without CPU handling of misses did not beat static offload.

The executable runtime is upstream llama.cpp, not a new APEX engine. These results support further investigation of selective residency and CPU/GPU cooperation; they do not establish GPT-OSS-120B feasibility.

## Held-out performance

Four configurations × four process repetitions × four 128-token performance prompts. Configuration order and prompt order rotate across rounds. All configurations use six CPU threads, a 2,048-token context limit, batch size 256, memory-mapped loading, no prompt reuse, the same checkpoint, and direct-final generation. Full controls and timing definitions: [protocol](../PROTOCOL.md).

| Configuration | Aggregate decode tokens/s | Per-round range | Mean 128-token request | Mean first-token latency | Maximum sampled whole-GPU use |
|---|---:|---:|---:|---:|---:|
| CPU experts, GPU attention; cache disabled | 22.60 | 22.04–22.99 | 6.630 s | 1.011 s | 3,590 MiB |
| Ordinary static offload, 12 layers | 26.66 | 26.24–27.64 | 5.448 s | 0.684 s | 7,125 MiB |
| 4 GiB expert cache, all misses uploaded | 23.53 | 22.98–24.24 | 6.169 s | 0.773 s | 7,509 MiB |
| 4 GiB cache, 95% of misses on CPU | **33.54** | **31.38–35.05** | **4.475 s** | **0.689 s** | **7,243 MiB** |

The hybrid configuration reduced mean request latency by **17.9% versus static offload** and **32.5% versus the CPU-expert baseline**. First-token latency was effectively similar to static placement in this sample; the principal improvement was generation speed.

Decode throughput is total timed decode steps divided by total decode time. The engine times `predicted_n - 1` steps because the first output token comes from prefill. Request latency includes prefill and local HTTP overhead but excludes startup, warm-up and template formatting. All performance requests generated exactly 128 tokens and reused zero prompt tokens.

Process startup averaged 2.12 s for CPU experts, 3.32 s for static offload, 2.62 s for cache-only and 2.52 s for hybrid. The filesystem cache was not flushed; these are not cold-disk loading figures. Sampled process RSS was roughly 11.4 GiB, including mapped file pages. Minimum system available RAM during the primary comparison was 4.11 GiB. These figures do not establish absence of paging or measure unique private allocations.

## Quality and execution-path differences

The primary test explicitly forced the assistant's final channel to control generation length. Each configuration passed **20/24 checks**: the same five of six distinct questions, repeated four times. Every configuration answered `1` to `3x + 7 = 22`, where the expected answer is `5`. The error is retained, not scored away.

A separate post-evaluation diagnostic used the normal chat endpoint with low reasoning effort and a 256-token cap:

| Configuration | Correct distinct checks | Mean complete answer latency |
|---|---:|---:|
| CPU experts | 6/6 | 3.635 s |
| Static 12-layer offload | 6/6 | 2.731 s |
| Hybrid 4 GiB cache | 6/6 | 2.438 s |

Normal reasoning corrected the algebra answer in all three configurations. This diagnostic supports using the intended reasoning path for student-facing answers; six checks are far too few to certify general quality. The fixed-length throughput table is not a measurement of full reasoning-enabled conversations.

Generated performance texts differed between execution paths, and the hybrid path sometimes differed across repeated runs despite fixed sampling settings. All six short answer-check outputs were identical across the primary configurations. There is no claim of bitwise equivalence or broad quality equivalence. The throughput comparison holds prompts and output counts fixed, not the exact generated token stream.

## Actual expert routing

Three trace captures each contain **48,768 expert accesses in evaluated decode**, across four prompts and 508 decode steps. Each capture touched **756 of 768 layer/expert pairs**. Routing is real GPT-OSS-20B execution, not Experiment 001's synthetic distributions.

The eight most frequent experts in each layer accounted for 46.5% of accesses on average. This describes the observed distribution after the fact; it is not a prediction policy. Nearly all experts were used, so the trace does not justify permanently discarding supposedly unneeded experts.

Experiment 001's policies were replayed against the same recorded routes. Warm-up, the observed startup probe and prefill tail update cache state; the table measures evaluated decode only.

| Simulated GPU expert budget | Global LRU hit rate | Global LFU hit rate | Cost-aware hit rate |
|---|---:|---:|---:|
| 512 MiB | 0.0% | 11.0% | 11.0% |
| 1,024 MiB | 0.0% | 20.7% | 20.7% |
| 2,048 MiB | 49.8% | 38.3% | 38.3% |
| 3,072 MiB | 66.8% | 53.5% | 53.5% |
| 4,096 MiB | 79.0% | 66.4% | 66.4% |

Small global LRU caches cannot retain a full cycle of selected experts across layers and thrash in this trace. At larger budgets, recency beats accumulated frequency on the changing prompts. Cost-aware equals LFU here because expert sizes and the modeled reload cost per byte are equal.

**These are replay hit rates, not live engine counters or measured speedups.** Upstream's engine uses a different decaying-frequency cache, reserves additional slots, and can compute misses on the CPU. The replay does not model those scheduling costs. The real cache-only result demonstrates why an apparent byte-saving opportunity is insufficient to establish faster inference.

The trace includes all measured decode steps, the four-token prefill tail in layers 0–22, one prefill row in layer 23, and a separately labeled startup probe. Bulk prefill is absent. A post-measurement flush workload protects measured records from Windows process-termination buffering; incomplete tail records are explicitly excluded. Raw and normalized traces are retained as gzip files with SHA-256 hashes and per-request coverage.

## Tracing overhead and uncertainty

Three paired trace-on/off runs changed measured decode time by **+2.47%, -11.73%, and -4.30%**. The sign changes and large spread indicate that these runs do **not** isolate a stable instrumentation overhead. No negative-overhead or zero-overhead claim is made. The main performance table uses tracing disabled.

The machine remained a working Windows desktop. GPU usage includes other applications, resources were sampled rather than continuously measured, and no application shutdown or system setting changes were made. Four rounds give useful local evidence, not a formal confidence interval or an exhaustive optimization result.

## Failed and negative findings

- A 1 GiB cache with pinned host loading reached only 11.46 tokens/s on calibration.
- A 2 GiB cache failed during cuBLAS setup; its retry and the 3 GiB pinned-host variants failed CUDA allocation. The failures and memory samples are retained. The experiment does not isolate the root cause.
- Memory-mapped loading allowed 3 GiB and 4 GiB caches to run. Every held-out configuration then used that same mode.
- Cache-only execution remained slower than ordinary static offload, despite a large cache.
- Direct-final generation missed a simple algebra question; normal reasoning corrected it in the small follow-up.
- Routing instrumentation overhead remains unresolved at the precision of these measurements.

## What this changes about APEX

The useful direction is **measured hybrid execution**, with cache size, CPU share and memory-loading behavior tested together. Larger caches helped in calibration, but the data do not imply unlimited improvement as cache size increases. More complex caching must earn its cost against ordinary offload.

The next authorized phase searches for the smallest reliable configuration near the best observed performance, rather than selecting the largest allocation automatically. Its calibration and validation evidence will remain separate from this series.

Nothing here tests a model larger than host RAM, sustained SSD expert streaming, a 120B checkpoint, long-context conversations or concurrent users. No 120B throughput is extrapolated from the 20B measurements.

## Evidence and reproduction

- [Machine-readable summary](summary.json), [frozen evaluation plan](evaluation-plan.json), [quality follow-up plan](reasoning-check-plan.json), [trace plan](tracing-plan.json).
- `runs/*/run.json`: commands, prompts, outputs, token IDs, timings and failures.
- `runs/*/resource-samples.json`: sampled RAM/GPU evidence.
- `runs/trace-*-on/routing-analysis.json`: routing coverage and policy replay.
- [433 saved-evidence checks](verification.json), plus parser/corruption and answer-check unit tests.
- [Executed evaluation source identity](evaluation-source.json): a byte-exact snapshot of the Python source loaded by the evaluation process. Later on-disk edits added separate diagnostics; this note disambiguates the per-run file hash from the already-loaded evaluation source.
- [Setup incidents](setup/incidents.json), [download verification](setup/downloads.json), [environment and binary hashes](setup/environment.json).

Primary implementation references: [pinned llama.cpp cache documentation](https://github.com/ggml-org/llama.cpp/blob/b1b770d90d94cad9e9ea5b993bdbc383a522fb28/docs/moe-cache.md), [pinned trace producer](https://github.com/ggml-org/llama.cpp/blob/b1b770d90d94cad9e9ea5b993bdbc383a522fb28/src/llama-moe-cache.cpp), and [pinned checkpoint repository](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/tree/41e8a94d15d62070b36cf4553a0250717ecf46b8).
