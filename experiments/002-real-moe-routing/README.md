# Experiment 002 — Real MoE Routing and Cache Effectiveness

One experiment within [Student AI Platform](../../README.md).

**Question:** Do actual student-oriented workloads reuse experts enough for caching to reduce end-to-end latency on the available Windows PC?

**Complete:** primary comparison, real routing captures, the separately requested 4–32 GiB cache-budget search, repeated fresh-prompt validation, answer checks and graphs. The tuning phase preserves the primary comparison.

GPT-OSS-20B MXFP4 is the approved experimental model. GPT-OSS-120B remains a separate architectural target; this smaller model's results will not establish 120B feasibility or beyond-RAM SSD-streaming performance.

## Results and evidence

- [Primary report](results/REPORT.md): measured inference, routing, quality, failures and interpretation.
- [Protocol](PROTOCOL.md): controls, definitions, scope and limitations.
- [Primary summary JSON](results/summary.json) and [433 evidence checks](results/verification.json).
- [Sweet-spot phase](results/sweet-spot/README.md): wider curves, repeat checks, fresh validation and graphs.

The final measured tradeoff is **5 GiB expert cache, four CPU threads, 90% of misses computed on the CPU, mmap loading**. It reached **43.14 decode tokens/s** and **3.530 seconds** mean complete 128-token request on fresh validation prompts. The fastest decode setting was 5.25 GiB at 44.27 tokens/s, only 2.54% ahead. Larger settings showed substantial falloff, with no confirmed competitive late recovery. Read the [sweet-spot report](results/sweet-spot/REPORT.md) and [full 4–32 GiB graph](results/sweet-spot/graphs/wide-range.png); requested budgets above about 9.52 GiB hit this checkpoint's full-expert cache cap.

Those validation rates use different prompts from the primary table below. Comparisons within the final validation showed 17.2% higher decode throughput than the original hybrid profile and 54.6% higher than static offload. The selected profile passed six of six normal-reasoning checks.

| Primary configuration | Decode tokens/s | Mean complete 128-token request |
|---|---:|---:|
| CPU experts / GPU attention, no expert cache | 22.60 | 6.630 s |
| Ordinary static GPU offload, 12 layers | 26.66 | 5.448 s |
| 4 GiB expert cache, misses uploaded | 23.53 | 6.169 s |
| 4 GiB cache, 95% of misses on CPU | **33.54** | **4.475 s** |

These are actual local inference measurements, using upstream llama.cpp rather than a new APEX engine. Four process repetitions per configuration used separate held-out prompts and rotated orders. Caching alone was slower than static offload; the combined cache/CPU path improved decode throughput by 25.8%.

The performance tests explicitly selected the final-answer channel. In that mode, all configurations missed the same algebra question and passed five of six distinct checks. A separate normal-reasoning diagnostic passed all six on CPU, static and hybrid execution. These small checks do not certify broad quality or numerical equivalence.

Real routing captures touched 756 of 768 layer/expert pairs. Replays compare Experiment 001's cache policies against the same events; simulated hit rates are distinguished from measured speed. Trace-on/off variation did not establish a stable instrumentation overhead.

## Environment and scope

- RTX 5060, Intel i5-14400F, 32 GiB RAM, Windows 11, E: NVMe model storage.
- GPT-OSS-20B MXFP4 GGUF: 12,109,566,624 bytes, verified SHA-256 `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901`.
- Model revision `41e8a94d15d62070b36cf4553a0250717ecf46b8`.
- llama.cpp `b12359`, commit `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`, Windows x64 CUDA 13.5.
- Python 3.14.0 and dependencies in `requirements-measured.txt`.
- No paid API/cloud inference, public service, system-memory changes or execution of generated code.

The model fits host RAM. These results do not test GPT-OSS-120B, SSD streaming beyond RAM, long context or concurrent students.

## Layout

`scripts/` contains download, inference, trace/replay, evidence-check and graph tools. `fixtures/` contains frozen tasks. `results/setup/` records pins, hashes, environment and incidents. `results/runs/` retains commands, outputs, timings, memory samples and compact engine logs. `results/harness-snapshots/` preserves executed source. `results/sweet-spot/` is the separate tuning phase.

Downloaded weights, runtime binaries, environments, bulk logs and uncompressed traces are ignored by Git. Compressed raw and normalized routing traces are retained with hashes and coverage.

## Reproduce

Run from the repository root, using its existing `.venv`. Experiment 001's cache definitions are imported by repository path. On a new machine, create a Python environment and install `requirements-measured.txt` first. No CUDA toolkit installation is needed for the pinned runtime.

Use the saved plan to download the same checkpoint and runtime (about 12.7 GB network transfer plus extraction space):

```powershell
.\.venv\Scripts\python.exe experiments/002-real-moe-routing/scripts/bootstrap.py download
.\.venv\Scripts\python.exe experiments/002-real-moe-routing/scripts/bootstrap.py sources
```

The `metadata` action refreshes pins from upstream; do not use it to reproduce the archived version. Existing downloads must match saved hashes.

Run a single configuration with a **new run label**:

```powershell
.\.venv\Scripts\python.exe experiments/002-real-moe-routing/scripts/benchmark.py my-hybrid-repeat --suite evaluation --load-mode mmap -- -ngl 99 -cmoe --moe-cache-mib 4096 --moe-cache-cpu-share 0.95
```

The hidden server binds only `127.0.0.1` and stops after the run. A restricted execution sandbox may need permission for localhost access. Run labels refuse to overwrite existing evidence.

The primary sequence used `calibration-matrix`, `calibration-cache-followup`, `calibration-mmap --load-mode mmap`, and `calibration-cache4096 --load-mode mmap`, followed by:

```powershell
.\.venv\Scripts\python.exe experiments/002-real-moe-routing/scripts/benchmark.py evaluation-matrix --suite evaluation --load-mode mmap
.\.venv\Scripts\python.exe experiments/002-real-moe-routing/scripts/benchmark.py reasoning-check --suite evaluation --load-mode mmap
.\.venv\Scripts\python.exe experiments/002-real-moe-routing/scripts/benchmark.py tracing-matrix --suite routing --load-mode mmap
```

Use a separate reproduction checkout or new matrix labels to retain the archived results. Some pinned-host calibration candidates failed; those failures are part of the evidence.

Analyze and validate without rerunning inference:

```powershell
.\.venv\Scripts\python.exe experiments/002-real-moe-routing/scripts/test_routing.py
.\.venv\Scripts\python.exe experiments/002-real-moe-routing/scripts/summarize.py
.\.venv\Scripts\python.exe experiments/002-real-moe-routing/scripts/verify.py
```

`routing.py trace-r1-on` parses and replays one raw trace. On another machine restore `routing.trace` from the retained `.trace.gz`, or inspect normalized `.jsonl.gz` directly. The byte-exact primary harness snapshot and its provenance note are linked in the report.

Treat measured inference, replay estimates, quality checks and failed trials as different evidence. APEX remains a research hypothesis: additional complexity must earn its cost against ordinary offload.
