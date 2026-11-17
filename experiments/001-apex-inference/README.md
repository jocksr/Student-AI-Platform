# Experiment 001 — APEX Inference Research

Advanced Predictive Expert eXecution: one experiment within [Student AI Platform](../../README.md), investigating memory-constrained Mixture-of-Experts inference.

**APEX is a research experiment, not a predetermined platform solution.** Its purpose is to reject weak ideas, measure physical limits, and identify improvements worth integrating into existing engines. GPT-OSS-120B remains this experiment's primary architectural target. The current deliverable is a working measurement and simulation foundation; it does not run that model or claim an inference speedup.

Start with [the experimental report](results/initial-report.md), [research findings](RESEARCH.md), [architecture](ARCHITECTURE.md), and [evidence-based roadmap](ROADMAP.md).

## Working capabilities

- Read-only hardware discovery, machine-readable inventory and software provenance.
- Conservative RAM-copy and storage profiling, including separate Windows buffered and `FILE_FLAG_NO_BUFFERING` measurements.
- Real CUDA Driver API transfer, allocation and reference FP32 projection measurements without a CUDA toolkit or CUDA-enabled PyTorch installation.
- Validated JSONL expert-routing format; deterministic uniform, hot-set and changing-hot-set workloads; multiple request identities and optional timing/transfer metadata.
- Byte-bounded RAM and GPU caches with no-cache, LRU, LFU and frequency/cost-per-byte policies. Configurable model dimensions, expert sizes and budgets.
- Independent lossless tiles: raw, GZIP, LZ4 and Zstandard; reversible byte shuffling, bitplanes and XOR prediction; checksums, offset indexes and independent reads.
- Compression comparisons at 16, 64 and 256 KiB, including actual small GPT-OSS samples, synthetic controls, decoder timing and partial allocation measurements.
- Cross-expert exact-block and XOR-residual tests, counting the base and dependency index.
- Joint simulations using measured transfer rates, compressed sizes and decoding times, with every extrapolation identified.
- Automated correctness tests and reproducible JSON/Markdown reports.

There is no trained predictor, asynchronous prefetch runtime, model server, agent sandbox, public service, real routing capture, or fused MXFP4 inference kernel yet.

## Environment on this machine

Experiment root: `E:\Project_APEX\experiments\001-apex-inference`. The repository root is `E:\Project_APEX`. Windows 11 Home, i5-14400F (10 cores / 16 logical processors), 2 x 16 GiB TeamGroup DIMMs configured at 3200 MT/s, RTX 5060 with 8,151 MiB reported by nvidia-smi, and Samsung 990 PRO 2 TB backing E:.

CPU AVX2/FMA are available; NumPy reports no AVX-512. The installed NVIDIA driver is 621.24 and the CUDA Driver API reports compute capability 12.0. CUDA toolkit/nvcc are absent from the checked installation location and PATH. MSVC is installed under Visual Studio 18 Community, although `cl.exe` is not on the shell PATH. WSL2 Ubuntu and docker-desktop are installed and were stopped at discovery; Docker CLI is installed, daemon operation was not tested.

The existing Python is 3.14.0. The current checkout reuses the repository-local `.venv` (`--system-site-packages`) to avoid downloading large dependencies. Only the small LZ4 wheel and the editable local APEX package were installed. New setups may use a private `.venv` in this experiment. Existing CPU-only PyTorch is not required by APEX. Package versions and executable paths are recorded in each result. This reuse is convenient, but not a hermetic environment.

The experiment was moved into this folder on November 17, 2026. Historical JSON measurements retain their original file paths, timestamps, source hashes and input hashes. They are not rewritten to imply that the measurements were collected after the move. `results/tests-layout-check.json` records the post-move verification separately.

## Run locally

From PowerShell, enter this experiment's directory and select its local environment or the existing shared environment:

```powershell
Set-Location E:\Project_APEX\experiments\001-apex-inference
$python = if (Test-Path .venv\Scripts\python.exe) {
    (Resolve-Path .venv\Scripts\python.exe).Path
} else {
    (Resolve-Path ..\..\.venv\Scripts\python.exe).Path
}
& $python scripts/run_tests.py
& $python -m apex hardware
& $python -m apex benchmark-memory
& $python -m apex benchmark-storage
& $python -m apex benchmark-gpu
& $python -m apex generate-trace --steps 64 --workload phase-shift --seed 17
& $python -m apex simulate --trace results/trace.jsonl --policy lru
& $python -m apex benchmark-compression --samples benchmarks/weight-samples
& $python -m apex cross-expert
& $python -m apex compare
& $python -m apex report
```

`simulate` uses explicitly labeled default bandwidth assumptions unless you supply `--storage-gbs` and `--h2d-gbs`. `compare` consumes the measured `storage.json` and `gpu.json`, official metadata, and compression measurements. It replays the same traces for every policy. Do not read simulated movement time as measured token latency.

`./scripts/run_initial.ps1` runs tests and the complete local experiment pipeline using the already downloaded metadata and samples. The script resolves the experiment root itself and restores the caller's working directory. It prefers an experiment-local environment and otherwise uses the existing repository-local environment. It overwrites the current result files; copy results to a named run directory first if retaining a run. It performs no network requests. The richer Windows inventory is a separate read-only command, run from the experiment directory:

```powershell
./scripts/inventory.ps1
```

The storage benchmark writes one new 128 MiB pseudorandom temporary file under `.apex-cache`, reads it at queue depth 1, and removes only that temporary fixture. Memory copy uses two 64 MiB arrays. GPU profiling uses bounded buffers and reference matrices; it neither reserves all VRAM nor changes global GPU settings.

## Recreate the environment

For a clean machine, enter this experiment directory, create a private venv, install the small packages in `requirements-measured.txt`, then install the experiment package:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-measured.txt
.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

If setuptools is absent, install it first. No CUDA toolkit is required for the driver profiler; an NVIDIA CUDA-capable driver is. Non-Windows environments skip the Windows direct-I/O measurement, and GPU unavailability is reported explicitly. The calibrated comparison requires both successful direct-I/O and pinned-transfer measurements; it refuses to fabricate substitutes.

Metadata/samples can be recreated separately, with network access:

```powershell
& $python scripts/fetch_model_metadata.py
& $python scripts/fetch_weight_samples.py
```

Metadata is capped at 8 MiB and pinned to the resolved official revision. Weight samples are capped at 4 MiB; this run fetched exactly 3 MiB in 24 ranges. A server that ignores Range is rejected before consuming its payload. These scripts never download full shards. Samples are ignored by Git; their manifest records exact offsets, tensor names and SHA-256 hashes. Model ownership and Apache-2.0 licensing remain with the original publisher.

**Downloading the full model still requires explicit authorization.** This platform has made no purchases, changed no pagefile/BIOS settings, and exposed no network service. The user is acting independently; no educational institution sponsors or authorizes this project.
