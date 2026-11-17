# APEX initial experimental results

Date: November 17, 2026. All rates below use decimal GB/s; capacities explicitly marked GiB are binary.

This is a research platform, not a completed inference engine. Measured microbenchmarks, sampled weight compression, and synthetic simulation are separate evidence categories. No inference token rate or quality improvement has been demonstrated.

## Verified target metadata

Official revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`: 36 layers, 128 experts/layer, top-4. Header-derived logical parameters: 116,829,156,672. Checkpoint tensor payload: 60.768 GiB; serialized files: 65,248,893,184 bytes.

Expert serialized mean: 13,236,480 bytes (12.623 MiB), including packed data, scales and biases. One decode step's logical expert accesses: 1.906 GB. This is a logical access sum, not actual PCIe or SSD traffic.

## Measured hardware

RAM copy payload: **16.55 GB/s**. Counting logical reads plus writes: 33.10 GB/s. Warm pages, single NumPy copy call, not a STREAM result.

| Storage path | Access | Block | GB/s | Median latency (us) |
|---|---|---:|---:|---:|
| buffered_warm | sequential | 1024 KiB | 6.188 | 166.0 |
| buffered_warm | random | 4 KiB | 0.948 | 3.6 |
| buffered_warm | random | 1024 KiB | 6.783 | 155.6 |
| windows_unbuffered | sequential | 1024 KiB | 4.039 | 247.6 |
| windows_unbuffered | random | 4 KiB | 0.057 | 67.2 |
| windows_unbuffered | random | 1024 KiB | 3.686 | 275.8 |

Storage: one 128 MiB pseudorandom fixture, queue depth 1, one write pass. Unbuffered reads bypass the OS file cache but do not disable device cache. Cached results must not be used as NVMe throughput. JSON includes every timing sample.

| CUDA host buffer | Direction | Size | GB/s |
|---|---|---:|---:|
| pageable | h2d | 32 MiB | 8.97 |
| pageable | d2h | 32 MiB | 11.93 |
| pinned | h2d | 32 MiB | 13.25 |
| pinned | d2h | 32 MiB | 14.03 |

CUDA Driver API status: `ok`. Compute capability: [12, 0]. Memory after context initialization: {'free': 7372537856, 'total': 8518369280}. WDDM/driver memory reporting differs from physical board capacity; nvidia-smi reported 8,151 MiB total. Use conservative budgets, not reported free bytes as an allocation guarantee.

| FP32 projection | Batch | CPU NumPy ms | GPU reference PTX ms | Max absolute error |
|---|---:|---:|---:|---:|
| 512 x 512 | 1 | 0.0186 | 0.0493 | 1.2e-07 |
| 512 x 512 | 16 | 0.2963 | 0.1088 | 2.2e-07 |
| 2880 x 2880 | 1 | 0.6214 | 0.3282 | 1e-06 |
| 2880 x 2880 | 16 | 1.4565 | 0.3724 | 1.1e-06 |

Projection measurements exclude weight upload and use FP32 rather than MXFP4. CPU uses the local BLAS defaults; GPU uses an intentionally unoptimized SIMT kernel. These are correctness/measurement plumbing checks, not an optimized CPU-vs-Tensor-Core comparison. The simulation excludes compute by default.

## Measured lossless compression

All candidates round-trip exactly. Sizes include 20-byte tile envelopes and an 8-byte offset per tile plus terminal offset. The table shows 64 KiB tiles, no transform, to make the baseline codecs directly comparable. Other tile sizes and reversible transforms are in compression.json.

| Corpus | Codec | Stored / original | Decode ms per MiB |
|---|---|---:|---:|
| random-control | raw | 1.0004 | 0.102 |
| random-control | gzip | 1.0555 | 3.530 |
| random-control | lz4 | 1.0044 | 0.169 |
| random-control | zstd | 1.0006 | 0.212 |
| synthetic-packed-nibbles | raw | 1.0004 | 0.118 |
| synthetic-packed-nibbles | gzip | 1.0619 | 3.326 |
| synthetic-packed-nibbles | lz4 | 1.0044 | 0.170 |
| synthetic-packed-nibbles | zstd | 0.8989 | 1.046 |
| synthetic-bf16 | raw | 1.0004 | 0.110 |
| synthetic-bf16 | gzip | 1.0503 | 3.462 |
| synthetic-bf16 | lz4 | 1.0044 | 0.178 |
| synthetic-bf16 | zstd | 0.7807 | 1.144 |
| zero-control | raw | 1.0004 | 0.108 |
| zero-control | gzip | 0.0105 | 0.381 |
| zero-control | lz4 | 0.0045 | 0.258 |
| zero-control | zstd | 0.0007 | 0.287 |
| target-packed-blocks | raw | 1.0004 | 0.303 |
| target-packed-blocks | gzip | 1.0543 | 4.146 |
| target-packed-blocks | lz4 | 1.0044 | 0.426 |
| target-packed-blocks | zstd | 0.9649 | 1.314 |
| target-scales | raw | 1.0004 | 0.108 |
| target-scales | gzip | 0.3840 | 1.739 |
| target-scales | lz4 | 0.4502 | 0.895 |
| target-scales | zstd | 0.2189 | 2.633 |

Target samples are limited portions of experts 0 and 64 in layers 0, 18 and 35, with packed blocks and scales sampled separately. Samples do not establish a whole-checkpoint ratio. The synthetic corpora are test controls. Native decoder temporary memory and CPU utilization are not fully captured by Python allocation/process-time measurements.

## Synthetic cache replay

Identical deterministic traces for every policy, three seeds per distribution, empty initial caches. Simulated budgets: 8 GiB RAM expert cache and 1 GiB GPU expert cache. Step latency here is serialized data movement plus decoding only; it is **not actual token latency**. No inferred tokens/s is reported.

| Distribution | Policy | Storage / host | Mean step ms, mean across seeds | Range across seeds | SSD GiB, mean | GPU hit rate |
|---|---|---|---:|---:|---:|---:|
| uniform | none | plain / raw | 660.89 | 660.89–660.89 | 113.61 | 0.0% |
| uniform | none | zstd / compressed | 3146.35 | 3146.35–3146.35 | 104.65 | 0.0% |
| uniform | none | zstd / raw | 3146.35 | 3146.35–3146.35 | 104.65 | 0.0% |
| uniform | none | adaptive / compressed | 1190.05 | 1190.05–1190.05 | 113.66 | 0.0% |
| uniform | lru | plain / raw | 602.15 | 601.14–602.82 | 100.70 | 0.0% |
| uniform | lru | zstd / compressed | 3079.37 | 3078.65–3079.89 | 89.93 | 0.0% |
| uniform | lru | zstd / raw | 2805.24 | 2799.38–2809.15 | 92.76 | 0.0% |
| uniform | lru | adaptive / compressed | 1131.28 | 1130.27–1131.96 | 100.75 | 0.0% |
| uniform | lfu | plain / raw | 592.30 | 591.34–593.71 | 98.99 | 1.4% |
| uniform | lfu | zstd / compressed | 3041.76 | 3040.21–3044.08 | 89.94 | 1.4% |
| uniform | lfu | zstd / raw | 2757.82 | 2752.09–2765.79 | 91.18 | 1.4% |
| uniform | lfu | adaptive / compressed | 1113.97 | 1113.13–1115.56 | 99.03 | 1.4% |
| uniform | cost | plain / raw | 592.30 | 591.34–593.71 | 98.99 | 1.4% |
| uniform | cost | zstd / compressed | 3041.76 | 3040.21–3044.08 | 89.94 | 1.4% |
| uniform | cost | zstd / raw | 2757.82 | 2752.09–2765.79 | 91.18 | 1.4% |
| uniform | cost | adaptive / compressed | 1113.97 | 1113.13–1115.56 | 99.03 | 1.4% |
| hotset | none | plain / raw | 660.89 | 660.89–660.89 | 113.61 | 0.0% |
| hotset | none | zstd / compressed | 3146.35 | 3146.35–3146.35 | 104.65 | 0.0% |
| hotset | none | zstd / raw | 3146.35 | 3146.35–3146.35 | 104.65 | 0.0% |
| hotset | none | adaptive / compressed | 1190.05 | 1190.05–1190.05 | 113.66 | 0.0% |
| hotset | lru | plain / raw | 311.17 | 308.50–314.84 | 36.77 | 0.0% |
| hotset | lru | zstd / compressed | 2804.72 | 2802.59–2808.12 | 29.58 | 0.0% |
| hotset | lru | zstd / raw | 1115.57 | 1100.04–1136.85 | 33.87 | 0.0% |
| hotset | lru | adaptive / compressed | 840.18 | 837.51–843.85 | 36.78 | 0.0% |
| hotset | lfu | plain / raw | 242.48 | 241.42–244.32 | 24.69 | 9.5% |
| hotset | lfu | zstd / compressed | 2516.23 | 2511.39–2521.19 | 22.13 | 9.5% |
| hotset | lfu | zstd / raw | 782.58 | 776.21–795.03 | 22.74 | 9.5% |
| hotset | lfu | adaptive / compressed | 721.04 | 720.13–721.54 | 24.70 | 9.5% |
| hotset | cost | plain / raw | 242.48 | 241.42–244.32 | 24.69 | 9.5% |
| hotset | cost | zstd / compressed | 2516.23 | 2511.39–2521.19 | 22.13 | 9.5% |
| hotset | cost | zstd / raw | 782.58 | 776.21–795.03 | 22.74 | 9.5% |
| hotset | cost | adaptive / compressed | 721.04 | 720.13–721.54 | 24.70 | 9.5% |
| phase-shift | none | plain / raw | 660.89 | 660.89–660.89 | 113.61 | 0.0% |
| phase-shift | none | zstd / compressed | 3146.35 | 3146.35–3146.35 | 104.65 | 0.0% |
| phase-shift | none | zstd / raw | 3146.35 | 3146.35–3146.35 | 104.65 | 0.0% |
| phase-shift | none | adaptive / compressed | 1190.05 | 1190.05–1190.05 | 113.66 | 0.0% |
| phase-shift | lru | plain / raw | 325.99 | 323.03–327.63 | 40.02 | 0.0% |
| phase-shift | lru | zstd / compressed | 2822.28 | 2820.52–2824.65 | 33.44 | 0.0% |
| phase-shift | lru | zstd / raw | 1201.58 | 1184.42–1211.13 | 36.87 | 0.0% |
| phase-shift | lru | adaptive / compressed | 855.00 | 852.05–856.65 | 40.04 | 0.0% |
| phase-shift | lfu | plain / raw | 395.91 | 394.60–397.26 | 56.95 | 4.9% |
| phase-shift | lfu | zstd / compressed | 2764.39 | 2761.45–2766.58 | 49.71 | 4.9% |
| phase-shift | lfu | zstd / raw | 1641.79 | 1634.90–1649.70 | 52.46 | 4.9% |
| phase-shift | lfu | adaptive / compressed | 898.83 | 896.96–900.14 | 56.97 | 4.9% |
| phase-shift | cost | plain / raw | 395.91 | 394.60–397.26 | 56.95 | 4.9% |
| phase-shift | cost | zstd / compressed | 2764.39 | 2761.45–2766.58 | 49.71 | 4.9% |
| phase-shift | cost | zstd / raw | 1641.79 | 1634.90–1649.70 | 52.46 | 4.9% |
| phase-shift | cost | adaptive / compressed | 898.83 | 896.96–900.14 | 56.97 | 4.9% |

Compressed scenarios use measured sample ratios and decoding times, weighted by the full checkpoint's actual packed/scale/bias sizes. Extrapolation to whole experts is unvalidated. Each cache respects its own byte budget; duplication across RAM and GPU is counted in both. The cost policy is frequency x avoided reload cost / bytes; equal-sized/equal-cost experts can make it equivalent to LFU.

## Decisions and next evidence

1. Preserve raw storage as a real candidate. Compression must beat read + decode + transfer cost, not merely reduce bytes.
2. Use current llama.cpp expert caching and CPU-miss handling as the first prospective inference baseline. Reassess custom components against it, with exact version and configuration pinned.
3. Collect actual routing and full-expert timings before making target-performance claims. Synthetic hot sets intentionally favor reuse; uniform and phase-shifting traces expose weaker conditions.
4. Fit transfer and SSD cost curves over more sizes and queue depths, then test overlap. Current cost sums are intentionally serial and linear.
5. Defer speculative decoding, learned policies, public serving, APEX-Q, and fused compression kernels until simpler baselines justify them.

## Reproduction and limitations

Run `scripts/run_initial.ps1` from this experiment's root, or `experiments/001-apex-inference/scripts/run_initial.ps1` from the repository root. It reuses the bounded downloaded metadata/samples and produces new local measurements. See the experiment README.md for individual commands and separate network steps. Full-model downloading requires prior authorization.

No real-model inference baseline, real routing trace, target quality/tool-call evaluation, asynchronous prefetch controller, learned cache, or production memory allocator has been implemented. Hardware measurements were made on an active desktop, without controlling background applications. Small sample count and changing load limit precision.
