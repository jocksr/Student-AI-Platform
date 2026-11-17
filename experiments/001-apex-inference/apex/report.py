from collections import defaultdict
from pathlib import Path
import statistics
from .common import read_json


def render(root="results"):
    root=Path(root)
    target=read_json(root/"model-metadata/summary.json")
    memory=read_json(root/"memory.json"); storage=read_json(root/"storage.json"); gpu=read_json(root/"gpu.json")
    compression=read_json(root/"compression.json"); comparison=read_json(root/"comparison.json")
    lines=["# APEX initial experimental results", "", "Date: November 17, 2026. All rates below use decimal GB/s; capacities explicitly marked GiB are binary.","",
        "This is a research platform, not a completed inference engine. Measured microbenchmarks, sampled weight compression, and synthetic simulation are separate evidence categories. No inference token rate or quality improvement has been demonstrated.","",
        "## Verified target metadata", "",
        f"Official revision `{target['revision']}`: {target['layers']} layers, {target['experts_per_layer']} experts/layer, top-{target['top_k']}. Header-derived logical parameters: {target['logical_parameters_from_shapes']:,}. Checkpoint tensor payload: {target['tensor_payload_bytes']/2**30:.3f} GiB; serialized files: {target['serialized_checkpoint_bytes']:,} bytes.","",
        f"Expert serialized mean: {target['expert_mean_serialized_bytes']:,} bytes ({target['expert_mean_serialized_bytes']/2**20:.3f} MiB), including packed data, scales and biases. One decode step's logical expert accesses: {target['expert_mean_serialized_bytes']*target['layers']*target['top_k']/1e9:.3f} GB. This is a logical access sum, not actual PCIe or SSD traffic.","",
        "## Measured hardware", "",
        f"RAM copy payload: **{memory['payload_GB_s']:.2f} GB/s**. Counting logical reads plus writes: {memory['logical_read_plus_write_GB_s']:.2f} GB/s. Warm pages, single NumPy copy call, not a STREAM result.","",
        "| Storage path | Access | Block | GB/s | Median latency (us) |", "|---|---|---:|---:|---:|"]
    for r in storage["rows"]:
        if "throughput_GB_s" in r: lines.append(f"| {r['mode']} | {r['pattern']} | {r['block_bytes']//1024} KiB | {r['throughput_GB_s']:.3f} | {r['median_ns']/1000:.1f} |")
    lines += ["", "Storage: one 128 MiB pseudorandom fixture, queue depth 1, one write pass. Unbuffered reads bypass the OS file cache but do not disable device cache. Cached results must not be used as NVMe throughput. JSON includes every timing sample.","",
        "| CUDA host buffer | Direction | Size | GB/s |", "|---|---|---:|---:|"]
    for r in gpu["transfer_rows"]:
        if r["bytes"]>=2**20: lines.append(f"| {r['host_memory']} | {r['direction']} | {r['bytes']//2**20} MiB | {r['GB_s']:.2f} |")
    lines += ["",f"CUDA Driver API status: `{gpu['status']}`. Compute capability: {gpu.get('compute_capability')}. Memory after context initialization: {gpu.get('vram_after_init')}. WDDM/driver memory reporting differs from physical board capacity; nvidia-smi reported 8,151 MiB total. Use conservative budgets, not reported free bytes as an allocation guarantee.","",
        "| FP32 projection | Batch | CPU NumPy ms | GPU reference PTX ms | Max absolute error |", "|---|---:|---:|---:|---:|"]
    for r in gpu["compute_rows"]:
        lines.append(f"| {r['k']} x {r['n']} | {r['batch']} | {r['cpu_numpy']['median_ns']/1e6:.4f} | {r['gpu_naive_ptx']['median_ns']/1e6:.4f} | {r['max_abs_error']:.2g} |")
    lines += ["","Projection measurements exclude weight upload and use FP32 rather than MXFP4. CPU uses the local BLAS defaults; GPU uses an intentionally unoptimized SIMT kernel. These are correctness/measurement plumbing checks, not an optimized CPU-vs-Tensor-Core comparison. The simulation excludes compute by default.","",
        "## Measured lossless compression", "", "All candidates round-trip exactly. Sizes include 20-byte tile envelopes and an 8-byte offset per tile plus terminal offset. The table shows 64 KiB tiles, no transform, to make the baseline codecs directly comparable. Other tile sizes and reversible transforms are in compression.json.","",
        "| Corpus | Codec | Stored / original | Decode ms per MiB |", "|---|---|---:|---:|"]
    for r in compression["rows"]:
        if r["tile_size"]==65536 and r["transform"]=="none":
            lines.append(f"| {r['corpus']} | {r['codec']} | {r['ratio']:.4f} | {r['decode_ns_per_byte']*2**20/1e6:.3f} |")
    lines += ["", "Target samples are limited portions of experts 0 and 64 in layers 0, 18 and 35, with packed blocks and scales sampled separately. Samples do not establish a whole-checkpoint ratio. The synthetic corpora are test controls. Native decoder temporary memory and CPU utilization are not fully captured by Python allocation/process-time measurements.","",
        "## Synthetic cache replay", "", "Identical deterministic traces for every policy, three seeds per distribution, empty initial caches. Simulated budgets: 8 GiB RAM expert cache and 1 GiB GPU expert cache. Step latency here is serialized data movement plus decoding only; it is **not actual token latency**. No inferred tokens/s is reported.","",
        "| Distribution | Policy | Storage / host | Mean step ms, mean across seeds | Range across seeds | SSD GiB, mean | GPU hit rate |", "|---|---|---|---:|---:|---:|---:|"]
    grouped=defaultdict(list)
    for r in comparison["rows"]: grouped[(r["workload"],r["policy"],r["storage_profile"],r["host_representation"])].append(r)
    for (w,p,s,h),group in grouped.items():
        times=[r["mean_step_movement_ms"] for r in group]
        lines.append(f"| {w} | {p} | {s} / {h} | {statistics.mean(times):.2f} | {min(times):.2f}–{max(times):.2f} | {statistics.mean(r['metrics']['ssd_bytes'] for r in group)/2**30:.2f} | {statistics.mean(r['gpu_hit_rate'] for r in group):.1%} |")
    lines += ["", "Compressed scenarios use measured sample ratios and decoding times, weighted by the full checkpoint's actual packed/scale/bias sizes. Extrapolation to whole experts is unvalidated. Each cache respects its own byte budget; duplication across RAM and GPU is counted in both. The cost policy is frequency x avoided reload cost / bytes; equal-sized/equal-cost experts can make it equivalent to LFU.","",
        "## Decisions and next evidence", "",
        "1. Preserve raw storage as a real candidate. Compression must beat read + decode + transfer cost, not merely reduce bytes.",
        "2. Use current llama.cpp expert caching and CPU-miss handling as the first prospective inference baseline. Reassess custom components against it, with exact version and configuration pinned.",
        "3. Collect actual routing and full-expert timings before making target-performance claims. Synthetic hot sets intentionally favor reuse; uniform and phase-shifting traces expose weaker conditions.",
        "4. Fit transfer and SSD cost curves over more sizes and queue depths, then test overlap. Current cost sums are intentionally serial and linear.",
        "5. Defer speculative decoding, learned policies, public serving, APEX-Q, and fused compression kernels until simpler baselines justify them.","",
        "## Reproduction and limitations", "",
        "Run `scripts/run_initial.ps1` from this experiment's root, or `experiments/001-apex-inference/scripts/run_initial.ps1` from the repository root. It reuses the bounded downloaded metadata/samples and produces new local measurements. See the experiment README.md for individual commands and separate network steps. Full-model downloading requires prior authorization.","",
        "No real-model inference baseline, real routing trace, target quality/tool-call evaluation, asynchronous prefetch controller, learned cache, or production memory allocator has been implemented. Hardware measurements were made on an active desktop, without controlling background applications. Small sample count and changing load limit precision.",""]
    path=root/"initial-report.md"; path.write_text("\n".join(lines),encoding="utf-8"); return path
