# Evidence-based roadmap

## Initial deliverable status

| Milestone | Status | Evidence / remaining work |
|---|---|---|
| 1: Research environment | Implemented | Local Git repository, venv, executable package, Windows/Python inventories and documentation |
| 2: Hardware profiler | Initial implementation measured | RAM, direct/cached SSD, both PCIe directions, allocation, VRAM and FP32 reference projections. Broader I/O/concurrency and tuned expert kernels remain |
| 3: Trace simulator | Implemented for synthetic traces | Validation, deterministic traces, four policies, exact budget/accounting tests; real routing capture remains |
| 4: Compression laboratory | Initial implementation measured | Four codecs, three transforms, independent tiles, real small samples and synthetic controls; whole experts and wider sampling remain |
| 5: Joint optimizer | Partial | Compression/residency and explicit compute placement cost model work. Async prefetch, contention, KV competition, full scheduling and real trace calibration remain |
| 6: Real-model integration | Not completed | No model has been loaded or generated tokens; no engine speedup demonstrated |
| 7: Native acceleration | Not completed as an optimization | Driver profiler executes reference PTX, but no new accelerated inference/decompression path has beaten a baseline |

This establishes the initial Level 1 research foundation. Levels 2–6 require independent evidence; microbenchmark speed differences and simulated cache wins do not complete them.

## Next experiments, in order

1. **Pin and inspect a current llama.cpp baseline.** Its main branch now contains MoE caching, CPU-miss execution, and a routing-based tuning script. Test those existing mechanisms before inventing another cache runtime. Record exact revision, build options, load mode, thread count, context and expert placement. Keep instrumentation isolated.
2. **Obtain actual routes.** Reuse a local small MoE if one is available, or request a specific bounded model download. Label those routes `real-small-model`; they do not establish GPT-OSS-120B behavior. Capture target routes only after the full checkpoint download is authorized and a memory plan is reviewed.
3. **Characterize the cost surface.** Extend direct reads to 16/64/256 KiB and full-expert-sized requests, queue depths and warm/cold ranges. Fit latency plus bandwidth with confidence intervals. Repeat GPU transfers over size and pinning conditions. Include background-load observations.
4. **Test a tuned expert path.** Compare real MXFP4 expert computation on CPU and GPU with the same inputs, backend numerical contract and batch sizes. Include activation transfers and staging. The naive FP32 profiler is only a starting measurement tool.
5. **Expand the compression sample.** Predeclare randomized layer/expert/offset sampling and separate packed values, scales, biases and non-expert BF16 weights. Measure complete expert reconstruction, cold reads and hot hits. Preserve raw passthrough when decode costs outweigh byte savings. Avoid extrapolating the scale-stream ratio to all weights.
6. **Compare policy alternatives on real traces.** Implement per-layer quotas, aging LFU and a source-verified Least-Stale reproduction only after baselines are stable. Measure per-layer conflicts and phase changes. Add prefetch only with bounded outstanding bytes, no lookahead leakage, waste accounting and automatic disable criteria.
7. **Validate end to end.** Compare actual TTFT, decode, whole task latency, peak physical RAM/VRAM, SSD traffic and correctness at fixed settings. Include tool-call validity and agent-task success before claiming educational usability.

## Findings changing priorities

- Raw tiles win the initial measured access-cost objective on the small target samples. Additional lossless compression is not automatically an inference optimization.
- Sampled MXFP4 packed bytes compress little, while scales compress substantially; their byte weights are very different. Any aggregate estimate must respect that.
- Cross-expert XOR bases increased stored size for every sampled pair; exact 4 KiB block matches were absent in those comparisons. APEX-Q and shared-base factorization remain low priority until better evidence appears.
- Stable-hot-set and phase-shift traces favor different policies. Global LFU without aging is not an automatic upgrade over LRU.
- Existing engine features overlap APEX's proposal. Reusing them could be more valuable than maintaining a separate runtime.

No cloud rentals, purchases, public multi-user hosting or VPS changes are prerequisites. Large downloads, new toolchains and full-model experiments are separate steps with explicit scope; no additional spending is authorized.
