# Experiment 002 protocol

## Research question and scope

Can observed expert reuse improve actual local inference latency, and does a cache help more than ordinary CPU/GPU offload? A positive result is specific to the tested workload, software and hardware. A negative result, allocation failure or quality regression is retained.

This is an independent experiment within Student AI Platform, building on Experiment 001's measurement and cache-simulation tools. The executable engine is upstream llama.cpp; this experiment does not claim to have built a new APEX inference engine.

GPT-OSS-20B MXFP4 has 24 layers, 32 experts per layer and four active experts per token. Its approved GGUF checkpoint fits host RAM. It does not reproduce the beyond-RAM storage pressure of GPT-OSS-120B.

## Frozen model and runtime

- GGUF repository: `ggml-org/gpt-oss-20b-GGUF`, revision `41e8a94d15d62070b36cf4553a0250717ecf46b8`.
- File: `gpt-oss-20b-MXFP4.gguf`, 12,109,566,624 bytes.
- SHA-256: `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901`.
- llama.cpp release `b12359`, commit `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`, Windows x64 CUDA 13.5 build.
- Exact release assets, download hashes, runtime DLL hashes, environment and primary-source URLs are under `results/setup/`.
- Localhost only. No tools, shell execution of model output, public service, paid API or system configuration changes.

## Calibration

Two prompts about dimensional analysis and binary search are used only to select configuration settings. Each produces 96 output tokens after a separate 32-token warm-up. The measured trials and failures remain in `results/runs/cal-*/`.

Candidates cover six versus ten CPU threads, eight versus twelve statically offloaded layers, 1–4 GiB expert caches, and CPU shares of 0, 0.5 and 0.95 for cache misses. These are bounded candidates, not an exhaustive global search. Larger caches failed with `--load-mode none` on this system; a separate calibration established workable memory-mapped loading. The held-out comparison uses memory-mapped loading for every configuration.

## Held-out evaluation

Four configurations:

| Name | Placement |
|---|---|
| `cpu` | All expert weights on CPU, other layers on GPU; no expert cache |
| `static12` | Ordinary static offload of 12 layers; no expert cache |
| `cache4096` | CPU expert weights plus 4,096 MiB GPU expert cache; all misses uploaded |
| `hybrid4096` | Same cache; 95% of cache misses computed on CPU after the cache is full |

Common controls: six CPU threads and six batch threads, batch size 256, physical batch size 256, context 2,048, one request slot, flash attention enabled, automatic fitting disabled, memory-mapped loading, prompt cache disabled, temperature zero and seed 1729. All actual commands are stored per run. The 95% setting applies to **misses**, not to 95% of total computation.

Each process receives the same 32-token warm-up. Expert-cache state persists across requests inside a process and resets when the process exits. Every request processes its full prompt (`cache_prompt=false`; the recorded `cache_n` must be zero). KV reuse and expert caching are separate mechanisms.

Four rounds rotate configuration order using a Latin square, so every configuration occupies every ordinal position once. The four performance prompts rotate order between rounds; order is identical across configurations within each round. Six answer checks follow. This produces 16 processes, 64 fixed-length performance requests and 96 answer-check requests (six distinct checks, repeated four times per configuration).

Performance prompts cover biology, code review, historical sources and algebra instruction. Each generates exactly 128 tokens with EOS ignored. These are fixed-length throughput workloads, not necessarily complete answers. The native template is applied, then the assistant's `final` channel is explicitly selected to remove variation in hidden reasoning length. Consequently the primary measurements characterize **direct-final generation**, not normal reasoning-enabled chat.

Six separate, naturally terminated answer checks cover arithmetic, algebra, sorting, code comprehension, structured extraction and logic. They use the same direct-final path with a 64-token cap. Exact text/JSON checks are deliberately narrow; they do not certify educational usefulness or model-wide quality. JSON fences and surrounding whitespace are accepted. Generated code is never executed.

## Timing and resource definitions

- **Startup:** subprocess launch to successful health response. The OS file cache is not flushed; these are process-start measurements with a warm filesystem cache, not cold-disk loading tests.
- **Prefill:** the engine's `prompt_ms`, recorded with prompt token count.
- **TTFT:** elapsed client time from starting the streaming HTTP request to the first streamed token ID. First nonempty text is also recorded. Template formatting occurs before this timer.
- **Decode:** the pinned engine's `predicted_ms`. Its throughput uses `predicted_n - 1`, because the first output token comes from prompt evaluation. Aggregates divide total timed decode steps by total decode seconds.
- **Request latency:** time to receive the complete streaming response, including local HTTP overhead. Startup, prior warm-up and template formatting are excluded.
- **Memory:** periodic process RSS, system available RAM and whole-GPU VRAM use. RSS includes mapped file pages, not just private allocations. VRAM includes the desktop and other applications, not only the server. Sampling can miss short peaks.
- GPU temperature and utilization are sampled alongside memory. The same monitoring runs for every measured configuration. Other desktop applications are not terminated or controlled; this is a local-machine experiment, not an isolated laboratory certification.

Windows swap-performance counters were unavailable in the first smoke test. No hard-fault/pagefile or physical PCIe-traffic measurements are claimed. Loading errors are evidence about the tested runtime path, not a definitive explanation of the hardware's physical limits.

## Routing and observer overhead

The pinned engine's `LLAMA_MOE_CACHE_TRACE` emits actual expert choices at cache slot lookups. Three paired trace-on/trace-off repetitions use the selected hybrid configuration and the four performance prompts. Pair order alternates. Performance timings exclude the extra flush workload.

The trace exposes small-batch routing: the four-token prefill tail in layers 0–22, one prefill row in layer 23, and every decode step used to produce later output tokens. A two-row startup probe (one row in the last layer) is labeled separately and updates replay cache state without contributing to workload statistics. Bulk prefill has no cache slot lookup and is absent. The parser checks headers, expert ranges, top-k uniqueness, ordered layer cycles and exact per-request decode coverage. It fails on an unexpected layout instead of manufacturing missing routes.

On Windows, terminating the hidden server does not flush C stdio buffers. A separate 64-token tail workload follows traced requests, pushing their buffered bytes to disk. The parser requires its boundary after all measured requests and excludes its cycles and any incomplete trailing record. This preserves complete measured routing without claiming the final file contains every tail token.

Compressed raw traces, normalized real-model events, hashes and coverage reports are retained. Experiment 001's global LRU, LFU and cost-aware policies replay the same events at fixed byte budgets. Warm-up and observed prefill-tail events update simulated state; reported rates cover evaluated decode. The replay is a policy comparison, not a measured engine hit counter, PCIe-traffic measurement or inference-speed prediction. Upstream's live cache uses a different, decaying-frequency policy.

## Quality follow-up

The frozen direct-final test exposed an algebra error. A separate diagnostic uses normal `/v1/chat/completions`, `reasoning_effort=low`, temperature zero, and a 256-token cap on the same six checks with CPU, static and hybrid configurations. It is explicitly post-evaluation and does not retune the performance settings or replace the direct-final results. Answer text, generation counts, stop reasons and timing are retained.

## Interpretation limits

Four repeated rounds on four short performance prompts establish a bounded local comparison. They do not establish long-context behavior, concurrency, broad accuracy, an optimal cache policy, exact numerical equivalence, energy efficiency, or GPT-OSS-120B feasibility. Any recommendation must distinguish actual inference measurements from replay estimates and follow-up diagnostics.

## Separately authorized sweet-spot phase

After this primary series completed, the user requested a sweet-spot search, checks for falloff and later recovery, and graphs. An initial 0.5–4.5 GiB sweep was subsequently expanded at the user's request to **4–32 GiB requested cache budgets**. It is preserved as a separate adaptive experiment under `results/sweet-spot/`.

The broad stress sweep uses two 192-token requests, four threads, 90% and 95% CPU miss shares, a 180-second process cap and a 2 GiB available-RAM reserve. A shorter recovery probe uses one 128-token request, 90% CPU miss share, a 90-second cap and a 1 GiB reserve. These different protocols are graphed separately. Runs stopped by a reserve or time guard are not represented as zero throughput or proof that an intrinsic hardware limit has been found.

The engine caps its cache bank at the number of model experts plus reserved slots. With the measured 13,219,200 bytes per expert, 768 experts and five extra slots, the full buffer is about 9.52 GiB. Requested budgets above this cap do not correspond to distinct effective cache capacities. The records distinguish requested budget, reported allocation, failed allocation and successful inference.

Final validation uses the same four fresh 128-token prompts across all contenders, three rotated rounds, a 180-second process cap and a 1 GiB reserve. It retains the strongest setting from each broad workload protocol, the original reference, a conservative alternative, a 6 GiB falloff control and static offload. Selection is frozen before these fresh results: smallest cache within 5% of the fastest cache setting completing every validation round, with a separate conservative choice under the original headroom targets. Normal-reasoning answer checks follow the performance choice.
