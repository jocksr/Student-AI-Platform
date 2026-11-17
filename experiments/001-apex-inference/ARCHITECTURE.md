# Architecture and evidence boundaries

APEX separates observations from predictions. Hardware measurements, weight byte measurements, routing traces, and simulations carry distinct `kind` fields, timestamps, software versions, and Python-source fingerprints. The target checkpoint's revision and tensor layout are recorded independently. A simulation cannot certify inference performance.

```text
Windows / NVIDIA driver ──> hardware + microbenchmark JSON
Official checkpoint metadata ──> verified model shape / byte counts
Bounded weight ranges ──> exact codec laboratory ──> measured tile profiles
Synthetic or imported JSONL routes ──> cache replay <── measured cost inputs
                                                   │
                                                   v
                                     comparison JSON + experimental report
```

## Modules

| File | Responsibility |
|---|---|
| `apex/hardware.py`, `scripts/inventory.ps1` | Hardware and environment inventory; no system modifications |
| `apex/benchmarks.py` | RAM copy, buffered/direct storage; bounded generated fixtures |
| `apex/gpu.py` | Driver API context, allocations, pinned/pageable copies, reference PTX GEMM |
| `apex/traces.py` | Immutable event schema, deterministic workloads, validation, co-activation analysis |
| `apex/cache.py` | Immutable expert description; byte-bounded mutable cache state |
| `apex/compression.py`, `apex/container.py` | Reversible transforms, established codecs, checksummed tiles, offset index |
| `apex/simulation.py` | Explicit RAM/GPU residency and serialized movement/decode/compute decisions |
| `apex/experiments.py` | Identical-trace comparisons, measured-profile integration, cross-expert experiments |
| `apex/report.py`, `apex/cli.py` | Local report and command interface |

## Residency and costs

The model is an immutable collection of `(layer, expert)` objects. RAM stores either the original serialized representation or a compressed tile representation. GPU cache accounting uses the original serialized expert size, assuming a future kernel can consume native packed MXFP4. **Actual engine conversion/padding/scratch costs are not yet validated.** A backend that expands to FP16/BF16 needs a much larger GPU budget than this model.

GPU hits avoid host accesses. GPU misses consult RAM. RAM misses read compressed storage, then decode before CPU or GPU execution. Raw RAM residency decodes only at load; compressed RAM residency decodes at each GPU miss. GPU transfers contain restored serialized bytes; no GPU entropy decoder exists. RAM/GPU caches may hold the same expert and account for both copies. Oversized objects stream through without evicting the whole cache.

Cost is explicitly serial: storage + decoding + transfer + compute. Effective measured bandwidths already include the measured call overhead, so fixed overhead is zero in the calibrated movement experiment to avoid double-counting. Linear extrapolation from 1 MiB storage reads and 32 MiB PCIe copies is a limitation. Queueing, overlap, OS page-cache hits and memory pressure are not simulated.

CPU/adaptive placement is implemented at the cost-model level and unit-tested. It requires explicit positive CPU and GPU compute estimates. The initial comparison omits compute rather than treating an FP32 reference projection as a tuned MXFP4 expert. Activation round-trip cost is a separate term. Batches have trace metadata but do not yet change the placement cost automatically.

## Cache policies

No-cache disables both caches. LRU evicts by last access. LFU uses cumulative per-expert frequency, including earlier misses, with recency as a deterministic tie breaker. Cost-aware retention uses frequency times avoided reload latency divided by resident bytes. These are simple engineering baselines; equal costs/sizes make cost-aware behave like LFU. They are not reproductions of published learned policies.

Caches are single-threaded simulator state. They are not synchronization-safe production allocators. Trace order is authoritative; concurrent request IDs remain distinct, but the replay is serialized. Optional `measurements` fields can carry actual cache, transfer and timing observations without treating synthetic timestamps as measurements.

## Tile format

`APX1` envelope: magic (4 bytes), codec (1), transform (1), reserved (2), original length (4), payload length (4), CRC32 of restored bytes (4), then payload. Decoding is independent and bounded by 16 MiB original tile length. Equality against original bytes is checked during every benchmark iteration. CRC detects ordinary corruption; it is not authentication for untrusted network inputs.

`APC1` container: magic and tile count (8 bytes), uint64 offsets for every tile plus terminal offset, followed by independent tiles. Container reads validate offsets and read only the requested payload. The compression tables include tile envelopes and the offset index; a complete standalone container adds an 8-byte global prefix. Transform scratch allocations can be much larger than the encoded tile, especially bitplanes. Python allocation observations are not full native peak-memory measurements.

Offline selection measures candidates and estimates storage + decode + transfer cost. It does not recompress during inference. First experiments compare plain storage, fixed Zstandard with raw RAM, fixed Zstandard with compressed RAM, and cost-selected tile representations. Raw passthrough remains a first-class baseline.

## Memory budgets

The initial simulator uses 8 GiB host expert cache and 1 GiB GPU expert cache. These are hypothetical reservations, not actual model allocations. The remaining capacity must support roughly 3.96 GiB of checkpoint non-expert tensors (backend layout may differ), Windows/desktop applications, runtime buffers, KV cache, activations, staging and decoder workspaces. KV capacity depends on context, attention implementation, batch and datatype; no usable context length is certified here.

The physical machine has 32 GiB installed RAM, substantially less than the 60.77 GiB checkpoint payload before runtime overhead. An 8 GiB GPU does not close that gap. Efficient disk access and actual routing reuse therefore matter, but no particular speed is promised.
