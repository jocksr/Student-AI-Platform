# Research notes — November 17, 2026

This is an initial source review supporting implementation choices, not an exhaustive literature review or a reproduction of the cited systems. Research uses standard web tools; TinyFish is not used. External code from these repositories has not been executed. Published speedups are not predictions for this PC.

## Target facts verified directly

The official [configuration](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json), index and all 15 safetensors headers were fetched and saved under `results/model-metadata`. The index names 15 actual shards despite its `of-00014` filename suffix; the code enumerates actual filenames rather than trusting that suffix. The resolved revision is `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.

| Header/config-derived property | Value |
|---|---:|
| Layers / experts / selected experts | 36 / 128 / 4 |
| Hidden and expert intermediate width | 2880 / 2880 |
| Logical parameters, including biases | 116,829,156,672 |
| Logical expert parameters | 114,701,598,720 |
| Logical non-expert parameters | 2,127,557,952 |
| All tensor payload bytes | 65,248,815,744 |
| Expert payload bytes | 60,993,699,840 |
| Non-expert payload bytes | 4,255,115,904 |
| Mean serialized expert | 13,236,480 bytes |

The [official implementation](https://github.com/openai/gpt-oss) documents two packed FP4 values per byte, separate scale tensors and BF16 for other tensors. Header-derived parameter counts exclude scale metadata as parameters but include it in storage. The published model/reference implementation is Apache-2.0. The packed expert layout is not an assumption that all tensors occupy four bits.

## Existing engines change the baseline

**llama.cpp is the first prospective local baseline.** Its MIT-licensed [repository](https://github.com/ggml-org/llama.cpp) supports the target; current main-branch [MoE cache documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/moe-cache.md) describes a device expert cache and a configurable fraction of misses computed on CPU. Its [tuning script](https://github.com/ggml-org/llama.cpp/blob/master/scripts/moe-cache-tune.py) records routes, calibrates costs and replays alternatives. These are inspected main-branch files, not merely an issue proposal. The exact packaged Windows release, build and target runtime behavior still need pinning and validation. This directly overlaps APEX's plan: test existing mechanisms before writing replacements.

**vLLM** remains a possible later baseline, but its [installation documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) requires Linux and does not claim native Windows support. WSL is installed here but has not been used for APEX. Its larger dependency/runtime footprint and model residency requirements need a separate feasibility check. No vLLM installation or performance claim is made.

**SGLang HiCache** manages hierarchical KV/prefix caching across GPU, host and storage, as documented in [HiCache](https://docs.sglang.io/docs/advanced_features/hicache). This is a useful separate design axis from expert-weight caching. APEX does not implement it and must not mix private request state across users. Version, backend fit and code license must be checked before any adoption.

## Low-memory systems: evidence and limits

| Source | Relevant evidence | Implication for APEX |
|---|---|---|
| [Fiddler](https://arxiv.org/abs/2402.07033) | CPU/GPU orchestration avoids transferring every cold expert to the GPU | Include CPU computation and activation transfer in placement costs; its Mixtral results on other hardware do not predict GPT-OSS here |
| [Flash-MoE implementation notes](https://github.com/danveloper/flash-moe/blob/main/CLAUDE.md) | C/Metal Qwen engine on Apple hardware; reports custom caches, LZ4 and several predictors losing to simpler approaches | Measure competition with the OS file cache. Unified memory and different model/quantization make its rates non-transferable |
| [WinMoE](https://github.com/idan82labs/WinMoE) | Windows experiments report changes to active-expert count and quantization in their faster configurations | These comparisons are not correctness-preserving baselines for APEX-LC; do not adopt altered routing silently |
| [InferSSD](https://github.com/wici-ai/inferssd) | Describes SSD expert streaming built on llama.cpp | Candidate comparison only; supported target/layout, Windows build, revision, license and correctness not yet validated |

The repositories above are research inputs, not installed APEX backends. No code reuse is approved merely because a project reports a useful benchmark. Retain each benchmark's model, quantization, routing, machine and quality conditions when investigating it further.

## Compression: distinguish exact representation from model changes

- [ZipNN](https://github.com/zipnn/zipnn) is a specialized lossless neural-weight compressor with a verified [MIT license](https://github.com/zipnn/zipnn/blob/main/LICENSE.txt). Its documented high decode rates were obtained on a much larger server CPU; those numbers should not populate this PC's cost model. FP/BF16 compression results do not establish MXFP4 residual redundancy. Not installed or benchmarked here yet.
- [ZipServ](https://arxiv.org/abs/2603.17435), submitted March 18, 2026, co-designs a hardware-aware format and fused decompression/GEMM. That supports investigating a compatible fixed-format kernel, not treating generic Zstandard streams as Tensor Core inputs. Applicability to this target's native MXFP4 needs evidence; no kernel was adopted.
- [NVIDIA nvCOMP](https://developer.nvidia.com/nvcomp) supplies GPU compression/decompression primitives. Separate decode is not the same as fused expert computation. Check algorithm, supported GPU, temporary workspace and release/license terms before integrating; no download or local measurement yet.
- [D²-MoE](https://proceedings.mlr.press/v267/gu25c.html), ICML 2025, uses shared bases, SVD and structured pruning. It belongs in a separately labeled model-approximation track, not as evidence of bit-exact serialized reconstruction.

The implemented laboratory directly checks the simpler hypothesis first. Independent tiles include metadata and checksum costs. Raw is allowed to win. Actual samples and synthetic controls are distinguished in every row.

## Caching and prediction

[FlashMoE learned caching](https://arxiv.org/abs/2601.17063) is a different project from the Apple C/Metal repository. Submitted January 22, 2026 and revised October 5, it combines recency/frequency features for SSD offloading. Its results justify a comparison candidate, not an assumption that learning is required.

[SpecMD](https://arxiv.org/abs/2602.03921), submitted February 3, 2026, introduces Least-Stale and emphasizes interactions among cache/prefetch policies. A source-faithful reproduction is deferred; APEX's simple cost policy is not mislabeled Least-Stale.

The supplied MoE-SpeQ reference now resolves to [Specter](https://arxiv.org/abs/2511.14102), revision 3 dated October 22, 2026. It uses a quantized draft for multi-step speculative offloading. Acceptance correctness, draft memory, additional expert loads and transfer contention still require measurement on the target.

[Oracle-MoE](https://proceedings.mlr.press/v267/zhou25b.html), ICML 2025, changes routing architecture to encourage locality, with experiments on smaller GPT-2-based architectures. It is not a drop-in cache policy preserving the pretrained GPT-OSS computation.

## Immediate engineering conclusion

Prefer a measured baseline and small interfaces over a new general inference engine. The strongest first questions are whether an existing engine's cache is sufficient, whether CPU execution avoids costly transfers, and whether sampled compressed bytes save enough I/O to pay for decoding. Exact tensor-network or quantum-inspired work has no supporting local evidence yet and stays deferred.
