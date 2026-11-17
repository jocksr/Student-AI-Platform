# Initial experiment log

Date: November 17, 2026, America/Chicago. Project: `E:\Project_APEX`.

Organizational correction, November 17, 2026: APEX is experiment 001 within Student AI Platform. Its files now live at `E:\Project_APEX\experiments\001-apex-inference`. Original measurement JSON and hashes were preserved; their original paths document where the measurements were taken. The repository-level README covers the broader project, and this experiment retains its own README, dependencies, implementation, tests, data and findings. The existing ignored virtual environment remains at repository root and the runner also supports a private experiment environment. Post-move correctness results are saved separately in `results/tests-layout-check.json`.

## Scope and method

The first deliverable is research infrastructure. The user explicitly reinforced that the design must remain open to disproof and alternatives. No TinyFish calls were made; research used standard web tools after the user's instruction. No model inference or paid service was used.

Hardware discovery used read-only CIM, nvidia-smi, Python and tool inventory. CIM/WSL required running outside the restricted sandbox. A local virtual environment reuses existing scientific packages; a 101 kB LZ4 wheel was added. The local editable package needed an out-of-sandbox install because setuptools interpreted braces in the sandbox temporary directory as template variables. No system Python packages were changed.

The NVIDIA driver was sufficient for real CUDA transfers and a reference PTX projection kernel; a large CUDA/PyTorch installation was unnecessary for this milestone. MSVC exists outside PATH. WSL and Docker availability were inventoried without starting their services.

## Data provenance

- `results/hardware-windows.json`: CPU, DIMMs, disk/volume mapping, tool paths, WSL and nvidia-smi snapshot.
- `results/hardware.json`: available RAM, CPU instruction flags, driver output and software versions.
- `results/memory.json`, `storage.json`, `gpu.json`: actual microbenchmark observations and raw timing samples.
- `results/model-metadata/`: official config/index, all tensor headers, pinned revision and derived byte counts. Metadata response bodies totaled 157,698 bytes in the initial fetch.
- `benchmarks/weight-samples/manifest.json`: 24 exact byte ranges totaling 3,145,728 bytes from the official checkpoint. Three layers, two experts, packed/scales for both projections. Sampling is small and nonrandom.
- `results/compression.json`: 234 corpus/tile/codec/transform comparisons, each with exact reconstruction checks.
- `results/cross-expert.json`: 12 sampled pairs, exact-block comparisons and reversible XOR residual costs with the base included.
- `results/comparison.json`: three seeded traces for each of three workload families, four policies, plain/fixed-codec/adaptive representation scenarios. No real routing data.
- `results/tests.json`: test count and pass/failure summary; source digest identifies the implementation tested.

## Observations and interpretation

Early hardware run: RAM copy payload approximately 17.0 GB/s; OS-cache-bypassing sequential SSD approximately 4.0 GB/s; random 1 MiB SSD approximately 3.4 GB/s; pinned H2D approximately 13.1 GB/s and D2H approximately 14.1 GB/s. These values vary with desktop load; the latest detailed run lives in the JSON files and generated report.

At 64 KiB, sampled packed data showed only about 3.5% size reduction with Zstandard, while sampled scales shrank roughly 78%. The latter cannot be applied to the much larger packed-data stream. LZ4 slightly expanded packed bytes. Decoding cost made raw tiles the initial access-cost choice. This is a negative result for the current CPU codec path, not proof that all lossless compression is useless.

None of the sampled cross-expert pairs shared an exact 4 KiB block. Storing a Zstandard base plus an XOR residual increased size in all 12 pairs versus independent compression. This discourages prioritizing shared bases until broader tests give contrary evidence.

Stable hot sets favored frequency retention in the synthetic runs. After a hot-set change, LRU adapted while cumulative LFU retained stale experts. Cost-aware retention was equivalent to LFU under uniform expert sizes/costs. Zero GPU LRU hits at the chosen small cache budget are possible because a full layer sweep exceeds its capacity; they are not an inference measurement.

## Open limitations

Only E:'s Samsung NVMe was benchmarked; the C: Kingston and USB HDD were inventoried, not stress-tested. Storage queue depth is one and the device cache is not disabled. DIMM configuration does not independently prove dual-channel behavior; 51.2 GB/s would be the nominal aggregate for two effective 64-bit DDR4-3200 channels, not a measured sustained rate. NumPy logical read-plus-write traffic is not a DRAM bus counter.

The GPU kernel uses FP32 and ordinary SIMT instructions. It tests numerical agreement and the measurement path, not MXFP4 Tensor Core performance. The venv is not hermetic. Runtime memory budgets, filesystem-cache interaction, full expert decode time and inference correctness remain unvalidated.

The final pipeline is rerun after accounting and CLI integration checks. Tests cover byte equality, malformed payloads, independent tile reads, randomized cache occupancy, hand-calculated cost paths and a subprocess CLI trace/replay flow. No large model checkpoint, backend inference engine or external research code was installed.
