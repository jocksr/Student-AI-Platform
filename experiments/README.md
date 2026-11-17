# Experiments

Each directory contains one bounded research effort within Student AI Platform. An experiment can support, change, or disprove a design hypothesis; it does not establish the overall platform architecture.

| ID | Directory | Research question | Current evidence |
|---|---|---|---|
| 001 | [APEX inference](001-apex-inference/README.md) | Can expert-aware memory management, lossless compression, and CPU/GPU placement make very large MoE models more usable on modest hardware? | Local microbenchmarks, exact sample reconstruction, and synthetic cache simulations; no full-model inference result |

Use stable numbered directories such as `001-apex-inference`. Each experiment should have its own `README.md` with its hypothesis, scope, setup, reproduction commands, results, limitations, and next decision. Keep experiment-specific source, dependencies, tests, sample manifests, measurements, and notes in that directory. Label synthetic data and retain negative findings.

Shared platform code belongs outside an experiment only once a real shared use has been established. Historical result records should retain the original paths and software provenance from the time of measurement, even if the experiment is reorganized later.

Return to the [Student AI Platform README](../README.md).
