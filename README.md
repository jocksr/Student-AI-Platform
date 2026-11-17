# Student AI Platform

A research and development project exploring how students can access useful AI tools on affordable hardware without depending on paid subscriptions or APIs.

This repository holds **multiple independent experiments and, eventually, validated platform components**. No single experiment defines the platform's architecture. Approaches may be revised, combined, replaced, or rejected as evidence develops.

The project is independently developed. It does not claim college sponsorship, institutional authorization, or access to institutional resources.

## Current experiments

| ID | Experiment | Status | Entry point |
|---|---|---|---|
| 001 | APEX: memory-constrained MoE inference research | Working measurement, simulation, and lossless-compression harness; real-model inference remains unproven | [Experiment README](experiments/001-apex-inference/README.md) |

**APEX is one experiment within Student AI Platform.** Its primary architectural target is GPT-OSS-120B, but its hypotheses are not predetermined conclusions or requirements for every future platform component.

## Repository layout

```text
Student-AI-Platform/
├── README.md                         # Overall project scope and navigation
├── .gitignore                        # Shared exclusions
└── experiments/
    ├── README.md                     # Experiment index and organization
    └── 001-apex-inference/
        ├── README.md                 # Experiment scope, setup, and commands
        ├── ARCHITECTURE.md
        ├── RESEARCH.md
        ├── ROADMAP.md
        ├── EXPERIMENT_LOG.md
        ├── pyproject.toml
        ├── requirements-measured.txt
        ├── apex/                     # Experiment implementation
        ├── scripts/                  # Experiment runners and bounded data retrieval
        ├── tests/
        ├── benchmarks/               # Sample manifests and local fixtures
        └── results/                  # Measurements, simulations, and reports
```

The current local checkout is `E:\Project_APEX`; the GitHub repository is [jocksr/Student-AI-Platform](https://github.com/jocksr/Student-AI-Platform). The local folder name is historical and does not define the scope of the platform.

## Start with the existing experiment

Read the [APEX experiment README](experiments/001-apex-inference/README.md), [experimental report](experiments/001-apex-inference/results/initial-report.md), and [roadmap](experiments/001-apex-inference/ROADMAP.md).

On the current configured machine, run the experiment from the repository root:

```powershell
.\experiments\001-apex-inference\scripts\run_initial.ps1
```

The runner uses an experiment-local `.venv` if present, otherwise the existing repository-local `.venv`. Environments are ignored by Git. Setup on a new machine is documented inside the experiment; cloning this repository does not install dependencies or retrieve model samples automatically.

The full runner repeats benchmarks and replaces the current generated result files. To run only the correctness suite without rerunning hardware experiments:

```powershell
.\.venv\Scripts\python.exe experiments/001-apex-inference/scripts/run_tests.py --out results/tests-layout-check.json
```

The test runner resolves paths relative to the experiment. Its output above is saved inside the experiment's `results/` directory, preserving the original benchmark records.

## Research principles

- Measure actual behavior and preserve enough provenance to reproduce results.
- Distinguish hardware observations, synthetic simulations, sampled-data findings, and real inference results.
- Compare against existing software and simple baselines before adding complexity.
- Treat negative results as useful evidence and keep architecture decisions open.
- Preserve numerical correctness and evaluate practical task quality, not throughput alone.
- Keep each experiment's implementation, dependencies, data, tests, and findings within its own folder.
- Do not commit credentials, virtual environments, model checkpoints, or large generated fixtures.

No public student service or production inference platform has been deployed. Future experiments and any shared components should be added as their scope becomes concrete, rather than represented by empty placeholder modules.
