# WBCDiffusion

Research repository for high-quality white blood cell (WBC) image generation.

## Research plan

The project roadmap, research practices, Phase 0 checklist, and mandatory experiment record are documented in [Research Plan v2.0](docs/research_plan_v2.md). Cluster SSD/HDD user paths remain pending confirmation from the NTU EEE Cluster allocation; do not infer them from mount names.

## Current state

- The repository currently contains small DDPM forward-process and toy-model learning scripts at its root.
- The formal project structure and experiment-recording workflow are being established.
- The WBC dataset location, cluster storage allocation, GPU access, and formal training environment have not yet been verified here.

## Getting started

1. Complete the storage and environment checks in the research plan before moving or deleting environments or data.
2. Use `experiments/experiment_template.md` for each formal experiment. Assign a unique Experiment ID and preserve unsuccessful runs.
3. Keep raw data unchanged. Separate processed data, metadata, checkpoints, logs, and generated results.
4. Record the exact configuration, seed, software versions, hardware, and command needed to reproduce each result.

## Repository map

See [the research plan](docs/research_plan_v2.md#4-project-structure) for the intended phase and data layout. Large datasets and checkpoints belong only in cluster locations whose allocation has been confirmed.
