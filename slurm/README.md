# Spartan SLURM scripts

Project: `punim2920` · Project root: `/data/gpfs/projects/punim2920/glaucoma-rdcnn`

Every job sources `slurm/env_spartan.sh` to load modules, activate the conda env in the project dir, and point all caches (torch, HF, pip, conda packages) at the project directory. Home is too small.

| Script | Partition | Time | GPUs | Purpose |
|---|---|---|---|---|
| `001_setup_env.slurm` | `sapphire` | 2h | — | one-time conda env build |
| `002_download_data.slurm` | `sapphire` | 2h | — | wraps `scripts/download_data.sh` + `scripts/preprocess.py` |
| `003_smoke_test.slurm` | `gpu-a100-short` | 30m | 1 A100 | CUDA check + pytest + `fast_dev_run` |
| `004_train_a100_1gpu.slurm` | `gpu-a100` | 24h | 1 A100 | default training |
| `005_train_a100_4gpu.slurm` | `gpu-a100` | 48h | 4 A100 | DDP, single node |
| `006_train_h100_4gpu.slurm` | `gpu-h100` | 72h | 4 H100 | when you can wait through the H100 queue |
| `007_train_array.slurm` | `gpu-a100` | 24h | 1 A100 each | hyperparam sweep via `slurm/sweeps/repro.csv` |
| `008_evaluate.slurm` | `gpu-a100-short` | 1h | 1 A100 | runs `scripts/evaluate.py` against a checkpoint |
| `interactive.sh` | `gpu-a100-short` | 4h | 1 A100 | wraps `sinteractive` |

## Common pitfalls

- **`__init__.py` everywhere.** Missing one in `utils/` cost a sibling project a failed job. Already covered in this repo.
- **Don't `pip install --user`.** Conflicts with conda env (e.g. stale `threadpoolctl` in `~/.local`).
- **CUDA wheel must match the loaded module.** We pin `torch 2.5.1+cu121` against `CUDA/12.1.1`. Don't mix.
- **Caches in project dir, not `$HOME`.** `env_spartan.sh` exports `TORCH_HOME`, `HF_HOME`, `PIP_CACHE_DIR`, `CONDA_PKGS_DIRS`.
- **Don't run on login nodes.** Use `sinteractive` for quick checks, `sbatch` for everything else.
- **Queue waits.** `gpu-a100-short` minutes-to-hours. `gpu-a100` 1–2 days for 2-GPU. `gpu-h100` often 6+ days. Plan accordingly.
