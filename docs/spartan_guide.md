# Spartan HPC guide

This repo is wired for the University of Melbourne **Spartan** cluster, project allocation `punim2920`.

## One-time setup

1. SSH in (SAML, UoM password):

   ```bash
   ssh spartan
   ```

2. Move to the project space and clone:

   ```bash
   cd /data/gpfs/projects/punim2920
   git clone https://github.com/basim-azam/glaucoma-rdcnn.git
   cd glaucoma-rdcnn
   ```

3. Add your HuggingFace token once. The export is **never echoed** by the env script and **never committed**:

   ```bash
   grep -q HF_TOKEN ~/.bashrc || echo 'export HF_TOKEN=<your-hf-token>' >> ~/.bashrc
   source ~/.bashrc
   ```

4. Build the conda env (runs on `sapphire`, ~30 minutes):

   ```bash
   mkdir -p logs
   sbatch slurm/001_setup_env.slurm
   squeue --me
   ```

5. (Optional) download data inside Spartan:

   ```bash
   sbatch slurm/002_download_data.slurm
   ```

## Smoke test

```bash
sbatch slurm/003_smoke_test.slurm
tail -f logs/rdcnn-smoke_*.out
```

You should see:

- `CUDA available: True`
- `NVIDIA A100-SXM4-80GB`
- `pytest` passing
- one fast-dev-run batch through the model

## Train

| Command | Partition | GPUs | Wall |
|---|---|---|---|
| `sbatch slurm/004_train_a100_1gpu.slurm` | `gpu-a100` | 1 | 24 h |
| `sbatch slurm/005_train_a100_4gpu.slurm` | `gpu-a100` | 4 (DDP) | 48 h |
| `sbatch slurm/006_train_h100_4gpu.slurm` | `gpu-h100` | 4 (DDP) | 72 h, but expect a long queue |
| `sbatch slurm/007_train_array.slurm` | `gpu-a100` | 1 each | sweep, 6 rows |

Monitor:

```bash
squeue --me
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode
tail -f logs/rdcnn-train-*.out
```

Pull artifacts back from your laptop:

```bash
rsync -av spartan:/data/gpfs/projects/punim2920/glaucoma-rdcnn/outputs/ ./outputs/
wandb sync outputs/wandb/offline-run-*
```

## Storage layout

| Path | Use |
|---|---|
| `/data/gpfs/projects/punim2920/glaucoma-rdcnn/conda_envs/glaucoma` | conda env (built once) |
| `/data/gpfs/projects/punim2920/glaucoma-rdcnn/data/...` | raw + preprocessed data |
| `/data/gpfs/projects/punim2920/glaucoma-rdcnn/outputs/...` | run dirs, checkpoints, wandb offline runs |
| `/data/gpfs/projects/punim2920/glaucoma-rdcnn/.cache/{torch,huggingface,pip,conda_pkgs}` | caches |
| `/data/gpfs/projects/punim2920/glaucoma-rdcnn/logs/` | SLURM .out and .err |
| `$HOME` | **DO NOT** install anything heavy here — quota is small |

## Partition cheat sheet

| Partition | Max time | GPUs/node | Use for |
|---|---|---|---|
| `gpu-a100-short` | 4h | 4×80GB A100 | smoke tests, fast turnaround |
| `gpu-a100` | 7d | 4×80GB A100 | default training |
| `gpu-h100` | 7d | 4×80GB H100 | faster, but 6+ day queue is common |
| `gpu-l40s` | 7d | 4×48GB L40S | fallback when A100 is queued |
| `sapphire` | 2d | CPU only | env setup, downloads, preprocessing |

Check live availability:

```bash
sinfo -O cpusstate -p gpu-a100
sinfo -O cpusstate -p gpu-l40s
```

## Common pitfalls (from sibling project's job log)

- **Missing `__init__.py`.** Cost the sibling project a failed job. Already covered in this repo.
- **`pip install --user`** conflicts with the conda env (e.g., stale `threadpoolctl` in `~/.local`). Don't.
- **CUDA wheel mismatch.** `CUDA/12.1.1` module + `torch ... cu118` wheel = cryptic runtime errors. Match them; this repo pins `cu121`.
- **Caches in `$HOME`.** Home quota is small; `env_spartan.sh` redirects `TORCH_HOME`, `HF_HOME`, `PIP_CACHE_DIR`, `CONDA_PKGS_DIRS` into the project dir.
- **Running on the login node.** Don't. Use `sinteractive` (4h via `slurm/interactive.sh`) or `sbatch`.
- **Long H100 queues.** Don't block on `gpu-h100` — submit, do something else.

## When module versions differ

If `module avail Anaconda3` doesn't show `Anaconda3/2024.02-1`, edit `slurm/env_spartan.sh` to whichever version is listed and note the change in `docs/spartan_environment_<date>.log` (created by Phase 1 of the original setup prompt).

### Verified module versions (2026-05-02)

CUDA modules currently on Spartan: `11.5.2`, `11.7.0`, `11.8.0`, `12.2.0`, `12.4.1`, `12.5.1` (default).

This repo loads **`CUDA/12.2.0` + `cuDNN/8.9.7.29-CUDA-12.2.0`** because:
- The conda env is built against the **`cu121`** PyTorch wheel (`torch==2.5.1+cu121`).
- CUDA 12.2 is forward-compatible with cu121 binaries (PyTorch wheels bundle their own `libcudart`, so the system CUDA module mainly provides driver-side bits and `nvcc`).
- CUDA 12.4 / 12.5 also work; we pick 12.2 for the cuDNN minor-version match.
