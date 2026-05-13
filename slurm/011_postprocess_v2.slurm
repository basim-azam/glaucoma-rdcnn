#!/bin/bash
#SBATCH --job-name=rdcnn-v2-postproc
#SBATCH --partition=gpu-a100-short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
#
# Post-process all v2 checkpoints on both datasets.
#
# Usage:
#   sbatch slurm/011_postprocess_v2.slurm                          # all v2 checkpoints
#   sbatch slurm/011_postprocess_v2.slurm v2_drishti_seed42        # specific run
#

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

PATTERN="${1:-*}"

echo "=== finding v2 checkpoints matching pattern: $PATTERN ==="
mapfile -t CKPTS < <(ls -d outputs/v2_drishti_${PATTERN}*/best.ckpt outputs/v2_rimone_${PATTERN}*/best.ckpt 2>/dev/null || true)
echo "found ${#CKPTS[@]} checkpoints:"
for c in "${CKPTS[@]}"; do echo "  $c"; done
echo ""

if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "!! no v2 checkpoints found"
  exit 1
fi

for ckpt in "${CKPTS[@]}"; do
  run_dir=$(dirname "$ckpt")
  if [[ "$run_dir" == *drishti* ]]; then
    DATA_ROOT="data/drishti_gs"
  else
    DATA_ROOT="data/rim_one_v3"
  fi
  echo ""
  echo "=== $run_dir ==="
  srun $PY scripts/postprocess_v2.py \
      --checkpoint "$ckpt" \
      --data-root "$DATA_ROOT" \
      --split test \
      --output-json "$run_dir/postproc_test.json"
done

echo ""
echo "=== aggregate across all post-processed runs ==="
$PY - << 'PY'
import json, glob, statistics
from collections import defaultdict

# Group by (dataset, recipe)
by_dataset_recipe = defaultdict(lambda: defaultdict(list))
for p in sorted(glob.glob("outputs/v2_*/postproc_test.json")):
    ds = "drishti" if "drishti" in p else "rimone"
    data = json.load(open(p))
    for recipe, metrics in data["results"].items():
        by_dataset_recipe[ds][recipe].append(metrics)

def ms(xs): return f"{100*statistics.mean(xs):.2f} +/- {100*statistics.stdev(xs):.2f}" if len(xs) >= 2 else (f"{100*xs[0]:.2f}" if xs else "-")
def ma(xs): return f"{statistics.mean(xs):.3f} +/- {statistics.stdev(xs):.3f}" if len(xs) >= 2 else (f"{xs[0]:.3f}" if xs else "-")

for ds in ("drishti", "rimone"):
    if ds not in by_dataset_recipe:
        continue
    print(f"\n[{ds}] post-processing recipes (n_runs varies per recipe)")
    print(f"  {'recipe':22s} {'OD Dice':>16s}  {'OC Dice':>16s}  {'AUC':>16s}  {'CDR MAE':>16s}")
    for recipe, run_metrics in by_dataset_recipe[ds].items():
        od = [m["od_dice"] for m in run_metrics]
        oc = [m["oc_dice"] for m in run_metrics]
        auc = [m["glaucoma_auc"] for m in run_metrics]
        cdr = [m["cdr_mae"] for m in run_metrics]
        print(f"  {recipe:22s} {ms(od):>16s}  {ms(oc):>16s}  {ma(auc):>16s}  {ma(cdr):>16s}")
PY

echo ""
echo "=== done ==="
