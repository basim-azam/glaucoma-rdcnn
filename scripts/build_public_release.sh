#!/usr/bin/env bash
# Rebuild cd-former-public/ from the current state of glaucoma-rdcnn.
# Run from glaucoma-rdcnn root.
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=cd-former-public
if [[ -d "$TARGET" ]]; then
    echo "Target $TARGET exists; removing src + scripts + tests subtrees for clean copy"
    rm -rf "$TARGET/src/cd_former" "$TARGET/scripts" "$TARGET/tests"
fi
mkdir -p "$TARGET/src/cd_former/models" "$TARGET/scripts" "$TARGET/tests" "$TARGET/slurm"
# Source
for f in __init__.py data.py losses.py postproc.py evaluator.py trainer.py chromatic.py; do
    cp src/glaucoma_rdcnn_v2/$f "$TARGET/src/cd_former/$f"
done
for f in __init__.py backbone.py mask2former_head.py roi_localizer.py; do
    cp src/glaucoma_rdcnn_v2/models/$f "$TARGET/src/cd_former/models/$f"
done
# Scripts (rename)
cp scripts/train_v2.py        "$TARGET/scripts/train.py"
cp scripts/evaluate_v2.py     "$TARGET/scripts/evaluate.py"
cp scripts/postprocess_v2.py  "$TARGET/scripts/postprocess.py"
cp scripts/ensemble_v2.py     "$TARGET/scripts/ensemble.py"
cp scripts/make_qualitative_figure.py "$TARGET/scripts/make_qualitative_figure.py"
cp scripts/make_ablation_qualitative.py "$TARGET/scripts/make_ablation_qualitative.py"
cp scripts/plot_paper_figures.py "$TARGET/scripts/plot_paper_figures.py"
cp scripts/prepare_drishti_gs.py "$TARGET/scripts/prepare_drishti_gs.py"
cp scripts/prepare_rim_one.py    "$TARGET/scripts/prepare_rim_one.py"
cp scripts/prepare_refuge2.py    "$TARGET/scripts/prepare_refuge2.py"
cp scripts/prepare_smdg19.py     "$TARGET/scripts/prepare_smdg19.py"
[ -f scripts/preprocess.py ] && cp scripts/preprocess.py "$TARGET/scripts/preprocess.py" || true
# Tests
cp tests_v2/conftest.py            "$TARGET/tests/conftest.py"
cp tests_v2/test_03_seg_head.py    "$TARGET/tests/test_seg_head.py"
cp tests_v2/test_04_compound_loss.py "$TARGET/tests/test_compound_loss.py"
# Rename glaucoma_rdcnn_v2 → cd_former across copied files
find "$TARGET/src/cd_former" "$TARGET/scripts" "$TARGET/tests" -name '*.py' \
    -exec sed -i 's/glaucoma_rdcnn_v2/cd_former/g' {} +
# Sync the paper
cp -r paper/accv "$TARGET/paper/"
echo "Done — $TARGET is ready for git init && git remote add origin <new repo>"
