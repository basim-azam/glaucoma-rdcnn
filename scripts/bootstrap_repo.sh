#!/usr/bin/env bash
# One-shot: git init → conventional-commit sequence (authored as Basim) → gh repo create → push.
#
# Run from the repo root. Requires: git, gh CLI authenticated as basim-azam.
#
# All commits are authored as "Basim Azam <basimazam0@gmail.com>". This script
# refuses to run if git config has any "Claude" / "Anthropic" / "AI" string in
# user.name or user.email.
#
# Usage:
#   cd /path/to/glaucoma-rdcnn
#   bash scripts/bootstrap_repo.sh

set -euo pipefail

REPO_NAME=${REPO_NAME:-glaucoma-rdcnn}
GH_USER=${GH_USER:-basim-azam}
GIT_NAME=${GIT_NAME:-"Basim Azam"}
GIT_EMAIL=${GIT_EMAIL:-basimazam0@gmail.com}

# --- Identity guard ---------------------------------------------------------
case " $GIT_NAME $GIT_EMAIL " in
    *Claude*|*claude*|*Anthropic*|*anthropic*|*"AI Assistant"*)
        echo "BAD AUTHOR — refusing to run with Claude/AI identity." >&2
        exit 1 ;;
esac

git init -q
git config user.name  "$GIT_NAME"
git config user.email "$GIT_EMAIL"
git config commit.gpgsign false 2>/dev/null || true

# --- Branching --------------------------------------------------------------
git checkout -q -B main
git checkout -q -B dev

# --- Conventional commit sequence ------------------------------------------
# Every commit is a small, focused snapshot of one piece of the build.
add_and_commit() {
    local msg="$1"; shift
    git add "$@"
    git commit -q -m "$msg" --author="$GIT_NAME <$GIT_EMAIL>"
}

add_and_commit "chore: initial project scaffold" \
    LICENSE README.md .gitignore pyproject.toml requirements.txt environment.yml \
    CITATION.cff .pre-commit-config.yaml .github/

add_and_commit "feat(data): dataset download and preprocessing pipeline" \
    scripts/download_data.sh scripts/preprocess.py \
    src/glaucoma_rdcnn/__init__.py \
    src/glaucoma_rdcnn/data/

add_and_commit "feat(models): ResNet34 backbone with DAC block" \
    src/glaucoma_rdcnn/models/__init__.py \
    src/glaucoma_rdcnn/models/dac.py \
    src/glaucoma_rdcnn/models/backbone.py

add_and_commit "feat(models): DPN and CPN with torchvision RPN" \
    src/glaucoma_rdcnn/models/dpn.py \
    src/glaucoma_rdcnn/models/cpn.py

add_and_commit "feat(models): disc attention module + full R-DCNN assembly" \
    src/glaucoma_rdcnn/models/attention.py \
    src/glaucoma_rdcnn/models/rdcnn.py

add_and_commit "feat(geometry): inscribed ellipse fitting and CDR computation" \
    src/glaucoma_rdcnn/geometry.py

add_and_commit "feat(metrics): DC/JC/E/SE/SP and glaucoma AUC" \
    src/glaucoma_rdcnn/metrics.py \
    src/glaucoma_rdcnn/losses.py

add_and_commit "feat(utils): logging, checkpointing, seed, viz" \
    src/glaucoma_rdcnn/utils/

add_and_commit "feat(engine): DDP-aware trainer, evaluator, inference" \
    src/glaucoma_rdcnn/engine/ \
    scripts/train.py scripts/evaluate.py scripts/infer.py scripts/make_figures.py

add_and_commit "feat(slurm): Spartan job scripts and conda env loader" \
    slurm/

add_and_commit "feat(configs): Hydra config tree" \
    configs/

add_and_commit "test: smoke tests for models, data, geometry, metrics" \
    tests/

add_and_commit "docs: reproducing paper, Spartan guide, dataset schema" \
    docs/

add_and_commit "feat(notebooks): exploration, walkthrough, reproduction" \
    notebooks/ assets/

# Catch any stragglers
git add -A
if ! git diff --cached --quiet; then
    git commit -q -m "chore: tidy" --author="$GIT_NAME <$GIT_EMAIL>"
fi

# --- Identity audit --------------------------------------------------------
echo
echo "=== Authors of every commit on dev ==="
git log --format='%h %an <%ae> %s'
if git log --format='%an %ae' | grep -iE 'claude|anthropic|^ai '; then
    echo "REFUSE: found AI-attributed commit" >&2
    exit 1
fi
echo "✓ all commits attributed to $GIT_NAME <$GIT_EMAIL>"

# --- GitHub repo + push ----------------------------------------------------
echo
echo "=== Creating GitHub repo $GH_USER/$REPO_NAME ==="
if ! gh auth status >/dev/null 2>&1; then
    echo "gh CLI not authenticated. Run: gh auth login"
    exit 2
fi

gh repo create "$GH_USER/$REPO_NAME" \
    --public \
    --source=. \
    --description="Replication of R-DCNN (Li et al., Eye 2023) for joint optic disc and cup segmentation" \
    --push=false || echo "(repo may already exist)"

git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$GH_USER/$REPO_NAME.git"

git push -u origin dev
git checkout main
git merge --ff-only dev
git push -u origin main

# --- Topics + tag ----------------------------------------------------------
gh repo edit "$GH_USER/$REPO_NAME" \
    --add-topic glaucoma \
    --add-topic medical-imaging \
    --add-topic pytorch \
    --add-topic segmentation \
    --add-topic paper-replication \
    --add-topic unimelb-spartan \
    --add-topic slurm \
    --add-topic object-detection 2>/dev/null || true

git tag -a v0.1.0 -m "Initial scaffold + reference implementation"
git push origin v0.1.0

echo
echo "✓ Repo live at https://github.com/$GH_USER/$REPO_NAME"
