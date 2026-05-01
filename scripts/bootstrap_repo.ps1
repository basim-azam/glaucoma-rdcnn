# PowerShell hand-off: identical sequence to bootstrap_repo.sh.
# Run from the repo root in PowerShell:  .\scripts\bootstrap_repo.ps1
#
# Requires: git, gh (GitHub CLI) authenticated as basim-azam.

$ErrorActionPreference = "Stop"

$RepoName = "glaucoma-rdcnn"
$GhUser   = "basim-azam"
$GitName  = "Basim Azam"
$GitEmail = "basimazam0@gmail.com"

# --- Identity guard ---
if ($GitName -match "Claude|Anthropic|AI Assistant" -or $GitEmail -match "Claude|Anthropic") {
    throw "BAD AUTHOR -- refusing to run with Claude/AI identity."
}

git init -q
git config user.name  $GitName
git config user.email $GitEmail
git config commit.gpgsign false

git checkout -q -B main
git checkout -q -B dev

function Add-Commit {
    param([string]$Msg, [string[]]$Paths)
    git add @Paths
    git commit -q -m $Msg --author="$GitName <$GitEmail>"
}

Add-Commit "chore: initial project scaffold" @(
    "LICENSE","README.md",".gitignore","pyproject.toml","requirements.txt",
    "environment.yml","CITATION.cff",".pre-commit-config.yaml",".github/")

Add-Commit "feat(data): dataset download and preprocessing pipeline" @(
    "scripts/download_data.sh","scripts/preprocess.py",
    "src/glaucoma_rdcnn/__init__.py","src/glaucoma_rdcnn/data/")

Add-Commit "feat(models): ResNet34 backbone with DAC block" @(
    "src/glaucoma_rdcnn/models/__init__.py",
    "src/glaucoma_rdcnn/models/dac.py","src/glaucoma_rdcnn/models/backbone.py")

Add-Commit "feat(models): DPN and CPN with torchvision RPN" @(
    "src/glaucoma_rdcnn/models/dpn.py","src/glaucoma_rdcnn/models/cpn.py")

Add-Commit "feat(models): disc attention module + full R-DCNN assembly" @(
    "src/glaucoma_rdcnn/models/attention.py","src/glaucoma_rdcnn/models/rdcnn.py")

Add-Commit "feat(geometry): inscribed ellipse fitting and CDR computation" @(
    "src/glaucoma_rdcnn/geometry.py")

Add-Commit "feat(metrics): DC/JC/E/SE/SP and glaucoma AUC" @(
    "src/glaucoma_rdcnn/metrics.py","src/glaucoma_rdcnn/losses.py")

Add-Commit "feat(utils): logging, checkpointing, seed, viz" @(
    "src/glaucoma_rdcnn/utils/")

Add-Commit "feat(engine): DDP-aware trainer, evaluator, inference" @(
    "src/glaucoma_rdcnn/engine/","scripts/train.py","scripts/evaluate.py",
    "scripts/infer.py","scripts/make_figures.py")

Add-Commit "feat(slurm): Spartan job scripts and conda env loader" @("slurm/")
Add-Commit "feat(configs): Hydra config tree" @("configs/")
Add-Commit "test: smoke tests for models, data, geometry, metrics" @("tests/")
Add-Commit "docs: reproducing paper, Spartan guide, dataset schema" @("docs/")
Add-Commit "feat(notebooks): exploration, walkthrough, reproduction" @("notebooks/","assets/")

git add -A
if (-not (git diff --cached --quiet)) {
    git commit -q -m "chore: tidy" --author="$GitName <$GitEmail>"
}

Write-Host ""
Write-Host "=== Authors on dev ==="
git log --format='%h %an <%ae> %s'

$bad = git log --format='%an %ae' | Select-String -Pattern "claude|anthropic" -CaseSensitive:$false
if ($bad) { throw "REFUSE: found AI-attributed commit" }
Write-Host "OK: all commits attributed to $GitName <$GitEmail>"

Write-Host ""
Write-Host "=== Creating GitHub repo $GhUser/$RepoName ==="
gh auth status | Out-Null
gh repo create "$GhUser/$RepoName" `
    --public --source=. `
    --description="Replication of R-DCNN (Li et al., Eye 2023) for joint optic disc and cup segmentation" `
    --push=$false 2>$null

git remote remove origin 2>$null
git remote add origin "https://github.com/$GhUser/$RepoName.git"

git push -u origin dev
git checkout main
git merge --ff-only dev
git push -u origin main

gh repo edit "$GhUser/$RepoName" `
    --add-topic glaucoma --add-topic medical-imaging --add-topic pytorch `
    --add-topic segmentation --add-topic paper-replication `
    --add-topic unimelb-spartan --add-topic slurm --add-topic object-detection 2>$null

git tag -a v0.1.0 -m "Initial scaffold + reference implementation"
git push origin v0.1.0

Write-Host ""
Write-Host "OK Repo live at https://github.com/$GhUser/$RepoName"
