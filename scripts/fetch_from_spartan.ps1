# Pull CD-Former results from Spartan via scp on Windows.
# Requires OpenSSH client (built into Windows 10+).
#
# Usage:
#   .\scripts\fetch_from_spartan.ps1
#

param([string]$Pattern = "")

$SpartanRepo = "/data/gpfs/projects/punim2920/glaucoma-rdcnn"

Write-Host "=== Fetching summary JSONs ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "outputs/_fetched_from_spartan" | Out-Null
# Use scp with a glob pattern; PowerShell doesn't have rsync natively
scp "spartan:${SpartanRepo}/outputs/v2_${Pattern}*/summary.json" "outputs/_fetched_from_spartan/" 2>$null
scp "spartan:${SpartanRepo}/outputs/v2_${Pattern}*/postproc_test.json" "outputs/_fetched_from_spartan/" 2>$null

Write-Host ""
Write-Host "=== Fetching test data (DRISHTI + RIM-ONE) ===" -ForegroundColor Cyan
foreach ($ds in @("drishti_gs", "rim_one_v3")) {
    New-Item -ItemType Directory -Force -Path "data/$ds/images" | Out-Null
    New-Item -ItemType Directory -Force -Path "data/$ds/masks/od" | Out-Null
    New-Item -ItemType Directory -Force -Path "data/$ds/masks/oc" | Out-Null
    New-Item -ItemType Directory -Force -Path "data/$ds/splits" | Out-Null
    scp -r "spartan:${SpartanRepo}/data/${ds}/images/*"     "data/${ds}/images/"
    scp -r "spartan:${SpartanRepo}/data/${ds}/masks/od/*"   "data/${ds}/masks/od/"
    scp -r "spartan:${SpartanRepo}/data/${ds}/masks/oc/*"   "data/${ds}/masks/oc/"
    scp -r "spartan:${SpartanRepo}/data/${ds}/splits/*"     "data/${ds}/splits/"
}

Write-Host ""
Write-Host "=== Fetching best checkpoints ===" -ForegroundColor Cyan
$drishtiBest = "v2_drishti_seed43_20260513"
$rimoneBest  = "v2_rimone_seed42_20260513"
scp -r "spartan:${SpartanRepo}/outputs/${drishtiBest}*" "outputs/"
scp -r "spartan:${SpartanRepo}/outputs/${rimoneBest}*"  "outputs/"

Write-Host ""
Write-Host "Done. Next: python scripts/plot_paper_figures.py" -ForegroundColor Green
