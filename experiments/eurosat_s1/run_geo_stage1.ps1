# run_geo_stage1.ps1 - stage 1 of the location-disjoint rerun.
# ASCII only: PowerShell 5.1 reads .ps1 as ANSI without a BOM and non-ASCII text
# breaks the parser (FAILURES.md).
#
# 2026-08-05 fix: the first attempt used the python on PATH, which is Python314
# and has no torch. Every run died instantly with ModuleNotFoundError, the error
# went into a redirect nobody read, and a night of "running" was nothing.
# Training needs the venv-gpu interpreter. Verified: torch 2.13.0+cu126, GTX 1070.
#
# Stage 1 gate: none x3 vs add x3 on the geo split. If add - none is below the
# detection threshold, the payload is empty on a clean split and the remaining
# arms are not worth spending.

$ErrorActionPreference = 'Stop'
Set-Location "C:\Users\aliso\Desktop\big-files\loc"

$py = "C:\Users\aliso\Desktop\big-files\venv-gpu\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "interpreter not found: $py" }

# must-differ style check: fail loudly now rather than silently later
& $py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "torch present but CUDA unavailable - aborting" }

$start = Get-Date
foreach ($arm in 'none','add') {
    foreach ($seed in 0,1,2) {
        $tag = "${arm}_s${seed}_geo"
        if (Test-Path "runs\$tag.json") { Write-Host "[skip] $tag"; continue }
        Write-Host "[run ] $tag   $(Get-Date -Format 'HH:mm:ss')   elapsed $((Get-Date) - $start)"
        & $py -u 24_train.py --arm $arm --seed $seed --split geo `
            1> "_run_$tag.log" 2> "_run_$tag.err"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[FAIL] $tag exit $LASTEXITCODE"
            Get-Content "_run_$tag.err" -Tail 5
        } elseif (-not (Test-Path "runs\$tag.json")) {
            Write-Host "[FAIL] $tag exited 0 but wrote no result file"
        } else {
            Write-Host "[ok  ] $tag"
        }
    }
}
Write-Host "[ALL DONE] $((Get-Date) - $start)"
