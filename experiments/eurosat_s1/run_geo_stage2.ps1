# run_geo_stage2.ps1 - stage 2 of the location-disjoint rerun. ASCII only.
#
# Stage 1 gate was met: add - none = +4.18 on the geo split, 2*SE = 1.10.
# The payload is real on a clean split, so routing is worth measuring here.
#
# Stage 2: token, gate, adaln, shuffle x 3 seeds = 12 runs, ~6 min each.

$ErrorActionPreference = 'Stop'
Set-Location "C:\Users\aliso\Desktop\big-files\loc"

$py = "C:\Users\aliso\Desktop\big-files\venv-gpu\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "interpreter not found: $py" }
& $py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA unavailable - aborting" }

$start = Get-Date
foreach ($arm in 'token','gate','adaln','shuffle') {
    foreach ($seed in 0,1,2) {
        $tag = "${arm}_s${seed}_geo"
        if (Test-Path "runs\$tag.json") { Write-Host "[skip] $tag"; continue }
        Write-Host "[run ] $tag   $(Get-Date -Format 'HH:mm:ss')"
        & $py -u 24_train.py --arm $arm --seed $seed --split geo `
            1> "_run_$tag.log" 2> "_run_$tag.err"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[FAIL] $tag exit $LASTEXITCODE"; Get-Content "_run_$tag.err" -Tail 5
        } elseif (-not (Test-Path "runs\$tag.json")) {
            Write-Host "[FAIL] $tag exited 0 but wrote no result file"
        } else { Write-Host "[ok  ] $tag" }
    }
}
Write-Host "[ALL DONE] $((Get-Date) - $start)"
