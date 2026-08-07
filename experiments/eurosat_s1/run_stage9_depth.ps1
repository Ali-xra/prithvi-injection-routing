# run_stage9_depth.ps1 - the depth sweep. ASCII only.
#
# Read docs/PREREG-stage9-depth.md BEFORE reading any number out of this.
# Short version: MID_AT=3 was chosen, never swept, and that is objection 5 in
# Measurement 22. The model has 6 blocks, so 3 is the middle.
#
# This is a SHAPE question, not a "which depth wins" question. Depth 3 stays the
# headline whatever this shows. n = 5 per point, fixed, no extension.
#
# 30 runs, ~6 min each, ~3 hours.

$ErrorActionPreference = 'Stop'
Set-Location "C:\Users\aliso\Desktop\big-files\loc"

$py = "C:\Users\aliso\Desktop\big-files\venv-gpu\Scripts\python.exe"
& $py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA unavailable - aborting" }

# the gate: T8 catches a silently ignored --mid-at, which would render as a
# perfectly flat profile. Never spend 3 GPU-hours before this passes.
& $py 23_test_arms.py > "_stage9_tests.log" 2>&1
if ($LASTEXITCODE -ne 0) { Get-Content "_stage9_tests.log" -Tail 20; throw "arm tests FAILED - aborting before any training" }
Write-Host "[env ] interpreter ok, CUDA ok, arm tests pass"

# depth 3 already has n=10 for both arms and is deliberately absent here.
$jobs = @()
foreach ($d in 1,2,4,6) { $jobs += ,@('gate_late', $d) }
foreach ($d in 1,6)     { $jobs += ,@('add_mid',   $d) }

$start = Get-Date
foreach ($seed in 0,1,2,3,4) {
    foreach ($j in $jobs) {
        $arm = $j[0]; $d = $j[1]
        $tag = "${arm}_s${seed}_geo_d${d}"
        if (Test-Path "runs\$tag.json") { Write-Host "[skip] $tag"; continue }
        Write-Host "[run ] $tag   $(Get-Date -Format 'HH:mm:ss')"
        & $py -u 24_train.py --arm $arm --seed $seed --split geo --mid-at $d `
            1> "_run_$tag.log" 2> "_run_$tag.err"
        if (-not (Test-Path "runs\$tag.json")) {
            Write-Host "[FAIL] $tag"; Get-Content "_run_$tag.err" -Tail 8
        } else {
            $acc = (Get-Content "runs\$tag.json" | ConvertFrom-Json).best_val_acc
            Write-Host ("[ok  ] {0}   {1:N2}" -f $tag, ($acc * 100))
        }
    }
}
Write-Host "`n[ALL DONE] $((Get-Date) - $start)"
& $py 38_analyse_depth.py 2>&1 | Out-File -Encoding utf8 "_stage9_analysis.log"
Write-Host "[analysis -> _stage9_analysis.log]"
