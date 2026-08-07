# run_stage7.ps1 - take the gate_late contrasts to n = 10. ASCII only.
#
# Read docs/PREREG-stage7-power.md BEFORE reading these numbers. Short version:
# at n = 5 every gate_late contrast fails the rule (p = 0.063 / 0.067 / 0.092).
# n = 5 had about 45% power for a 0.7-point effect at s = 0.56. n = 10 is the
# power calculation, not the p-value. FINAL n IS 10 - no seed 10 will be run
# whatever this returns.
#
# `add` is included so the headline contrast is 10-vs-10, not 10-vs-5.
# 33_analyse_geo.py is hard-limited to seeds 0-4 so these cannot leak into the
# pre-registered table.
#
# 15 runs, ~6 min each, ~95 min.

$ErrorActionPreference = 'Stop'
Set-Location "C:\Users\aliso\Desktop\big-files\loc"

$py = "C:\Users\aliso\Desktop\big-files\venv-gpu\Scripts\python.exe"
& $py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA unavailable - aborting" }
Write-Host "[env ] interpreter ok, CUDA ok"

$start = Get-Date
foreach ($seed in 5,6,7,8,9) {
    foreach ($arm in 'gate_late','add_mid','add') {
        $tag = "${arm}_s${seed}_geo"
        if (Test-Path "runs\$tag.json") { Write-Host "[skip] $tag"; continue }
        Write-Host "[run ] $tag   $(Get-Date -Format 'HH:mm:ss')"
        & $py -u 24_train.py --arm $arm --seed $seed --split geo `
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
& $py 33_analyse_geo.py  2>&1 | Out-File -Encoding utf8 "_stage7_prereg.log"
& $py 37_analyse_new_arms.py 2>&1 | Out-File -Encoding utf8 "_stage7_analysis.log"
Write-Host "[analysis -> _stage7_analysis.log]"
