# run_stage8_control.ps1 - strengthen the CONTROL, not the contrast. ASCII only.
#
# At n = 10 the headline contrast gate_late - add_mid = +0.82 clears the rule.
# The thing that certifies it is not capacity is gate_late + shuffled coordinates,
# and that control sits at n = 3. Two more seeds.
#
# This is NOT more seeds of the contrast - PREREG-stage7-power.md fixed that at
# n = 10 and it stays there. Adding seeds to a control can only make the claim
# harder to sustain, never easier, so it carries no optional-stopping risk.
#
# 2 runs, ~12 min.

$ErrorActionPreference = 'Stop'
Set-Location "C:\Users\aliso\Desktop\big-files\loc"

$py = "C:\Users\aliso\Desktop\big-files\venv-gpu\Scripts\python.exe"
& $py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA unavailable - aborting" }
Write-Host "[env ] interpreter ok, CUDA ok"

$start = Get-Date
foreach ($seed in 3,4) {
    $tag = "gate_late_s${seed}_geo_shufcoord"
    if (Test-Path "runs\$tag.json") { Write-Host "[skip] $tag"; continue }
    Write-Host "[run ] $tag   $(Get-Date -Format 'HH:mm:ss')"
    & $py -u 24_train.py --arm gate_late --seed $seed --split geo --shuffle-coords `
        1> "_run_$tag.log" 2> "_run_$tag.err"
    if (-not (Test-Path "runs\$tag.json")) {
        Write-Host "[FAIL] $tag"; Get-Content "_run_$tag.err" -Tail 8
    } else {
        $acc = (Get-Content "runs\$tag.json" | ConvertFrom-Json).best_val_acc
        Write-Host ("[ok  ] {0}   {1:N2}" -f $tag, ($acc * 100))
    }
}
Write-Host "`n[ALL DONE] $((Get-Date) - $start)"
& $py 37_analyse_new_arms.py 2>&1 | Out-File -Encoding utf8 "_stage8_analysis.log"
