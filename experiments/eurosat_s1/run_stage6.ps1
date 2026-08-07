# run_stage6.ps1 - close the one comparison that fell just short. ASCII only.
#
# At 3 seeds: gate_late 78.22, add_mid 77.44, difference +0.78 against a
# threshold of 1.017 for a 3-vs-3 contrast. gate_late clears `add` and `gate`
# comfortably, but the contrast that actually isolates "reading later" from
# "injecting later" is the one against add_mid, and that one is short.
#
# Seeds 3 and 4 for both arms takes the contrast to 5-vs-5, where the threshold
# drops to 0.788. If the effect is real it will clear; if it is not, we will
# know that too.
#
# 4 runs, ~6 min each.

$ErrorActionPreference = 'Stop'
Set-Location "C:\Users\aliso\Desktop\big-files\loc"

$py = "C:\Users\aliso\Desktop\big-files\venv-gpu\Scripts\python.exe"
& $py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA unavailable - aborting" }
Write-Host "[env ] interpreter ok, CUDA ok"

$start = Get-Date
foreach ($arm in 'gate_late','add_mid') {
    foreach ($seed in 3,4) {
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
& $py 33_analyse_geo.py > "_stage6_analysis.log" 2>&1
& $py 37_analyse_new_arms.py >> "_stage6_analysis.log" 2>&1
Write-Host "[analysis -> _stage6_analysis.log]"
