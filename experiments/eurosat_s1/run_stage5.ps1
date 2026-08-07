# run_stage5.ps1 - the follow-up experiments from Measurement 19. ASCII only.
#
# Six new arms plus three shuffled-coordinate controls, all on the geo split,
# 3 seeds each. 27 runs, ~6 min each, ~2.7 hours.
#
#   add_mid     control for gate_late - same mid injection, constant scalar
#   gate_late   gate read from the CLS token after 3 blocks, injected there
#   gate_std    gate input = [mean, std] of patch tokens (same entry point)
#   gate_max    gate input = max over patch tokens (same entry point)
#   gate_coord  gate fed by the coordinates instead of the image (exploratory)
#   film        per-dimension shift+scale from location, applied once
#
# Controls with --shuffle-coords: film, gate_coord, gate_late.
#
# Order is deliberate: the arms that answer the headline question run first, so
# that if the night is cut short the important ones are done.

$ErrorActionPreference = 'Stop'
Set-Location "C:\Users\aliso\Desktop\big-files\loc"

$py = "C:\Users\aliso\Desktop\big-files\venv-gpu\Scripts\python.exe"

# --- environment guards: the most expensive bug of this project was 15 hours
#     of "running" that was actually ModuleNotFoundError in an unread file ---
& $py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA unavailable - aborting" }
Write-Host "[env ] interpreter ok, CUDA ok"

& $py 23_test_arms.py > "_stage5_tests.log" 2>&1
if ($LASTEXITCODE -ne 0) { throw "23_test_arms.py FAILED - aborting before any training" }
Write-Host "[test] all arm tests pass"

$start = Get-Date

function Run-Arm($arm, $seed, $shuf) {
    $tag = "${arm}_s${seed}_geo"
    $extra = @()
    if ($shuf) { $tag = "${tag}_shufcoord"; $extra = @('--shuffle-coords') }
    if (Test-Path "runs\$tag.json") { Write-Host "[skip] $tag"; return }
    Write-Host "[run ] $tag   $(Get-Date -Format 'HH:mm:ss')"
    & $py -u 24_train.py --arm $arm --seed $seed --split geo @extra `
        1> "_run_$tag.log" 2> "_run_$tag.err"
    if (-not (Test-Path "runs\$tag.json")) {
        Write-Host "[FAIL] $tag"
        Get-Content "_run_$tag.err" -Tail 8
    } else {
        $acc = (Get-Content "runs\$tag.json" | ConvertFrom-Json).best_val_acc
        Write-Host ("[ok  ] {0}   {1:N2}" -f $tag, ($acc * 100))
    }
}

# ---- block 1: the headline pair (gate_late and its control) ----
Write-Host "`n=== block 1/4 : gate_late + add_mid ==="
foreach ($seed in 0,1,2) { Run-Arm 'gate_late' $seed $false }
foreach ($seed in 0,1,2) { Run-Arm 'add_mid'   $seed $false }

# ---- block 2: the missing rung ----
Write-Host "`n=== block 2/4 : film + its control ==="
foreach ($seed in 0,1,2) { Run-Arm 'film' $seed $false }
foreach ($seed in 0,1,2) { Run-Arm 'film' $seed $true }

# ---- block 3: alternative gate inputs ----
Write-Host "`n=== block 3/4 : gate_std + gate_max ==="
foreach ($seed in 0,1,2) { Run-Arm 'gate_std' $seed $false }
foreach ($seed in 0,1,2) { Run-Arm 'gate_max' $seed $false }

# ---- block 4: exploratory + remaining controls ----
Write-Host "`n=== block 4/4 : gate_coord + controls ==="
foreach ($seed in 0,1,2) { Run-Arm 'gate_coord' $seed $false }
foreach ($seed in 0,1,2) { Run-Arm 'gate_coord' $seed $true }
foreach ($seed in 0,1,2) { Run-Arm 'gate_late'  $seed $true }

Write-Host "`n[ALL DONE] $((Get-Date) - $start)"
& $py 33_analyse_geo.py > "_stage5_analysis.log" 2>&1
Write-Host "[analysis written to _stage5_analysis.log]"
