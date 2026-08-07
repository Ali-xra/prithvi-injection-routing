# ASCII ONLY.
# Six arms, five seeds each, on EuroSAT-S1.
#
#   powershell -ExecutionPolicy Bypass -File C:\Users\aliso\Desktop\big-files\loc\run_matrix.ps1
#
# The payload is identical in every arm: the same lon/lat for the same chip.
# Only the entry point changes, so any difference between arms is routing.
#
# Verified before spending a single GPU-minute:
#   tabular probe   image only 70.50 / image+location 87.02 / shuffled 69.98
#                   -> +16.52 points of real, non-redundant payload
#   23_test_arms.py 20/20 checks, including must-differ branches
#
# Seeds are the OUTER loop: after one pass we already have all six arms at
# seed 0, so a broken arm shows up in minutes rather than after the full sweep.

$B    = "C:\Users\aliso\Desktop\big-files"
$PY   = "$B\venv-gpu\Scripts\python.exe"
$SRC  = "$B\loc\24_train.py"
$RUNS = "$B\loc\runs"
$env:PYTHONIOENCODING = "utf-8"

$arms  = @("none","add","token","adaln","gate","shuffle")
$seeds = @(0,1,2,3,4)

Write-Host ""
Write-Host "=== EuroSAT-S1 injection matrix: 6 arms x 5 seeds ===" -ForegroundColor Cyan

foreach ($seed in $seeds) {
  foreach ($arm in $arms) {
    $tag  = "${arm}_s${seed}"
    $done = "$RUNS\$tag.json"
    if (Test-Path $done) {
      $d = Get-Content $done -Raw | ConvertFrom-Json
      Write-Host ("SKIP  {0,-12} {1:N2}" -f $tag, ($d.best_val_acc * 100)) -ForegroundColor DarkGray
      continue
    }
    $log = "$B\loc\_run_$tag.log"
    Write-Host ("RUN   {0,-12} {1}" -f $tag, (Get-Date -Format "HH:mm")) -ForegroundColor Green
    $t0 = Get-Date
    & $PY $SRC --arm $arm --seed $seed --epochs 30 1>$log 2>"$log.err"
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if (Test-Path $done) {
      $d = Get-Content $done -Raw | ConvertFrom-Json
      Write-Host ("DONE  {0,-12} val {1:N2}   {2} min" -f $tag, ($d.best_val_acc * 100), $mins) -ForegroundColor Green
    } else {
      Write-Host ("FAIL  {0} after {1} min - see {2}.err" -f $tag, $mins, $log) -ForegroundColor Red
      Get-Content "$log.err" -Tail 8 -ErrorAction SilentlyContinue
      Write-Host "stopping so the failure is not repeated 29 times" -ForegroundColor Red
      exit 1
    }
  }
}

Write-Host ""
Write-Host "=== summary ===" -ForegroundColor Cyan
foreach ($f in (Get-ChildItem "$RUNS\*.json" -ErrorAction SilentlyContinue | Sort-Object Name)) {
  $d = Get-Content $f.FullName -Raw | ConvertFrom-Json
  "{0,-12} val {1:N2}   inj {2,9}" -f $f.BaseName, ($d.best_val_acc * 100), $d.injection_params
}
