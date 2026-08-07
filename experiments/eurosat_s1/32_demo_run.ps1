# 32_demo_run.ps1 - the whole pipeline end to end in about five minutes. ASCII only.
#
# Purpose: run every moving part once, at a size where you can watch it, so the
# code you have read on paper is code you have seen execute. The accuracies are
# meaningless at 2 epochs - that is fine and deliberate. What you are checking is
# that each stage does what the walkthrough says it does.
#
#   step 1  23_test_arms.py    the must-differ test suite  (CPU, ~1 min)
#   step 2  six arms x 2 epochs                            (GPU, ~3 min)
#
# Results go to runs\_demo\ so they can never contaminate the real numbers.

$ErrorActionPreference = 'Stop'
Set-Location "C:\Users\aliso\Desktop\big-files\loc"

$py = "C:\Users\aliso\Desktop\big-files\venv-gpu\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path "runs\_demo" | Out-Null

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host " STEP 1 - the test suite. Every check has a must-differ twin." -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
& $py -u 23_test_arms.py
Write-Host ""
Write-Host "  exit code $LASTEXITCODE   (0 = all pass)" -ForegroundColor Yellow

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host " STEP 2 - six arms, 2 epochs each. Same payload, six entry points." -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

foreach ($arm in 'none','add','token','gate','adaln','shuffle') {
    Write-Host ""
    Write-Host "---- arm: $arm ----" -ForegroundColor Green
    & $py -u 24_train.py --arm $arm --seed 0 --epochs 2
    if (Test-Path "runs\${arm}_s0_e2.json") {
        Move-Item "runs\${arm}_s0_e2.json" "runs\_demo\" -Force
    }
    if (Test-Path "runs\${arm}_s0_e2_gates.npy") {
        Move-Item "runs\${arm}_s0_e2_gates.npy" "runs\_demo\" -Force
    }
}

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host " DONE. Two things to look at:" -ForegroundColor Cyan
Write-Host "   1. injection param counts:  0 / 1 / 37,056 / 9,313 / 889,344" -ForegroundColor Cyan
Write-Host "   2. 'shuffle: N coords permuted, 0 fixed points'" -ForegroundColor Cyan
Write-Host " Accuracies at 2 epochs mean nothing. Ignore them." -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
