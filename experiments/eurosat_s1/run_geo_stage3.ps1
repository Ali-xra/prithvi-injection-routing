# run_geo_stage3.ps1 - seeds 3 and 4 for all six arms on the geo split. ASCII only.
#
# Why: the pre-registration specifies 5 seeds per arm and defines the threshold as
# s*sqrt(2/5). Stages 1 and 2 ran 3 seeds and the rule was applied with sqrt(2/3),
# which is a stated deviation. This closes it.
#
# With 3 seeds the verdict was: all four injection arms beat `none` by 3.6-4.4
# points, and no pair of injection arms differs. 12 runs, ~6 min each.

$ErrorActionPreference = 'Stop'
Set-Location "C:\Users\aliso\Desktop\big-files\loc"

$py = "C:\Users\aliso\Desktop\big-files\venv-gpu\Scripts\python.exe"
& $py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA unavailable - aborting" }

$start = Get-Date
foreach ($seed in 3,4) {
    foreach ($arm in 'none','add','token','gate','adaln','shuffle') {
        $tag = "${arm}_s${seed}_geo"
        if (Test-Path "runs\$tag.json") { Write-Host "[skip] $tag"; continue }
        Write-Host "[run ] $tag   $(Get-Date -Format 'HH:mm:ss')"
        & $py -u 24_train.py --arm $arm --seed $seed --split geo `
            1> "_run_$tag.log" 2> "_run_$tag.err"
        if (-not (Test-Path "runs\$tag.json")) {
            Write-Host "[FAIL] $tag"; Get-Content "_run_$tag.err" -Tail 5
        } else { Write-Host "[ok  ] $tag" }
    }
}
Write-Host "[ALL DONE] $((Get-Date) - $start)"
