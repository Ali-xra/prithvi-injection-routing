# ASCII ONLY.
# Waits for the 30-epoch matrix to finish, then runs the same arms at 90 epochs.
#
#   powershell -ExecutionPolicy Bypass -File C:\Users\aliso\Desktop\big-files\loc\run_long.ps1
#
# Why: at 30 epochs EVERY arm was still climbing, and adaLN was climbing fastest
# (+3.51 over the last five epochs vs +2.84 for add), peaked latest (epoch 28.0
# vs 26.5), and ended exactly at its own peak (best-minus-final = 0.00) - the
# signature of a run that was cut off rather than converged.
#
# adaLN is zero-initialised, so it passes no location at step 0 and must learn
# to open its own gate, while add starts at scale=0.1 with location already
# flowing. DiT trains adaLN-Zero for hundreds of thousands of steps; we gave it
# about 1900.
#
# So "adaLN is the worst injection point" is currently a claim about a fixed
# short budget. Tripling the budget separates the two readings:
#   adaLN catches up  -> it is SLOWER, not worse (3x the compute for parity)
#   adaLN stays down  -> it is genuinely worse, and the claim gets stronger
# Both are usable. Neither is a wasted night.
#
# Runs are tagged _e90 so the two budgets can never overwrite each other.

$B    = "C:\Users\aliso\Desktop\big-files\loc"
$PY   = "C:\Users\aliso\Desktop\big-files\venv-gpu\Scripts\python.exe"
$SRC  = "$B\24_train.py"
$RUNS = "$B\runs"
$env:PYTHONIOENCODING = "utf-8"

# Wait for the 30-epoch matrix to COMPLETE - counted by results on disk, not by
# process detection.
#
# The first version watched for a running 24_train.py process. That fired the
# instant the matrix paused between two runs, so both sweeps ended up on the GPU
# together. Worse, the matrix had actually DIED seven runs earlier (shuffle_s3,
# a derangement bug) and "no process running" read as "finished".
#
# Absence of a process is not evidence of completion. Count the artefacts.
Write-Host "waiting for all 30 runs of the 30-epoch matrix ..." -ForegroundColor Cyan
while ($true) {
  $n = (Get-ChildItem "$RUNS\*.json" -ErrorAction SilentlyContinue |
        Where-Object { $_.BaseName -notlike "*_e90" } | Measure-Object).Count
  if ($n -ge 30) { break }
  $busy = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*24_train.py*" }
  if (-not $busy) {
    Write-Host ("only {0}/30 runs and nothing is training - the matrix stopped early." -f $n) -ForegroundColor Red
    Write-Host "fix it and rerun run_matrix.ps1; this script will not start." -ForegroundColor Red
    exit 1
  }
  Start-Sleep -Seconds 60
}
Write-Host "all 30 runs present, starting the 90-epoch sweep" -ForegroundColor Green

$arms  = @("none","add","token","adaln")
$seeds = @(0,1,2)

foreach ($seed in $seeds) {
  foreach ($arm in $arms) {
    $tag  = "${arm}_s${seed}_e90"
    $done = "$RUNS\$tag.json"
    if (Test-Path $done) {
      Write-Host ("SKIP  {0}" -f $tag) -ForegroundColor DarkGray
      continue
    }
    $log = "$B\_run_$tag.log"
    Write-Host ("RUN   {0,-16} {1}" -f $tag, (Get-Date -Format "HH:mm")) -ForegroundColor Green
    $t0 = Get-Date
    & $PY $SRC --arm $arm --seed $seed --epochs 90 1>$log 2>"$log.err"
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if (Test-Path $done) {
      $d = Get-Content $done -Raw | ConvertFrom-Json
      Write-Host ("DONE  {0,-16} val {1:N2}   {2} min" -f $tag, ($d.best_val_acc * 100), $mins) -ForegroundColor Green
    } else {
      Write-Host ("FAIL  {0} after {1} min" -f $tag, $mins) -ForegroundColor Red
      Get-Content "$log.err" -Tail 8 -ErrorAction SilentlyContinue
      exit 1
    }
  }
}
Write-Host "90-epoch sweep complete" -ForegroundColor Cyan
