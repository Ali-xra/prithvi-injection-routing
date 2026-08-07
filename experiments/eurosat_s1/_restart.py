"""Kill the 90-epoch sweep (it started early on a race), then relaunch both
sweeps in the right order: matrix first, long sweep waiting behind it."""
import subprocess, json, time

def procs(pat):
    out = subprocess.run(["powershell", "-Command",
        "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*" + pat +
        "*' } | Select-Object ProcessId | ConvertTo-Json"],
        capture_output=True, text=True).stdout.strip()
    if not out:
        return []
    d = json.loads(out)
    return [d] if isinstance(d, dict) else d

killed = 0
for pat in ("run_long.ps1", "--epochs 90"):
    for p in procs(pat):
        subprocess.run(["taskkill", "/PID", str(p["ProcessId"]), "/T", "/F"],
                       capture_output=True)
        killed += 1
print("killed", killed)
time.sleep(3)

for script in ("run_matrix.ps1", "run_long.ps1"):
    base = r"C:\Users\aliso\Desktop\big-files\loc"
    log = "_matrix.log" if "matrix" in script else "_long.log"
    subprocess.Popen(
        ["powershell", "-Command",
         f"Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File',"
         f"'{base}\\{script}' -RedirectStandardOutput '{base}\\{log}' "
         f"-RedirectStandardError '{base}\\{log}.err' -NoNewWindow"])
    print("launched", script)
    time.sleep(4)
