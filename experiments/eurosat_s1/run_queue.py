# -*- coding: utf-8 -*-
"""
run_queue.py - self-driving experiment runner for the gate sprint.

Runs a list of experiments back to back (queue.json). After each: reads the
result's headline, records it to the repo's QUEUE_STATUS.md, git-commits, and
  - if the item is marked halt_on_divergence and the headline lands outside
    [expected +/- tol] (or its verdict is in halt_verdicts), STOPS and writes
    DECISION-NEEDED.md so the next code can be rewritten. This is the
    "if a result changes direction, rewrite the code" rule.
  - otherwise moves to the next item.
Resumable: a completed item (its result json present and recorded) is skipped.

The runner does NOT invent new arms; it only runs scripts that already exist and
are validated. New arms (readout zoo) are added by a human between waves, each
with its must-differ twin, per the brief.
"""
import sys, json, time, subprocess, shutil
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
LOC = Path(__file__).resolve().parent
REPO = Path(r"C:\Users\aliso\Desktop\proje\uni\ideas\prithvi-injection-routing")
DOCS = REPO / "docs"
RESULTS = REPO / "experiments" / "eurosat_s1" / "results"
STATUS = DOCS / "QUEUE_STATUS.md"
STATE = LOC / "queue_state.json"
PY = r"C:\Users\aliso\Desktop\big-files\venv-gpu\Scripts\python.exe"
QUEUE = json.loads((LOC / "queue.json").read_text(encoding="utf-8"))


def dig(d, dotted):
    for k in dotted.split("."):
        d = d[k]
    return d


def git(msg, files):
    # add ONLY the runner's own files -- never `git add -A`, because the user may
    # be editing the repo at the same time and must not be swept into these commits.
    paths = [str(f) for f in files if Path(f).exists()]
    if not paths:
        return
    try:
        subprocess.run(["git", "add"] + paths, cwd=REPO, check=True,
                       capture_output=True, text=True)
        # commit ONLY these paths (pathspec), so nothing the user staged rides along
        subprocess.run(["git", "commit", "-m", msg, "--"] + paths, cwd=REPO, check=True,
                       capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("   git note:", (e.stderr or e.stdout or "").strip()[:200], flush=True)


def note(line):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"done": []}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=2))


def wait_for_json(path, must_contain_kv, timeout_min=180):
    """Poll until result json exists and matches a marker (e.g. design='per-seed')."""
    t0 = time.time()
    while time.time() - t0 < timeout_min * 60:
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                k, v = must_contain_kv
                if v in str(d.get(k, "")):
                    return d
            except Exception:
                pass
        time.sleep(20)
    return None


def run_item(item):
    name = item["name"]
    res_path = LOC / item["result"]
    if item.get("script"):
        print(f"\n### running {name}: {item['script']}", flush=True)
        log = LOC / f"_queue_{name}.log"
        with open(log, "w", encoding="utf-8") as lf:
            subprocess.run([PY, item["script"]], cwd=LOC, stdout=lf,
                           stderr=subprocess.STDOUT, text=True)
        d = json.loads(res_path.read_text(encoding="utf-8"))
    else:
        # already running elsewhere (e.g. logit-prior); wait for its marker
        marker = tuple(item["wait_marker"])
        print(f"\n### collecting {name}: waiting for {item['result']} ({marker})", flush=True)
        d = wait_for_json(res_path, marker)
        if d is None:
            return None, "TIMEOUT waiting for result"
    return d, None

def main():
    st = load_state()
    note(f"\n## queue run started {time.strftime('%Y-%m-%d %H:%M')}")
    git("queue: run started", [STATUS])
    for item in QUEUE:
        name = item["name"]
        if name in st["done"]:
            print(f"skip {name} (already done)", flush=True); continue

        d, err = run_item(item)
        if err:
            note(f"- **{name}**: ERROR {err} -- queue HALTED"); git(f"queue: {name} error, halted", [STATUS])
            print("HALT:", err, flush=True); return

        head = dig(d, item["headline"])
        verdict = str(d.get("verdict", ""))
        exp, tol = item.get("expected"), item.get("tol")
        diverged = False
        if verdict and verdict in item.get("halt_verdicts", []):
            diverged = True
        if exp is not None and tol is not None and abs(head - exp) > tol:
            diverged = True

        # persist the result json into the repo and record a line
        RESULTS.mkdir(parents=True, exist_ok=True)
        shutil.copy(LOC / item["result"], RESULTS / item["result"])
        note(f"- **{name}**  headline `{item['headline']}` = {head}"
             f"  (expected {exp}+/-{tol})  verdict={verdict or '-'}"
             f"  {'-> DIVERGED' if diverged else 'ok'}")

        st["done"].append(name); save_state(st)

        if diverged and item.get("halt_on_divergence", True):
            (REPO / "docs" / "DECISION-NEEDED.md").write_text(
                f"# Decision needed after `{name}`\n\n"
                f"headline `{item['headline']}` = {head}, expected {exp} +/- {tol}, "
                f"verdict = {verdict}.\n\nThis is a direction-changing result. The queue "
                f"halted so the next experiment's code can be rewritten before spending "
                f"more compute. See {item['result']}.\n", encoding="utf-8")
            note(f"  queue HALTED after {name} -- see DECISION-NEEDED.md")
            git(f"queue: {name} diverged (headline={head}), halted for rewrite",
                [STATUS, RESULTS / item["result"], REPO / "docs" / "DECISION-NEEDED.md"])
            print(f"HALT after {name}: diverged", flush=True); return

        git(f"queue: {name} done (headline={head}, verdict={verdict or '-'})",
            [STATUS, RESULTS / item["result"]])
        print(f"   {name} recorded, committed. continuing.", flush=True)

    note(f"\n## queue finished {time.strftime('%Y-%m-%d %H:%M')} -- all items done")
    git("queue: all items done", [STATUS])
    print("QUEUE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
