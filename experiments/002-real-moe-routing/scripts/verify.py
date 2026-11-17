"""Validate saved experimental evidence without rerunning inference."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import unittest

from benchmark import check_answer
from routing import read_mct, convert

ROOT = Path(__file__).resolve().parents[1]


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def verify():
    checks=[]
    def require(condition, description):
        if not condition: raise AssertionError(description)
        checks.append(description)
    plan=read(ROOT/"results/evaluation-plan.json")
    runs=[read(p) for p in sorted((ROOT/"results/runs").glob("eval-*/run.json"))]
    require(len(runs)==16,"Exactly 16 held-out evaluation processes")
    groups=Counter(r["label"].split("-",2)[2] for r in runs)
    require(groups==Counter({k:4 for k in plan["configurations"]}),"Four completed repetitions of each frozen configuration")
    for run in runs:
        require(run["status"]=="complete",run["label"]+": complete")
        fixed=[r for r in run["requests"] if r["payload"]["ignore_eos"]]
        quality=[r for r in run["requests"] if r["correct"] is not None]
        require(len(fixed)==4 and len(quality)==6,run["label"]+": four performance and six quality requests")
        tasks={t["id"]:t for t in run["tasks"]}
        for request in run["requests"]:
            t=request["response"]["timings"]
            require(t["cache_n"]==0,run["label"]+"/"+request["task_id"]+": no reused prompt tokens")
            if request["payload"]["ignore_eos"]:
                require(t["predicted_n"]==128 and len(request["tokens"])==128,run["label"]+"/"+request["task_id"]+": exactly 128 output tokens")
                expected=(t["predicted_n"]-1)/(t["predicted_ms"]/1000)
                require(abs(expected-t["predicted_per_second"])<1e-8,run["label"]+"/"+request["task_id"]+": decode denominator agrees with engine")
            else:
                require(request["correct"]==check_answer(tasks[request["task_id"]],request["content"]),run["label"]+"/"+request["task_id"]+": stored answer score reproduced")
    source=read(ROOT/"results/evaluation-source.json")
    require(hashlib.sha256((ROOT/source["source_snapshot"]).read_bytes()).hexdigest()==source["executed_evaluation_harness_sha256"],"Executed evaluation source snapshot hash matches")
    routing=list((ROOT/"results/runs").glob("trace-*-on/routing-analysis.json"))
    require(len(routing)==3,"Three real routing captures analyzed")
    for path in routing:
        folder=path.parent; report=read(path)
        raw=gzip.decompress((folder/"routing.trace.gz").read_bytes())
        normalized=gzip.decompress((folder/"routing-events.jsonl.gz").read_bytes())
        require(hashlib.sha256(raw).hexdigest()==report["raw_trace_sha256"],folder.name+": raw trace hash matches")
        require(hashlib.sha256(normalized).hexdigest()==report["normalized_trace_sha256"],folder.name+": normalized trace hash matches")
        events,coverage=convert(read(folder/"run.json"),read_mct(raw))
        require(len(events)==report["event_count"] and coverage==report["coverage"],folder.name+": complete request coverage reproduced")
    require(len(list((ROOT/"results/runs").glob("trace-*-off/run.json")))==3,"Three corresponding trace-disabled controls")
    for name in ("cpu","static12","hybrid4096"):
        run=read(ROOT/f"results/runs/reasoning-{name}/run.json")
        require(run["status"]=="complete" and len(run["requests"])==6,f"reasoning-{name}: separate six-task diagnostic complete")
    result=dict(utc=datetime.now(timezone.utc).isoformat(),passed=True,checks=len(checks),details=checks)
    (ROOT/"results/verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(f"Passed {len(checks)} evidence checks")


if __name__=="__main__": verify()
