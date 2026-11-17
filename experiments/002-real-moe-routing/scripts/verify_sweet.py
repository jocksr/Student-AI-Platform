"""Check sweep coverage, independent validation and recommendation constraints."""
import json
import hashlib
import math
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/sweet-spot"
def read(path):return json.loads(path.read_text(encoding="utf-8"))


def main():
    checks=[]
    def require(condition,message):
        if not condition:raise AssertionError(message)
        checks.append(message)
    initial=read(OUT/"calibration.json")["trials"]
    curves=read(OUT/"curve-calibration.json")["trials"]
    repeats=read(OUT/"curve-repeats.json")["trials"]
    protocol=read(OUT/"curve-plan.json")
    threads=read(OUT/"curve-thread-choice.json")["threads"]
    measured={(r["config"]["cache_mib"],r["config"]["cpu_share"],r["config"]["threads"]) for r in initial+curves}
    for c in protocol["cache_sizes_mib"]:
        for s in protocol["full_curve_cpu_shares"]:
            require((c,s,threads) in measured,f"Full curve covered {c} MiB / {s} CPU share / {threads} threads, including failures")
    require({2,4,6,8,10,12,16}.issubset({t for c,s,t in measured if c==4096 and s==.95}),"Thread sweep reached both ends and did not stop at the first falloff")
    plan=read(OUT/"validation-plan.json")
    validation=read(OUT/"validation.json")["trials"]
    wide=read(OUT/"wide-range.json")["trials"]
    recovery=read(OUT/"recovery-probe.json")["trials"]
    require(len(validation)==3*len(plan["configurations"]),"Three fresh-prompt validation attempts per candidate")
    prompts={t["prompt"] for t in plan["tasks"]}
    tuning={t["prompt"] for t in read(OUT/"plan.json")["tuning_tasks"]}
    require(not prompts.intersection(tuning),"Validation prompts are disjoint from tuning prompts")
    for row in initial+curves+repeats+validation+wide+recovery:
        run=read(ROOT/"results/runs"/row["label"]/"run.json")
        require(run["status"]==row["status"],row["label"]+": status retained accurately")
        snapshot=ROOT/Path(run["harness_source_snapshot"].replace("\\","/"))
        require(hashlib.sha256(snapshot.read_bytes()).hexdigest()==run["harness_sha256"],row["label"]+": executed harness snapshot hash verified")
        if row["status"]!="complete":
            require(bool(run.get("error")),row["label"]+": failure reason retained")
            continue
        for request in run["requests"]:
            timing=request["response"]["timings"]
            require(timing["cache_n"]==0,row["label"]+"/"+request["task_id"]+": no prompt reuse")
            require(timing["predicted_n"]==request["payload"]["n_predict"],row["label"]+"/"+request["task_id"]+": full fixed output count")
        timings=[q["response"]["timings"] for q in run["requests"]]
        expected=sum(t["predicted_n"]-1 for t in timings)/sum(t["predicted_ms"]/1000 for t in timings)
        require(math.isclose(expected,row["decode_tps"],rel_tol=1e-10),row["label"]+": throughput recomputed from raw engine timings")
    selection=read(OUT/"selection.json")
    chosen=selection["recommended"];fastest=selection["fastest"]
    require(chosen["runs"]==3,"Performance choice completed three fresh-prompt validation runs")
    require(chosen["decode_tps"]>=fastest["decode_tps"]*.95,"Performance choice is within 5% of fastest completed validated cache setting")
    near=[r for r in selection["aggregate"] if r.get("runs")==3 and not r["config"].get("static_layers") and r.get("decode_tps",0)>=fastest["decode_tps"]*.95]
    require(chosen["config"]["cache_mib"]==min(r["config"]["cache_mib"] for r in near),"Smallest completed validated near-best cache selected")
    conservative=selection.get("conservative")
    if conservative:
        require(conservative["eligible"] and conservative["min_sampled_gpu_free_mib"]>=512 and conservative["min_available_ram_gib"]>=3,"Conservative alternative meets original headroom criteria")
    wide_plan=read(OUT/"wide-range-plan.json")
    require({r["requested_gib"] for r in wide}==set(wide_plan["requested_gib"]),"Explicit 4–32 GiB requested range fully attempted")
    require(len(wide)==2*len(wide_plan["requested_gib"]),"Both CPU shares attempted throughout the wide range")
    require(any(r["requested_gib"]==32 for r in wide),"32 GiB request tested rather than inferred")
    for budget in (5,5.25,8,10):
        rows=[r for r in recovery if r["requested_gib"]==budget]
        require(len(rows)==3,f"Recovery probe at {budget} GiB repeated three times, including failed attempts")
    require({7,8,9,10,16,32}.issubset({r["requested_gib"] for r in recovery}),"Shorter recovery protocol reached beyond the first long-run stop")
    quality=read(OUT/"quality.json")
    require(quality["status"]=="complete" and quality["checked"]==6,"Selected setting received six normal-reasoning checks")
    require(quality["config"]==chosen["config"],"Quality checks use the actual selected setting")
    for stem in ("wide-range","cache-curves","cpu-settings","validation"):
        for extension in ("png","svg"):
            require((OUT/"graphs"/(stem+"."+extension)).stat().st_size>1000,f"{stem}.{extension} produced")
    result=dict(utc=datetime.now(timezone.utc).isoformat(),passed=True,checks=len(checks),details=checks)
    (OUT/"verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(f"Passed {len(checks)} sweep evidence checks")


if __name__=="__main__":main()
