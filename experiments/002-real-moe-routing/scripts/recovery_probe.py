"""Short, bounded probes beyond the broad sweep's performance drop."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re

from benchmark import ROOT,run,save
from sweet_spot import OUT,TUNING,flags
from summarize import metrics,resources


def read(path):return json.loads(path.read_text(encoding="utf-8"))


def main():
    if (OUT/"recovery-probe.json").exists():raise FileExistsError("Recovery evidence already exists")
    budgets=[4,5,5.25,5.5,5.75,6,7,8,9,10,16,32]
    tasks=[dict(TUNING[0],id="recovery-energy",tokens=128)]
    save(OUT/"recovery-plan.json",dict(utc=datetime.now(timezone.utc).isoformat(),requested_gib=budgets,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),tasks=tasks,
        cpu_share=.9,threads=4,maximum_seconds=90,minimum_available_ram_gib=1,
        rationale="The 2 GiB RAM reserve interrupted 7–8 GiB long runs after a measured 5.5-to-6 GiB performance drop. One shorter request and a 1 GiB reserve probe for recovery without treating guard stops as proof of a hardware performance limit.",
        selection_revision="Validate the fastest working setting and a smallest-within-5% performance choice separately from the earlier conservative headroom alternative. No mixing short-probe rates with longer tuning rates."))
    rows=[]
    def probe(gib,repetition):
        config=dict(cache_mib=int(gib*1024),cpu_share=.9,threads=4)
        label=f"recovery-r{repetition}-c{config['cache_mib']}-s90-t4"
        folder=ROOT/"results/runs"/label
        row=dict(label=label,requested_gib=gib,config=config,repetition=repetition)
        try:
            record=run(label,flags(config),tasks,threads=4,load_mode="mmap",maximum_seconds=90,minimum_available_ram_gib=1)
            row.update(status=record["status"],**metrics(record),resources=resources(folder/"resource-samples.json"))
        except Exception as error:row.update(status="failed",error=str(error))
        record=read(folder/"run.json")
        if record.get("resource_guard"):row["resource_guard"]=record["resource_guard"]
        text=(folder/"server.log").read_text(encoding="utf-8",errors="replace")
        match=re.search(r"MoE cache size =\s*([0-9.]+) MiB",text)
        failed=re.search(r"allocating ([0-9.]+) MiB on device 0: cudaMalloc failed",text)
        row["allocated_cache_mib"]=float(match.group(1)) if match else None
        row["failed_allocation_request_mib"]=float(failed.group(1)) if failed else None
        sample=read(folder/"resource-samples.json")
        free=[r["gpu"]["free_mib"] for r in sample if "free_mib" in r.get("gpu",{})]
        row["min_sampled_gpu_free_mib"]=min(free) if free else None
        rows.append(row);save(OUT/"recovery-probe.json",dict(trials=rows))
        print(f"RECOVERY {gib:g} GiB repeat {repetition}: {row['status']} {row.get('decode_tps')} tok/s guard={row.get('resource_guard')}",flush=True)
        return row
    for gib in budgets:probe(gib,1)
    complete=[r for r in rows if r["status"]=="complete"]
    if not complete:raise RuntimeError("No completed recovery probe")
    best=max(complete,key=lambda r:r["decode_tps"])
    near=[r for r in complete if r["decode_tps"]>=best["decode_tps"]*.95]
    smallest=min(near,key=lambda r:r["config"]["cache_mib"])
    repeat={best["requested_gib"],smallest["requested_gib"]}
    # Repeat a representative post-peak low point, and any subsequent distinct-
    # allocation recovery. Requests beyond the weight-set cap are not new cache sizes.
    later=[r for r in complete if r["requested_gib"]>best["requested_gib"]]
    turning=None
    if later:
        low=min(later,key=lambda r:r["decode_tps"]);repeat.add(low["requested_gib"])
        higher=[r for r in later if r["requested_gib"]>low["requested_gib"] and r["allocated_cache_mib"]>low["allocated_cache_mib"]+1 and r["decode_tps"]>=low["decode_tps"]*1.1]
        if higher:
            recovery=max(higher,key=lambda r:r["decode_tps"]);repeat.add(recovery["requested_gib"])
            turning=dict(low=low,recovery=recovery)
    save(OUT/"recovery-repeat-plan.json",dict(budgets_gib=sorted(repeat),candidate_turning_point=turning))
    for repetition in (2,3):
        order=sorted(repeat,reverse=repetition==2)
        for gib in order:probe(gib,repetition)
    aggregate=[]
    for gib in budgets:
        group=[r for r in rows if r["requested_gib"]==gib]
        good=[r for r in group if r["status"]=="complete"]
        if not good:continue
        aggregate.append(dict(requested_gib=gib,config=good[0]["config"],runs=len(good),all_complete=len(good)==len(group),
            decode_tps=sum(r["decode_steps"] for r in good)/sum(r["decode_seconds"] for r in good),
            run_decode_tps=[r["decode_tps"] for r in good],allocated_cache_mib=good[0]["allocated_cache_mib"],
            min_sampled_gpu_free_mib=min(r["min_sampled_gpu_free_mib"] for r in good)))
    save(OUT/"recovery-aggregates.json",dict(rows=aggregate))
    candidates=[r for r in aggregate if r["all_complete"]]
    best=max(candidates,key=lambda r:r["decode_tps"])
    near=[r for r in candidates if r["decode_tps"]>=best["decode_tps"]*.95]
    smallest=min(near,key=lambda r:r["config"]["cache_mib"])
    previous=read(OUT/"validation-plan.json");save(OUT/"validation-plan.before-recovery.json",previous)
    conservative=previous["calibration_smallest_near_best"]["config"]
    configs=[dict(cache_mib=4096,cpu_share=.95,threads=6),best["config"],smallest["config"],conservative]
    if any(r["requested_gib"]==6 and r["all_complete"] for r in aggregate):configs.append(dict(cache_mib=6144,cpu_share=.9,threads=4))
    configs.append(dict(static_layers=12,threads=6,cache_mib=0,cpu_share=0))
    unique=[]
    for config in configs:
        if config not in unique:unique.append(config)
    previous.update(configurations=unique,calibration_fastest=best,calibration_smallest_near_best=smallest,
        conservative_candidate=conservative,selection_mode="performance-first; conservative headroom alternative reported separately",
        maximum_seconds_per_validation_run=180,minimum_available_ram_gib=1,
        revision="after 4–32 GiB sweep and shorter recovery probes; includes 6 GiB falloff control")
    save(OUT/"validation-plan.json",previous)
    print("Recovery probe finished; final validation candidates",json.dumps(unique),flush=True)


if __name__=="__main__":main()
