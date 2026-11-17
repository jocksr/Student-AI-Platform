"""Requested 4–32 GiB cache-budget exploration, with actual allocations recorded."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re

from benchmark import ROOT,run,save
from sweet_spot import TUNING,OUT,flags
from summarize import metrics,resources


def read(path):return json.loads(path.read_text(encoding="utf-8"))


def main():
    if (OUT/"wide-range.json").exists():raise FileExistsError("Wide-range evidence already exists")
    budgets=[4,4.5,5,5.5,6,7,8,9,10,12,16,24,32]
    trace=read(ROOT/"results/runs/trace-r1-on/routing-analysis.json")
    expert_bytes=trace["header"]["groups"][0]["expert_bytes"]
    cap_mib=(768+4+1)*expert_bytes/2**20
    plan=dict(utc=datetime.now(timezone.utc).isoformat(),requested_gib=budgets,cpu_shares=[.9,.95],threads=4,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        expected_full_cache_buffer_mib_from_trace=cap_mib,
        scope="Stress exploration requested by user: 4–32 GiB budgets, including above normal GPU headroom. Requested budget may exceed actual allocation or fail.",
        guards="No minimum free-GPU guard for this stress sweep; per-run wall limit 180 s and system available-RAM floor 2 GiB. Existing OS memory settings are unchanged.",
        tasks=TUNING,selection="Stress results do not automatically qualify for recommendation; normal 512 MiB GPU / 3 GiB RAM headroom rules still apply.")
    save(OUT/"wide-range-plan.json",plan)
    rows=[]
    for gib in budgets:
        for share in (.9,.95):
            config=dict(cache_mib=int(gib*1024),cpu_share=share,threads=4)
            label=f"wide-c{config['cache_mib']}-s{int(share*100)}-t4"
            folder=ROOT/"results/runs"/label
            row=dict(label=label,requested_gib=gib,config=config)
            try:
                result=run(label,flags(config),TUNING,threads=4,load_mode="mmap",
                    maximum_seconds=180,minimum_available_ram_gib=2)
                row.update(status=result["status"],**metrics(result),resources=resources(folder/"resource-samples.json"))
            except Exception as error:
                row.update(status="failed",error=str(error))
            record=read(folder/"run.json")
            if record.get("resource_guard"):row["resource_guard"]=record["resource_guard"]
            log=(folder/"server.log").read_text(encoding="utf-8",errors="replace")
            allocated=re.search(r"MoE cache size =\s*([0-9.]+) MiB",log)
            failed_alloc=re.search(r"allocating ([0-9.]+) MiB on device 0: cudaMalloc failed",log)
            row["allocated_cache_mib"]=float(allocated.group(1)) if allocated else None
            row["failed_allocation_request_mib"]=float(failed_alloc.group(1)) if failed_alloc else None
            row["expected_effective_cache_cap_mib"]=cap_mib
            samples=read(folder/"resource-samples.json") if (folder/"resource-samples.json").exists() else []
            available=[r["gpu"]["free_mib"] for r in samples if "free_mib" in r.get("gpu",{})]
            row["min_sampled_gpu_free_mib"]=min(available) if available else None
            row["eligible"]=row["status"]=="complete" and bool(available) and min(available)>=512 and row["resources"]["min_available_ram_gib"]>=3
            rows.append(row)
            save(OUT/"wide-range.json",dict(trials=rows,requested_range_gib=[4,32]))
            print(f"WIDE requested={gib:g} GiB share={share:g}: {row['status']}, allocated={row['allocated_cache_mib']} MiB, failed_alloc={row['failed_allocation_request_mib']}, tps={row.get('decode_tps')}",flush=True)
    # Preserve the candidate plan before incorporating any new eligible finding.
    previous=read(OUT/"validation-plan.json")
    save(OUT/"validation-plan.before-wide.json",previous)
    existing=read(OUT/"curve-aggregates.json")["rows"]
    eligible=[r for r in existing if r["eligible"]]+[r for r in rows if r["eligible"]]
    fastest=max(eligible,key=lambda r:r["decode_tps"])
    near=[r for r in eligible if r["decode_tps"]>=fastest["decode_tps"]*.95]
    smallest=min(near,key=lambda r:(r["config"]["cache_mib"],r["config"]["threads"],-r["decode_tps"]))
    configs=[dict(cache_mib=4096,cpu_share=.95,threads=6),fastest["config"],smallest["config"]]
    # Also validate the best low-thread curve point, if it is a different tradeoff.
    lower_thread=[r for r in eligible if r["config"]["threads"]==4]
    if lower_thread:configs.append(max(lower_thread,key=lambda r:r["decode_tps"])["config"])
    # A negative control tests whether the 100% miss-share falloff persists on
    # fresh prompts; changing only the share isolates this observed cliff.
    configs.append(dict(fastest["config"],cpu_share=1.0))
    configs.append(dict(static_layers=12,threads=6,cache_mib=0,cpu_share=0))
    unique=[]
    for config in configs:
        if config not in unique:unique.append(config)
    previous.update(configurations=unique,calibration_fastest=fastest,calibration_smallest_near_best=smallest,
        revision="after explicit 4–32 GiB request-budget sweep; includes 100% CPU-miss negative control and low-thread alternative")
    save(OUT/"validation-plan.json",previous)
    print("Wide-range sweep complete; validation plan updated",flush=True)


if __name__=="__main__":main()
