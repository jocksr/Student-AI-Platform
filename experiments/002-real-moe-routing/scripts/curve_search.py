"""Extend the sweep through dips and later recoveries, before validation."""
import hashlib
import json
from pathlib import Path
import random

from benchmark import save
from sweet_spot import OUT,TUNING,VALIDATION,trial,key,name


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def main():
    if (OUT/"curve-calibration.json").exists(): raise FileExistsError("Curve evidence already exists")
    initial=read(OUT/"calibration.json")["trials"]
    original_plan=read(OUT/"validation-plan.json")
    save(OUT/"validation-plan.initial.json",original_plan)
    protocol=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reason="User requested sampling beyond performance falloff to detect a later re-rise, and a graph.",
        cache_sizes_mib=[512,1024,1536,2048,2560,3072,3584,4096,4352,4608],
        full_curve_cpu_shares=[.9,.95,1.0], additional_shares=[0,.5],
        additional_share_cache_sizes=[2048,3072,4096,4608],
        additional_thread_counts=[2,12,16],
        thread_selection="fewest threads within 3% of highest observed 4096 MiB/.95 calibration throughput",
        safeguards="abort a run below 256 MiB observed free GPU memory or 2 GiB available RAM; recommendation requires >=512 MiB GPU and >=3 GiB RAM",
        repeated_turning_points="repeat twice the endpoints of each >=5% cache or thread dip followed by >=5% recovery, plus leading eligible configurations",
        limits="4608 MiB upper cache test boundary on this 8 GiB display GPU; no inference of performance beyond that boundary")
    save(OUT/"curve-plan.json",protocol)
    extra=[]; known={key(r["config"]):r for r in initial}; rng=random.Random(2051)
    def measure(configs,stage):
        configs=list(configs);rng.shuffle(configs)
        for config in configs:
            if key(config) in known: continue
            row=trial("sweet-curve-"+name(config),config,TUNING);row["stage"]=stage
            extra.append(row);known[key(config)]=row
            save(OUT/"curve-calibration.json",dict(trials=extra,stage=stage))
            print(f"CURVE {name(config)}: {row.get('decode_tps')} tok/s; eligible={row['eligible']}; free={row.get('min_sampled_gpu_free_mib')}",flush=True)
    measure([dict(cache_mib=4096,cpu_share=.95,threads=t) for t in (2,12,16)],"extended-threads")
    thread_rows=[r for r in known.values() if r["status"]=="complete" and r["config"]["cache_mib"]==4096 and r["config"]["cpu_share"]==.95]
    best_rate=max(r["decode_tps"] for r in thread_rows)
    threads=min(r["config"]["threads"] for r in thread_rows if r["decode_tps"]>=best_rate*.97)
    save(OUT/"curve-thread-choice.json",dict(threads=threads,reference_best_tps=best_rate,candidates=thread_rows))
    measure([dict(cache_mib=c,cpu_share=s,threads=threads) for c in protocol["cache_sizes_mib"] for s in (.9,.95,1.0)],"full-cache-curves")
    measure([dict(cache_mib=c,cpu_share=.9,threads=threads) for c in (3840,3968,4224)],"near-peak-cache-resolution")
    measure([dict(cache_mib=c,cpu_share=s,threads=threads) for c in (2048,3072,4096,4608) for s in (0,.5)],"control-curves")
    all_rows=list(known.values()); turning=[]; repeat_configs={}
    def scan(rows,axis,label):
        rows=sorted([r for r in rows if r["status"]=="complete"],key=lambda r:r["config"][axis])
        for i in range(1,len(rows)-1):
            before,dip=rows[i-1],rows[i]
            if dip["decode_tps"]>=before["decode_tps"]*.95: continue
            later=next((r for r in rows[i+1:] if r["decode_tps"]>=dip["decode_tps"]*1.05),None)
            if later:
                turning.append(dict(curve=label,before=before["label"],dip=dip["label"],recovery=later["label"],axis=axis))
                for r in (before,dip,later):
                    if r["eligible"]: repeat_configs[key(r["config"])]=r["config"]
    for share in (0,.5,.9,.95,1.0):
        scan([r for r in all_rows if r["config"]["threads"]==threads and r["config"]["cpu_share"]==share],"cache_mib",f"cache-size/share={share}/threads={threads}")
    scan(thread_rows,"threads","threads/cache=4096/share=.95")
    for cache in sorted({r["config"]["cache_mib"] for r in all_rows}):
        scan([r for r in all_rows if r["config"]["cache_mib"]==cache and r["config"]["threads"]==threads],"cpu_share",f"cpu-share/cache={cache}/threads={threads}")
    eligible=[r for r in all_rows if r["eligible"]]
    if not eligible: raise RuntimeError("No eligible curve candidate")
    fastest=max(eligible,key=lambda r:r["decode_tps"])
    near=[r for r in eligible if r["decode_tps"]>=fastest["decode_tps"]*.95]
    smallest=min(near,key=lambda r:(r["config"]["cache_mib"],r["config"]["threads"],-r["decode_tps"]))
    for r in (fastest,smallest):repeat_configs[key(r["config"])]=r["config"]
    save(OUT/"turning-points.json",dict(preliminary=turning,repeat_configurations=list(repeat_configs.values()),threshold_pct=5,
        note="Single-run turning points are candidates, not established physical effects; repeated measurements determine stability."))
    repeats=[]
    for repetition in (2,3):
        configs=list(repeat_configs.values());rng.shuffle(configs)
        for config in configs:
            row=trial(f"sweet-repeat-r{repetition}-"+name(config),config,TUNING);repeats.append(row)
            save(OUT/"curve-repeats.json",dict(trials=repeats))
    aggregates=[]
    for config in [r["config"] for r in all_rows]:
        rows=[r for r in all_rows+repeats if key(r["config"])==key(config)]
        complete=[r for r in rows if r["status"]=="complete"]
        if not complete: continue
        aggregates.append(dict(config=config,runs=len(complete),eligible=all(r["eligible"] for r in rows),
            decode_tps=sum(r["decode_steps"] for r in complete)/sum(r["decode_seconds"] for r in complete),
            run_decode_tps=[r["decode_tps"] for r in complete],
            min_sampled_gpu_free_mib=min(r["min_sampled_gpu_free_mib"] for r in complete)))
    save(OUT/"curve-aggregates.json",dict(rows=aggregates))
    eligible=[r for r in aggregates if r["eligible"]]
    fastest=max(eligible,key=lambda r:r["decode_tps"])
    near=[r for r in eligible if r["decode_tps"]>=fastest["decode_tps"]*.95]
    smallest=min(near,key=lambda r:(r["config"]["cache_mib"],r["config"]["threads"],-r["decode_tps"]))
    candidates=[dict(cache_mib=4096,cpu_share=.95,threads=6),fastest["config"],smallest["config"]]
    unique=[]
    for config in candidates:
        if config not in unique: unique.append(config)
    unique.append(dict(static_layers=12,threads=6,cache_mib=0,cpu_share=0))
    save(OUT/"validation-plan.json",dict(configurations=unique,calibration_fastest=fastest,calibration_smallest_near_best=smallest,
        tasks=VALIDATION,repetitions=3,revision="extended curves, including falls and possible recoveries; initial plan retained separately"))
    print("Full curves complete; final validation candidates:",json.dumps(unique),flush=True)


if __name__=="__main__": main()
