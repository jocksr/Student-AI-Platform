import argparse
from pathlib import Path
from .common import write_json,read_json


def main():
    p=argparse.ArgumentParser(description="APEX research harness: measured benchmarks and explicitly labeled simulations")
    commands=p.add_subparsers(dest="command",required=True)
    for name in ("hardware","benchmark-memory","benchmark-storage","benchmark-gpu"):
        s=commands.add_parser(name); s.add_argument("--out",default="results/"+name.removeprefix("benchmark-")+".json")
        if name=="benchmark-storage": s.add_argument("--directory",default=".apex-cache")
    s=commands.add_parser("generate-trace")
    s.add_argument("--out",default="results/trace.jsonl")
    for name,default in (("steps",64),("layers",36),("experts",128),("top-k",4),("seed",17),("requests",1)):
        s.add_argument("--"+name,type=int,default=default)
    s.add_argument("--workload",choices=["uniform","hotset","phase-shift"],default="hotset")
    s=commands.add_parser("simulate")
    s.add_argument("--trace",required=True); s.add_argument("--out",default="results/simulation.json")
    s.add_argument("--policy",choices=["none","lru","lfu","cost"],default="lru")
    s.add_argument("--layers",type=int,default=36); s.add_argument("--experts",type=int,default=128)
    s.add_argument("--expert-bytes",type=int,default=13236480)
    s.add_argument("--ram-mib",type=int,default=8192); s.add_argument("--gpu-mib",type=int,default=1024)
    s.add_argument("--storage-gbs",type=float,default=1); s.add_argument("--h2d-gbs",type=float,default=8)
    s=commands.add_parser("benchmark-compression"); s.add_argument("--out",default="results/compression.json")
    s.add_argument("--samples",help="Directory with verified bounded weight sample manifest; omitted means synthetic only")
    s=commands.add_parser("cross-expert"); s.add_argument("--samples",default="benchmarks/weight-samples"); s.add_argument("--out",default="results/cross-expert.json")
    s=commands.add_parser("compare"); s.add_argument("--results",default="results"); s.add_argument("--steps",type=int,default=64)
    s=commands.add_parser("report"); s.add_argument("--results",default="results")
    a=p.parse_args()
    try:
        if a.command=="hardware":
            from .hardware import inventory
            result=inventory()
        elif a.command=="benchmark-memory":
            from .benchmarks import memory_benchmark
            result=memory_benchmark()
        elif a.command=="benchmark-storage":
            from .benchmarks import storage_benchmark
            result=storage_benchmark(a.directory)
        elif a.command=="benchmark-gpu":
            from .gpu import benchmark_gpu
            result=benchmark_gpu()
        elif a.command=="generate-trace":
            from .traces import generate,save_trace
            config={k:getattr(a,k) for k in ("steps","layers","experts","top_k","seed","workload","requests")}
            save_trace(a.out,generate(**config),dict(config,source="synthetic")); print(a.out); return
        elif a.command=="simulate":
            from .cache import ExpertRecord
            from .simulation import replay,CostModel
            from .traces import load_trace
            if min(a.layers,a.experts,a.expert_bytes)<1: raise ValueError("model dimensions must be positive")
            records={(l,e):ExpertRecord(l,e,a.expert_bytes,a.expert_bytes) for l in range(a.layers) for e in range(a.experts)}
            result=replay(load_trace(a.trace,a.layers,a.experts),records,a.ram_mib*2**20,a.gpu_mib*2**20,
                          a.policy,CostModel(a.storage_gbs,a.h2d_gbs))
        elif a.command=="benchmark-compression":
            from .experiments import load_corpora
            from .compression import benchmark
            result=benchmark(load_corpora(a.samples))
        elif a.command=="cross-expert":
            from .experiments import cross_expert
            result=cross_expert(a.samples)
        elif a.command=="compare":
            from .experiments import compare
            compare(a.results,a.steps); print(str(Path(a.results)/"comparison.json")); return
        else:
            from .report import render
            print(render(a.results)); return
        write_json(a.out,result); print(a.out)
    except (ValueError,OSError) as e:
        p.exit(2,f"APEX: {e}\n")
