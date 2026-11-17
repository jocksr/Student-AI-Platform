"""Serialized lower-level movement model, deliberately NOT an inference engine.

VRAM and host caches are independent (may duplicate). Storage is immutable.
All latencies sum sequentially: there is no invented overlap or free prefetch.
"""
from collections import Counter
from dataclasses import dataclass,asdict
import math
from .cache import ByteCache
from .common import provenance


@dataclass(frozen=True)
class CostModel:
    storage_bytes_per_ns: float = 1.0
    h2d_bytes_per_ns: float = 8.0
    storage_request_ns: float = 0.0
    h2d_request_ns: float = 0.0
    io_block_bytes: int = 1048576
    cpu_expert_ns: float = 0.0
    gpu_expert_ns: float = 0.0
    activation_roundtrip_ns: float = 0.0
    origin: str = "explicit scenario assumptions"

    def __post_init__(self):
        for value in (self.storage_bytes_per_ns,self.h2d_bytes_per_ns):
            if not math.isfinite(value) or value<=0: raise ValueError("bandwidth must be finite and positive")
        for value in (self.storage_request_ns,self.h2d_request_ns,self.cpu_expert_ns,self.gpu_expert_ns,self.activation_roundtrip_ns):
            if not math.isfinite(value) or value<0: raise ValueError("costs must be finite and nonnegative")
        if self.io_block_bytes<1: raise ValueError("invalid I/O block")

    def storage(self,size): return size/self.storage_bytes_per_ns+math.ceil(size/self.io_block_bytes)*self.storage_request_ns
    def transfer(self,size): return size/self.h2d_bytes_per_ns+self.h2d_request_ns


def measured_costs(storage,gpu):
    # Measured effective rate already includes API/latency at these sizes.
    # Do not add those same fixed costs again.
    candidates=[r for r in storage["rows"] if r.get("mode")=="windows_unbuffered" and r.get("pattern")=="random" and r.get("block_bytes")==1048576]
    transfers=[r for r in gpu["transfer_rows"] if r["direction"]=="h2d" and r["host_memory"]=="pinned" and r["bytes"]>=2**20]
    if not candidates or not transfers: raise ValueError("direct storage and pinned GPU transfer measurements required")
    return CostModel(storage_bytes_per_ns=candidates[0]["throughput_GB_s"],h2d_bytes_per_ns=transfers[-1]["GB_s"],
                     origin="Measured effective 1 MiB random unbuffered SSD rate and large pinned H2D rate; linear size extrapolation; compute excluded")


def replay(events,records,ram_budget,gpu_budget,policy,cost,host_representation="raw",placement="gpu"):
    if host_representation not in ("raw","compressed"): raise ValueError("unknown representation")
    if placement not in ("gpu","cpu","adaptive"): raise ValueError("unknown placement")
    if placement!="gpu" and (cost.cpu_expert_ns<=0 or cost.gpu_expert_ns<=0):
        raise ValueError("CPU/adaptive placement needs explicit positive compute costs")
    host=ByteCache(ram_budget,policy); gpu=ByteCache(gpu_budget,policy)
    metrics=Counter(); per_layer={}; seen=set(); token_costs={}; sources=set()
    for event in events:
        sources.add(event.source)
        token=(event.request_id,event.phase,event.step)
        token_costs.setdefault(token,0.0)
        for expert in event.selected_experts:
            key=(event.layer,expert); record=records[key]; metrics["accesses"]+=1
            layer=per_layer.setdefault(event.layer,Counter()); layer["accesses"]+=1
            metrics["logical_expert_bytes"]+=record.original_size
            gpu_hit=gpu.access(key); elapsed=0.0
            if gpu_hit:
                metrics["gpu_hits"]+=1; layer["gpu_hits"]+=1
                metrics["gpu_hit_bytes"]+=record.original_size
                elapsed=cost.gpu_expert_ns
                metrics["gpu_executions"]+=1
            else:
                metrics["gpu_misses"]+=1
                host_hit=host.access(key)
                if host_hit: metrics["ram_hits"]+=1
                else:
                    metrics["ssd_misses"]+=1
                    if key in seen: metrics["ssd_reloads"]+=1
                    metrics["ssd_bytes"]+=record.compressed_size
                    metrics["ssd_operations"]+=math.ceil(record.compressed_size/cost.io_block_bytes)
                    elapsed+=cost.storage(record.compressed_size)
                    # Raw host storage requires expansion at load; compressed host is decoded on use below.
                    if host_representation=="raw":
                        elapsed+=record.decode_ns; metrics["decode_ns"]+=record.decode_ns
                    host_size=record.original_size if host_representation=="raw" else record.compressed_size
                    host.insert(key,host_size,cost.storage(record.compressed_size)+record.decode_ns)
                if host_representation=="compressed":
                    elapsed+=record.decode_ns; metrics["decode_ns"]+=record.decode_ns
                cpu_cost=cost.cpu_expert_ns+cost.activation_roundtrip_ns
                gpu_cost=cost.transfer(record.original_size)+cost.gpu_expert_ns
                on_cpu=placement=="cpu" or (placement=="adaptive" and cpu_cost<gpu_cost)
                if on_cpu:
                    elapsed+=cpu_cost; metrics["cpu_executions"]+=1
                else:
                    elapsed+=gpu_cost; metrics["gpu_executions"]+=1
                    metrics["h2d_bytes"]+=record.original_size; metrics["h2d_operations"]+=1
                    gpu.insert(key,record.original_size,gpu_cost)
            seen.add(key); metrics["estimated_serial_ns"]+=elapsed; token_costs[token]+=elapsed
    accesses=metrics["accesses"]
    for name in ("gpu_hits","gpu_misses","ram_hits","ssd_misses","ssd_reloads","ssd_bytes","h2d_bytes","decode_ns","estimated_serial_ns","logical_expert_bytes","gpu_hit_bytes"):
        metrics.setdefault(name,0)
    totals=sorted(token_costs.values())
    return provenance("simulated-movement",policy=policy,host_representation=host_representation,
        placement=placement,source_categories=sorted(sources),cost_model=asdict(cost),metrics=dict(metrics),
        gpu_hit_rate=metrics["gpu_hits"]/accesses if accesses else 0,
        gpu_byte_hit_rate=metrics["gpu_hit_bytes"]/metrics["logical_expert_bytes"] if accesses else 0,
        token_steps=len(totals),mean_step_movement_ms=sum(totals)/len(totals)/1e6 if totals else 0,
        p95_step_movement_ms=totals[min(len(totals)-1,int(.95*len(totals)))]/1e6 if totals else 0,
        ram_budget=ram_budget,gpu_budget=gpu_budget,ram_peak=host.peak,gpu_peak=gpu.peak,
        ram_evictions=host.evictions,gpu_evictions=gpu.evictions,
        per_layer={str(k):dict(v) for k,v in per_layer.items()},
        limitations=["No attention, KV cache, routing, synchronization, actual inference, or multi-request contention in movement-only runs.",
                     "Sequential costs, no overlap/prefetch. Caches start empty. No filesystem cache is modeled.",
                     "Host and GPU budgets exclude OS, non-expert tensors, KV cache and temporary buffers, which must be reserved separately.",
                     "Synthetic routing and extrapolated compression are scenarios, not target-model performance."])
