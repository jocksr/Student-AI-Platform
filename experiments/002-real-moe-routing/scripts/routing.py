"""Validate MCT1 real routing records and replay Experiment 001 cache policies.

Format reference: pinned upstream src/llama-moe-cache.cpp, documented in
results/setup/source-manifest.json. This parser checks complete layer cycles
and request coverage; it does not invent traces for cache-disabled execution.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent/"001-apex-inference"))
from apex.cache import ByteCache
from apex.traces import TraceEvent


def read_mct(data):
    position = 0
    def take(fmt):
        nonlocal position
        size = struct.calcsize("<"+fmt)
        if position+size > len(data): raise ValueError("truncated MCT1 header")
        values = struct.unpack_from("<"+fmt, data, position)
        position += size
        return values
    if data[:4] != b"MCT1": raise ValueError("not an MCT1 trace")
    position = 4
    layers, top_k, n_groups = take("3I")
    if not (1 <= layers <= 256 and 1 <= top_k <= 128 and 1 <= n_groups <= 256):
        raise ValueError("invalid header dimensions")
    groups = []
    for _ in range(n_groups):
        device, experts, n_layers = take("3I")
        if not (1 <= experts <= 4096 and 1 <= n_layers <= layers): raise ValueError("invalid group dimensions")
        layer_ids = list(take(f"{n_layers}I"))
        split, split_sum = take("2d")
        host_bytes, device_host_bytes, alignment = take("3Q")
        n_tensors, = take("I")
        if not 1 <= n_tensors <= 32: raise ValueError("invalid tensor count")
        tensor_bytes = list(take(f"{n_tensors}Q"))
        groups.append(dict(device=device, experts=experts, layers=layer_ids,
            split=split, split_sum=split_sum, host_bytes=host_bytes,
            device_host_bytes=device_host_bytes, alignment=alignment,
            tensor_bytes=tensor_bytes, expert_bytes=sum(tensor_bytes)))
    groups_by_layer = {layer:g for g in groups for layer in g["layers"]}
    records = []
    while position+12 <= len(data):
        start = position
        layer, used, count = take("3i")
        if layer not in groups_by_layer or used != top_k or not (0 < count <= 32768) or count % used:
            raise ValueError(f"invalid record at {start}")
        if position+count*4 > len(data):
            position = start; break
        ids = list(take(f"{count}i"))
        for i in range(0, count, used):
            chosen = ids[i:i+used]
            if len(set(chosen)) != used or any(e < 0 or e >= groups_by_layer[layer]["experts"] for e in chosen):
                raise ValueError(f"invalid expert IDs at {start}")
        records.append(dict(layer=layer, used=used, ids=ids, offset=start))
    return dict(layers=layers, top_k=top_k, groups=groups,
        records=records, trailing_bytes=len(data)-position)


def cycles(parsed):
    """Return complete ordered sweeps of all layers; trailing tail is reported."""
    records = parsed["records"]; n = parsed["layers"]
    result = []
    complete = len(records)//n*n
    for offset in range(0, complete, n):
        group = records[offset:offset+n]
        if [r["layer"] for r in group] != list(range(n)): raise ValueError("incomplete/out-of-order layer cycle")
        sizes = [len(r["ids"])//r["used"] for r in group]
        # The pinned GPT-OSS graph evaluates only the final selected row in
        # its last layer during prefill. Decode still has one row everywhere.
        if len(set(sizes[:-1])) != 1 or sizes[-1] not in (1,sizes[0]):
            raise ValueError("inconsistent token counts across layers")
        result.append(dict(tokens=sizes[0], layer_tokens=sizes, records=group))
    return result, len(records)-complete


def convert(run, parsed):
    blocks, tail_records = cycles(parsed)
    offset = 0; events = []; boundaries = []; initialization_cycles = 0
    if blocks and blocks[0]["tokens"] == 2:
        # Observed startup probe, before the explicitly measured warm-up.
        # Preserve its cache accesses but exclude it from workload statistics.
        for rec in blocks[0]["records"]:
            count=len(rec["ids"])//rec["used"]
            for token in range(count):
                events.append(TraceEvent("initialization","prefill",2-count+token,rec["layer"],
                    tuple(rec["ids"][token*rec["used"]:(token+1)*rec["used"]]),
                    batch_size=count,context_length=2,source="real-small-model",
                    measurements={"origin":"llama.cpp MCT1","record_byte_offset":rec["offset"],"scope":"startup probe"}))
        offset=1; initialization_cycles=1
    requests = [("warmup",run["warmup"])] + [(r["task_id"],r) for r in run["requests"]]
    for request_id, request in requests:
        timing = request["response"]["timings"]
        # This server evaluates the last four prompt tokens in a separate small
        # batch for checkpoints. Assert that observed structure; never silently
        # assign an unrecognized trace layout to a prompt.
        if offset >= len(blocks) or blocks[offset]["tokens"] != 4:
            raise ValueError(f"{request_id}: expected observed 4-token prefill tail")
        start = offset
        selected = [("prefill", blocks[offset])]; offset += 1
        expected_decode = timing["predicted_n"]-1
        for _ in range(expected_decode):
            if offset >= len(blocks) or blocks[offset]["tokens"] != 1:
                raise ValueError(f"{request_id}: incomplete decode coverage")
            selected.append(("decode",blocks[offset])); offset += 1
        decode_step = 0
        for phase, block in selected:
            # MCT1 observes a layer at a time, with contiguous token IDs inside
            # its batch. Preserve that order (especially the prefill tail).
            for rec in block["records"]:
                count=len(rec["ids"])//rec["used"]
                for token in range(count):
                    selected_ids = tuple(rec["ids"][token*rec["used"]:(token+1)*rec["used"]])
                    step = timing["prompt_n"]-count+token if phase == "prefill" else decode_step
                    events.append(TraceEvent(request_id,phase,step,rec["layer"],selected_ids,
                        batch_size=count, context_length=timing["prompt_n"]+(decode_step+1 if phase=="decode" else 0),
                        source="real-small-model", measurements={"origin":"llama.cpp MCT1", "record_byte_offset":rec["offset"]}))
            if phase == "decode": decode_step += 1
        boundaries.append(dict(request_id=request_id, first_cycle=start, next_cycle=offset,
            prompt_tokens=timing["prompt_n"], observed_prefill_tokens_by_layer=blocks[start]["layer_tokens"],
            output_tokens=timing["predicted_n"], observed_decode_steps=expected_decode))
    # The explicit flush workload must start after measured data. This confirms
    # that buffered bytes lost at process termination belong only to the tail.
    if offset >= len(blocks) or blocks[offset]["tokens"] != 4:
        raise ValueError("missing flush-workload boundary after measured requests")
    return events, dict(requests=boundaries, initialization_cycles=initialization_cycles, excluded_flush_cycles=len(blocks)-offset,
        excluded_trailing_records=tail_records, trailing_bytes=parsed["trailing_bytes"])


def replay(events, parsed):
    sizes = {layer:g["expert_bytes"] for g in parsed["groups"] for layer in g["layers"]}
    rows = []
    for budget_mib in (512,1024,2048,3072,4096):
        for policy in ("none","lru","lfu","cost"):
            cache = ByteCache(budget_mib*2**20, policy); stats = Counter()
            by_request = {}
            for event in events:
                for expert in event.selected_experts:
                    key = (event.layer,expert); size = sizes[event.layer]
                    hit = cache.access(key)
                    if not hit: cache.insert(key,size,reload_ns=size)
                    # Warmup and the observed prefill tail update cache state,
                    # but the reported outcomes cover evaluated decode only.
                    if event.request_id in ("warmup","initialization") or event.phase != "decode": continue
                    request = by_request.setdefault(event.request_id,Counter())
                    for counter in (stats,request):
                        counter["accesses"] += 1; counter["logical_bytes"] += size
                        counter["hits"] += int(hit); counter["miss_bytes"] += 0 if hit else size
            rows.append(dict(budget_mib=budget_mib,policy=policy,**stats,
                hit_rate=stats["hits"]/stats["accesses"],
                avoided_transfer_fraction=1-stats["miss_bytes"]/stats["logical_bytes"],
                peak_cache_bytes=cache.peak, by_request=by_request))
    return rows


def analyze(folder):
    run = json.loads((folder/"run.json").read_text(encoding="utf-8"))
    data = (folder/"routing.trace").read_bytes()
    parsed = read_mct(data); events, coverage = convert(run,parsed)
    for event in events: event.validate(24,32,4)
    payload = b"".join((json.dumps(asdict(e),separators=(",",":"))+"\n").encode() for e in events)
    (folder/"routing.trace.gz").write_bytes(gzip.compress(data,mtime=0))
    (folder/"routing-events.jsonl.gz").write_bytes(gzip.compress(payload,mtime=0))
    frequency = Counter((e.layer,x) for e in events if e.request_id not in ("warmup","initialization") and e.phase == "decode" for x in e.selected_experts)
    total = sum(frequency.values())
    layers = []
    for layer in range(24):
        counts = [frequency[layer,e] for e in range(32)]; n = sum(counts)
        layers.append(dict(layer=layer,counts=counts,entropy_bits=-sum(c/n*math.log2(c/n) for c in counts if c),
            top8_share=sum(sorted(counts,reverse=True)[:8])/n))
    result = dict(source="real-small-model", model="GPT-OSS-20B MXFP4", run=run["label"],
        raw_trace_sha256=hashlib.sha256(data).hexdigest(), normalized_trace_sha256=hashlib.sha256(payload).hexdigest(),
        header={k:v for k,v in parsed.items() if k not in ("records","trailing_bytes")},
        coverage=coverage, event_count=len(events), measured_decode_expert_accesses=total,
        unique_experts=len(frequency), layers=layers, replay=replay(events,parsed),
        limitations=["Only cache slot lookups are observable: all decode steps and a prefill tail (four rows in layers 0-22, one in layer 23); bulk prefill is absent. Startup probe is labeled separately.",
            "Experiment 001 global LRU/LFU/cost replay differs from the engine's decayed-frequency cache and is not an engine hit counter.",
            "Replays estimate byte avoidance for the recorded execution path, not measured PCIe traffic or response-time speedup.",
            "All host experts fit RAM. No evidence about beyond-RAM 120B SSD streaming."])
    (folder/"routing-analysis.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("run","event_count","measured_decode_expert_accesses","unique_experts","coverage")},indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("run")
    args = parser.parse_args(); analyze(ROOT/"results/runs"/args.run)
