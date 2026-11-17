from collections import defaultdict
import math
from pathlib import Path
import numpy as np
from .cache import ExpertRecord
from .common import provenance,read_json,write_json,sha256
from .compression import benchmark,select_representation,synthetic_corpora,encode,decode
from .simulation import measured_costs,replay
from .traces import generate,save_trace,analyze


def load_corpora(sample_dir=None):
    corpora=synthetic_corpora()
    if sample_dir:
        base=Path(sample_dir); manifest=read_json(base/"manifest.json")
        groups=defaultdict(list)
        for row in manifest["samples"]:
            data=(base/row["file"]).read_bytes()
            if sha256(data)!=row["sha256"]: raise ValueError("weight sample integrity failure")
            family="target-scales" if row["tensor"].endswith("_scales") else "target-packed-blocks"
            groups[family].append(data)
        corpora.update({k:b"".join(v) for k,v in groups.items()})
    return corpora


def cross_expert(sample_dir):
    base=Path(sample_dir); manifest=read_json(base/"manifest.json"); by_tensor=defaultdict(dict); rows=[]
    for row in manifest["samples"]:
        data=(base/row["file"]).read_bytes()
        if sha256(data)!=row["sha256"]: raise ValueError("sample checksum mismatch")
        by_tensor[row["tensor"]][row["expert"]]=data
    for name,pair in by_tensor.items():
        if set(pair)!={0,64}: continue
        a,b=pair[0],pair[64]
        if len(a)!=len(b): continue
        residual=np.bitwise_xor(np.frombuffer(a,np.uint8),np.frombuffer(b,np.uint8)).tobytes()
        base_blob=encode(a,"zstd"); residual_blob=encode(residual,"zstd")
        restored=np.bitwise_xor(np.frombuffer(decode(base_blob),np.uint8),np.frombuffer(decode(residual_blob),np.uint8)).tobytes()
        if restored!=b: raise AssertionError("cross-expert residual roundtrip failed")
        blocks_a={a[i:i+4096] for i in range(0,len(a),4096)}
        exact=sum(b[i:i+4096] in blocks_a for i in range(0,len(b),4096))
        independent=len(base_blob)+len(encode(b,"zstd"))
        dependent=len(base_blob)+len(residual_blob)+8  # include base-reference index
        rows.append(dict(tensor=name,bytes_per_sample=len(a),matching_4k_blocks=exact,
            blocks_compared=math.ceil(len(b)/4096),independent_stored_bytes=independent,
            base_plus_residual_bytes=dependent,additional_bytes=dependent-independent,roundtrip_exact=True,
            caveat="Dependency retrieval/decode costs would be additional; base included in storage total."))
    return provenance("measured-cross-expert",rows=rows,sample_manifest_sha256=sha256((base/"manifest.json").read_bytes()))


def compare(output="results",steps=64,seeds=(17,29,41)):
    root=Path(output); target=read_json(root/"model-metadata/summary.json")
    costs=measured_costs(read_json(root/"storage.json"),read_json(root/"gpu.json"))
    compression=read_json(root/"compression.json"); rows=[]; assumptions=[]
    n=target["expert_mean_serialized_bytes"]
    records_raw={(l,e):ExpertRecord(l,e,n,n) for l in range(target["layers"]) for e in range(target["experts_per_layer"])}
    # Weight the sampled packed blocks and scale streams by the ACTUAL tensor payload sizes,
    # not by their deliberately equal-sized samples. Biases are kept raw.
    headers=read_json(root/"model-metadata/headers.json")
    part_bytes=defaultdict(int)
    for shard in headers.values():
        for name,tensor in shard["tensors"].items():
            if ".experts." not in name: continue
            part="target-packed-blocks" if name.endswith("_blocks") else "target-scales" if name.endswith("_scales") else "bias"
            part_bytes[part]+=tensor["data_offsets"][1]-tensor["data_offsets"][0]
    profiles={}; scenarios=[("plain", "raw", records_raw)]
    if all(any(r["corpus"]==p for r in compression["rows"]) for p in ("target-packed-blocks","target-scales")):
        for codec_mode in ("zstd", "adaptive"):
            stored=0.0; decode_ns=0.0; profile={}
            for part,size in part_bytes.items():
                per_expert=size/(target["layers"]*target["experts_per_layer"])
                if part=="bias": stored+=per_expert; continue
                candidates=[r for r in compression["rows"] if r["corpus"]==part and r["tile_size"]==65536]
                if codec_mode=="zstd":
                    chosen=next(r for r in candidates if r["codec"]=="zstd" and r["transform"]=="none")
                else:
                    chosen=select_representation(candidates,costs.storage_bytes_per_ns,costs.h2d_bytes_per_ns)
                profile[part]=chosen; stored+=per_expert*chosen["ratio"]
                decode_ns+=per_expert*chosen["decode_ns_per_byte"]
            profiles[codec_mode]=dict(parts=profile,stored_expert_bytes=math.ceil(stored),decode_ns_per_expert=decode_ns)
            compressed={(l,e):ExpertRecord(l,e,n,math.ceil(stored),codec_mode,decode_ns=decode_ns) for l,e in records_raw}
            scenarios.append((codec_mode,"compressed",compressed))
            if codec_mode=="zstd": scenarios.append((codec_mode,"raw",compressed))
        assumptions.append("Whole-expert compressed size and decode time are LINEAR EXTRAPOLATIONS from a small nonrandom real-target sample, weighted by header-verified stream sizes; unvalidated on full experts.")
    else:
        assumptions.append("No real target samples available: compression scenarios omitted.")
    for workload in ("uniform","hotset","phase-shift"):
        for seed in seeds:
            events=list(generate(steps,target["layers"],target["experts_per_layer"],target["top_k"],seed,workload))
            trace_path=root/"traces"/f"{workload}-{seed}.jsonl"
            meta=save_trace(trace_path,events,dict(source="synthetic",seed=seed,workload=workload,steps=steps,
                layers=target["layers"],experts=target["experts_per_layer"],top_k=target["top_k"]))
            for policy in ("none","lru","lfu","cost"):
                for storage_profile,representation,records in scenarios:
                    result=replay(events,records,8*2**30,1*2**30,policy,costs,representation)
                    result.update(workload=workload,seed=seed,trace_sha256=meta["sha256"],storage_profile=storage_profile)
                    rows.append(result)
            write_json(str(trace_path)+".analysis.json",analyze(events))
    result=provenance("synthetic-comparison",rows=rows,steps=steps,seeds=list(seeds),
        expert_bytes=n,compression_profiles=profiles,assumptions=assumptions,
        reservations="Simulated 8 GiB host expert cache and 1 GiB GPU expert cache; leaves capacity for OS, non-expert weights, KV and staging. No real weights allocated by simulator.",
        input_hashes={f:sha256((root/f).read_bytes()) for f in ("storage.json","gpu.json","compression.json","model-metadata/summary.json")})
    write_json(root/"comparison.json",result)
    return result
