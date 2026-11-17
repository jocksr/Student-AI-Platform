"""Fetch bounded official metadata, never checkpoint payloads.

Range requests require HTTP 206 and an exact Content-Range. A server ignoring
Range is closed before reading its body. Total metadata cap is 8 MiB.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from apex.common import provenance, write_json

MAX_METADATA=8*2**20
consumed=0


def download(url,limit=1024*1024,byte_range=None):
    global consumed
    headers={"User-Agent":"APEX-research/0.1", "Accept-Encoding":"identity"}
    if byte_range: headers["Range"]=f"bytes={byte_range[0]}-{byte_range[1]}"
    request=urllib.request.Request(url,headers=headers)
    with urllib.request.urlopen(request,timeout=30) as response:
        if byte_range:
            a,b=byte_range
            if response.status!=206 or not re.fullmatch(rf"bytes {a}-{b}/\d+",response.headers.get("Content-Range","")):
                raise RuntimeError("server did not honor bounded Range request; body not read")
        data=response.read(limit+1)
        if len(data)>limit: raise RuntimeError("response exceeded size cap")
        if byte_range and len(data)!=byte_range[1]-byte_range[0]+1: raise RuntimeError("short range")
        consumed+=len(data)
        if consumed>MAX_METADATA: raise RuntimeError("metadata budget exceeded")
        return data


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",default="results/model-metadata")
    args=parser.parse_args(); dest=Path(args.out); dest.mkdir(parents=True,exist_ok=True)
    repo="https://huggingface.co/openai/gpt-oss-120b"
    info=json.loads(download("https://huggingface.co/api/models/openai/gpt-oss-120b"))
    revision=info["sha"]
    if not re.fullmatch("[0-9a-f]{40}",revision): raise ValueError("bad revision")
    base=f"{repo}/resolve/{revision}"
    config=json.loads(download(base+"/config.json"))
    index=json.loads(download(base+"/model.safetensors.index.json"))
    write_json(dest/"config.json",config); write_json(dest/"index.json",index)
    shards=sorted(set(index["weight_map"].values())); all_headers={}; tensor_bytes=0; expert_bytes=0; params=0; expert_params=0
    for filename in shards:
        url=base+"/"+filename
        header_size=struct.unpack("<Q",download(url,8,(0,7)))[0]
        if not 1<=header_size<=512*1024: raise ValueError("oversized header")
        # Query distinguishes proxy caches that ignore Range in their cache key.
        header=json.loads(download(url+"?apex_header=1",header_size,(8,7+header_size)))
        all_headers[filename]=dict(header_bytes=header_size,tensors=header)
        for name,tensor in header.items():
            if name=="__metadata__": continue
            n=tensor["data_offsets"][1]-tensor["data_offsets"][0]
            tensor_bytes+=n; is_expert=".experts." in name
            if is_expert: expert_bytes+=n
            if name.endswith("_scales"): continue
            elements=1
            for d in tensor["shape"]: elements*=d
            if name.endswith("_blocks"): elements*=2
            params+=elements
            if is_expert: expert_params+=elements
        print(filename,header_size,flush=True)
    write_json(dest/"headers.json",all_headers)
    summary=provenance("official-model-metadata",repo=repo,revision=revision,shards=len(shards),
        downloaded_metadata_bytes=consumed,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        layers=config["num_hidden_layers"],experts_per_layer=config["num_local_experts"],top_k=config["num_experts_per_tok"],
        tensor_payload_bytes=tensor_bytes,expert_payload_bytes=expert_bytes,nonexpert_payload_bytes=tensor_bytes-expert_bytes,
        serialized_checkpoint_bytes=tensor_bytes+sum(x["header_bytes"]+8 for x in all_headers.values()),
        logical_parameters_from_shapes=params,logical_expert_parameters=expert_params,
        logical_nonexpert_parameters=params-expert_params,
        expert_mean_serialized_bytes=expert_bytes//(config["num_hidden_layers"]*config["num_local_experts"]),
        parameter_count_method="Two logical FP4 values per uint8 _blocks element; _scales excluded; all other elements counted once. Includes biases.",
        index_metadata=index.get("metadata"),weight_payload_downloaded=False)
    write_json(dest/"summary.json",summary); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
