"""Explicitly bounded real-weight experiment: <=4 MiB, never full shards."""
import json
from pathlib import Path
from fetch_model_metadata import download
from apex.common import provenance,write_json,sha256


def main():
    root=Path("results/model-metadata"); summary=json.loads((root/"summary.json").read_text())
    headers=json.loads((root/"headers.json").read_text()); dest=Path("benchmarks/weight-samples")
    dest.mkdir(parents=True,exist_ok=True)
    rows=[]; total=0
    for shard,entry in headers.items():
        for name,tensor in entry["tensors"].items():
            if not any(name.startswith(f"model.layers.{layer}.mlp.experts.") for layer in (0,18,35)): continue
            if not name.endswith(("_blocks","_scales")): continue
            if tensor["shape"][0]!=128: raise ValueError("unexpected expert axis")
            start,end=tensor["data_offsets"]
            stride=(end-start)//128
            if stride*128!=end-start: raise ValueError("uneven expert layout")
            for expert in (0,64):
                size=min(131072,stride); offset=8+entry["header_bytes"]+start+expert*stride
                if total+size>4*2**20: raise ValueError("sample cap exceeded")
                filename=f"{name}.expert{expert}.bin"
                url=f"{summary['repo']}/resolve/{summary['revision']}/{shard}?apex_sample={offset}"
                data=download(url,size,(offset,offset+size-1))
                (dest/filename).write_bytes(data); total+=len(data)
                rows.append(dict(file=filename,tensor=name,shard=shard,expert=expert,offset=offset,
                                 bytes=size,sha256=sha256(data),dtype=tensor["dtype"],tensor_shape=tensor["shape"]))
    write_json(dest/"manifest.json",provenance("real-target-weight-samples",revision=summary["revision"],
        repo=summary["repo"],total_payload_bytes=total,samples=rows,
        sampling="First 128 KiB (or full shorter stride) of gate/up and down packed blocks and scales for experts 0 and 64 of layers 0, 18, 35. Deliberately small, nonrandom sample; not whole-model representative."))
    print(f"Saved {total} bytes in {len(rows)} samples")


if __name__=="__main__": main()
