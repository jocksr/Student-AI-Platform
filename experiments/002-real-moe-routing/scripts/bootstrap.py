"""Read official release/checkpoint metadata and prepare pinned download plans."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import subprocess
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]

def get_json(url):
    request=urllib.request.Request(url,headers={"User-Agent":"Student-AI-Platform-Experiment-002","Accept":"application/json"})
    with urllib.request.urlopen(request,timeout=30) as r:
        data=r.read(4*2**20+1)
    if len(data)>4*2**20: raise ValueError("metadata response too large")
    return json.loads(data)

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")

def metadata():
    release=get_json("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
    channel_release=release
    if [a["name"] for a in release["assets"]]==["nightly-tag.txt"]:
        with urllib.request.urlopen(release["assets"][0]["browser_download_url"],timeout=30) as r:
            tag=r.read(128).decode().strip()
        if not tag.startswith("b") or not tag[1:].isdigit(): raise ValueError("unexpected nightly tag")
        release=get_json("https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/"+tag)
    model=get_json("https://huggingface.co/api/models/ggml-org/gpt-oss-20b-GGUF?blobs=true")
    save(ROOT/"results/setup/engine-release.json",release)
    save(ROOT/"results/setup/model-repository.json",model)
    wanted=next(s for s in model["siblings"] if s["rfilename"]=="gpt-oss-20b-MXFP4.gguf")
    config=get_json("https://huggingface.co/openai/gpt-oss-20b/resolve/main/config.json")
    save(ROOT/"results/setup/model-config.json",config)
    manifest=dict(utc=datetime.now(timezone.utc).isoformat(),model_revision=model["sha"],
        model_file=wanted,model_url=f"https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/{model['sha']}/{wanted['rfilename']}",
        engine_channel=channel_release["tag_name"],engine_tag=release["tag_name"],engine_target=release["target_commitish"],engine_published_at=release["published_at"],
        engine_assets=[{k:a.get(k) for k in ("name","size","browser_download_url","digest")} for a in release["assets"]])
    save(ROOT/"results/setup/download-plan.json",manifest)
    print(json.dumps(manifest,indent=2))

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while block:=f.read(8*2**20): h.update(block)
    return h.hexdigest()

def download_one(url,path,size,expected_hash):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.stat().st_size!=size or sha256(path)!=expected_hash: raise ValueError(f"existing file does not match: {path}")
        return dict(path=str(path.relative_to(ROOT)),bytes=size,sha256=expected_hash,reused=True)
    part=path.with_suffix(path.suffix+".part")
    cmd=["curl.exe","--fail","--location","--retry","3","--retry-delay","2","--continue-at","-", "--silent","--show-error","--output",str(part),url]
    p=subprocess.Popen(cmd,creationflags=subprocess.CREATE_NO_WINDOW)
    last=0
    while p.poll() is None:
        now=time.monotonic()
        if now-last>10:
            current=part.stat().st_size if part.exists() else 0
            print(f"{path.name}: {current/2**30:.2f}/{size/2**30:.2f} GiB",flush=True); last=now
        time.sleep(1)
    if p.returncode: raise RuntimeError(f"curl failed for {path.name}: {p.returncode}")
    if part.stat().st_size!=size: raise ValueError("download size mismatch")
    actual=sha256(part)
    if actual!=expected_hash: raise ValueError(f"SHA-256 mismatch for {path.name}")
    part.replace(path)
    print(f"verified {path.name}: {actual}",flush=True)
    return dict(path=str(path.relative_to(ROOT)),bytes=size,sha256=actual,reused=False)

def downloads():
    plan=json.loads((ROOT/"results/setup/download-plan.json").read_text())
    assets=[a for a in plan["engine_assets"] if ("win-cuda-13" in a["name"] and a["name"].endswith("x64.zip")) or ("cudart-llama-bin-win-cuda-13" in a["name"] and a["name"].endswith("x64.zip"))]
    if not assets: raise ValueError("No CUDA 13 Windows x64 release assets; review download plan")
    total=plan["model_file"]["size"]+sum(a["size"] for a in assets)
    if shutil.disk_usage(ROOT).free<total+4*2**30: raise ValueError("insufficient free disk")
    jobs=[(plan["model_url"],ROOT/"models"/plan["model_file"]["rfilename"],plan["model_file"]["size"],plan["model_file"]["lfs"]["sha256"])]
    for a in assets:
        if not (a.get("digest") or "").startswith("sha256:"): raise ValueError("release asset missing trusted digest")
        jobs.append((a["browser_download_url"],ROOT/"downloads"/a["name"],a["size"],a["digest"].split(":",1)[1]))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(download_one,*j) for j in jobs]
        results=[f.result() for f in futures]
    dest=ROOT/"runtime"; dest.mkdir(exist_ok=True)
    for a in assets:
        with zipfile.ZipFile(ROOT/"downloads"/a["name"]) as z:
            for member in z.infolist():
                resolved=(dest/member.filename).resolve()
                if not resolved.is_relative_to(dest.resolve()) or (member.external_attr>>16)&0o170000==0o120000:
                    raise ValueError("unsafe zip member")
            z.extractall(dest)
    save(ROOT/"results/setup/downloads.json",dict(utc=datetime.now(timezone.utc).isoformat(),files=results))
    print("Downloads verified and runtime extracted",flush=True)

def sources():
    plan=json.loads((ROOT/"results/setup/download-plan.json").read_text())
    records=[]
    for name in ("docs/moe-cache.md","scripts/moe-cache-tune.py","src/llama-moe-cache.cpp","tools/server/README.md","LICENSE"):
        url=f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{plan['engine_target']}/{name}"
        with urllib.request.urlopen(url,timeout=45) as r: data=r.read(8*2**20)
        path=ROOT/"runtime/upstream"/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
        records.append(dict(url=url,path=str(path.relative_to(ROOT)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    save(ROOT/"results/setup/source-manifest.json",records)
    print("Pinned upstream reference sources saved",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("action",choices=["metadata","download","sources"],default="metadata",nargs="?")
    args=parser.parse_args()
    {"metadata":metadata,"download":downloads,"sources":sources}[args.action]()
