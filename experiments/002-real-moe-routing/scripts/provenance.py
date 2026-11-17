"""Capture installed binary and source identities without accessing credentials."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import psutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(8*2**20): value.update(data)
    return value.hexdigest()


def command(args):
    process = subprocess.run(args,capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
    return dict(args=args,returncode=process.returncode,stdout=process.stdout,stderr=process.stderr)


def capture():
    binaries = [dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p))
        for p in sorted((ROOT/"runtime").glob("*")) if p.suffix in (".dll",".exe")]
    source_files = list((ROOT/"scripts").glob("*.py"))+list((ROOT/"fixtures").glob("*.json"))
    sources = [dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(source_files)]
    result = dict(utc=datetime.now(timezone.utc).isoformat(), binaries=binaries,sources=sources,
        engine=command([str(ROOT/"runtime/llama-server.exe"),"--version"]),
        gpu=command(["nvidia-smi","--query-gpu=name,memory.total,driver_version,pci.bus_id","--format=csv,noheader"]),
        python=platform.python_version(),psutil=psutil.__version__,platform=platform.platform(),
        cpu_physical=psutil.cpu_count(logical=False),cpu_logical=psutil.cpu_count(),
        memory=psutil.virtual_memory()._asdict())
    (ROOT/"results/setup/environment.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("Recorded",len(binaries),"binary hashes and",len(sources),"source hashes")


if __name__ == "__main__": capture()
