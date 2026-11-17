import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


def provenance(kind, **kwargs):
    versions = {}
    for name in ("numpy", "psutil", "torch", "lz4", "zstandard"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    root=Path(__file__).resolve().parents[1]
    digest=hashlib.sha256()
    for folder in ("apex","scripts","tests"):
        for path in sorted((root/folder).glob("*.py")):
            digest.update(str(path.relative_to(root)).replace("\\","/").encode()); digest.update(path.read_bytes())
    return dict(schema_version=1, kind=kind, utc=datetime.now(timezone.utc).isoformat(),source_sha256=digest.hexdigest(),
                python=sys.version, executable=sys.executable, platform=platform.platform(),
                packages=versions, **kwargs)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def stats(samples):
    values = sorted(samples)
    if not values:
        raise ValueError("at least one observation is required")
    return dict(samples_ns=samples, median_ns=statistics.median(values),
                p95_ns=values[min(len(values)-1, int(len(values)*0.95))], min_ns=values[0])
