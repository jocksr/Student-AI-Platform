import shutil
import subprocess
import numpy as np
import psutil
from .common import provenance


def inventory():
    features = np._core._multiarray_umath.__cpu_features__
    result = provenance("hardware", ram=psutil.virtual_memory()._asdict(),
                        physical_cores=psutil.cpu_count(logical=False),
                        logical_cpus=psutil.cpu_count(),
                        cpu_features={k: bool(v) for k,v in features.items()},
                        tools={x: shutil.which(x) for x in
                               ("python", "git", "nvcc", "nvidia-smi", "cl", "cmake", "wsl", "docker")})
    query = "name,memory.total,memory.free,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max"
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu="+query, "--format=csv"],
                           capture_output=True, text=True, timeout=20)
        result["nvidia_smi"] = dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except (OSError, subprocess.TimeoutExpired) as e:
        result["nvidia_smi"] = dict(error=str(e))
    result["caveats"] = ["Current PCIe generation can downshift at idle. nvidia-smi link maxima are driver-reported, not product specifications.",
                         "CUDA version in nvidia-smi is driver capability, not evidence of an installed toolkit."]
    return result
