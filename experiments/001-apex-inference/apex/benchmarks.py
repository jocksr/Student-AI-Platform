"""Bounded measurements. Disk benchmarks touch only a new temporary file."""
import ctypes as C
import os
from pathlib import Path
import random
import tempfile
import time
import numpy as np
import psutil
from .common import provenance, stats


def memory_benchmark(mib=64, repeats=9):
    n = mib * 2**20
    if not 1 <= mib <= 256 or n*3 > psutil.virtual_memory().available//4:
        raise ValueError("RAM benchmark exceeds the conservative allocation limit")
    a = np.full(n, 123, dtype=np.uint8)
    b = np.zeros_like(a)
    np.copyto(b, a)
    times = []
    for _ in range(repeats):
        t = time.perf_counter_ns(); np.copyto(b, a); times.append(time.perf_counter_ns()-t)
    assert np.array_equal(a,b)
    summary = stats(times)
    return provenance("measured-memory-copy", bytes_per_array=n, **summary,
                      payload_GB_s=n/summary["median_ns"],
                      logical_read_plus_write_GB_s=2*n/summary["median_ns"],
                      available_ram_after=psutil.virtual_memory().available,
                      conditions="Single-thread NumPy copy; warm pages; includes read and write. Not STREAM or theoretical channel bandwidth.")


class DirectReader:
    """Windows synchronous, queue-depth-one, OS-cache-bypassing file reads."""
    def __init__(self, path, max_block):
        from ctypes import wintypes as W
        self.k = C.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateFileW": ([W.LPCWSTR,W.DWORD,W.DWORD,C.c_void_p,W.DWORD,W.DWORD,W.HANDLE],W.HANDLE),
            "VirtualAlloc": ([C.c_void_p,C.c_size_t,W.DWORD,W.DWORD],C.c_void_p),
            "VirtualFree": ([C.c_void_p,C.c_size_t,W.DWORD],W.BOOL),
            "ReadFile": ([W.HANDLE,C.c_void_p,W.DWORD,C.POINTER(W.DWORD),C.c_void_p],W.BOOL),
            "SetFilePointerEx": ([W.HANDLE,C.c_longlong,C.c_void_p,W.DWORD],W.BOOL),
            "CloseHandle": ([W.HANDLE],W.BOOL),
        }
        for name,(args,ret) in signatures.items():
            f=getattr(self.k,name); f.argtypes=args; f.restype=ret
        self.handle=self.k.CreateFileW(str(Path(path).resolve()),0x80000000,1,None,3,0x20000000,None)
        if self.handle == C.c_void_p(-1).value: raise C.WinError(C.get_last_error())
        self.buffer=self.k.VirtualAlloc(None,max_block,0x3000,4)
        if not self.buffer:
            self.k.CloseHandle(self.handle); raise C.WinError(C.get_last_error())
    def read(self, offset, size):
        got=C.c_ulong()
        if not self.k.SetFilePointerEx(self.handle,offset,None,0): raise C.WinError(C.get_last_error())
        if not self.k.ReadFile(self.handle,self.buffer,size,C.byref(got),None): raise C.WinError(C.get_last_error())
        if got.value != size: raise IOError("short direct read")
    def close(self):
        self.k.CloseHandle(self.handle); self.k.VirtualFree(self.buffer,0,0x8000)


def storage_benchmark(directory, mib=128, seed=17):
    if not 16 <= mib <= 512: raise ValueError("storage fixture must be 16..512 MiB")
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    n=mib*2**20
    if shutil_free(directory) < n*2: raise ValueError("insufficient free space")
    result=provenance("measured-storage", file_bytes=n, seed=seed, directory=str(directory.resolve()),
                      queue_depth=1, rows=[], conditions="One pseudorandom file, written and fsynced once. OS cached and Windows NO_BUFFERING results separated. Device cache remains possible; not cold-drive certification.")
    with tempfile.TemporaryDirectory(prefix="apex-storage-",dir=directory) as tmp:
        path=Path(tmp)/"fixture.bin"
        rng=random.Random(seed)
        with path.open("wb",buffering=0) as f:
            for _ in range(mib): f.write(rng.randbytes(2**20))
            f.flush(); os.fsync(f.fileno())
        for mode in ("buffered_warm","windows_unbuffered"):
            rng=random.Random(seed+1)
            if mode=="windows_unbuffered" and os.name!="nt": continue
            reader=None
            try:
                reader=DirectReader(path,2**20) if mode=="windows_unbuffered" else path.open("rb",buffering=0)
                buf=bytearray(2**20)
                if mode=="buffered_warm":
                    while reader.readinto(buf): pass
                for pattern,block,count in (("sequential",2**20,mib),("random",4096,128),("random",2**20,64)):
                    offsets=list(range(0,n,block)) if pattern=="sequential" else [rng.randrange(n//block)*block for _ in range(count)]
                    samples=[]
                    for offset in offsets:
                        start=time.perf_counter_ns()
                        if mode=="windows_unbuffered": reader.read(offset,block)
                        else:
                            reader.seek(offset)
                            if reader.readinto(memoryview(buf)[:block]) != block: raise IOError("short buffered read")
                        samples.append(time.perf_counter_ns()-start)
                    result["rows"].append(dict(mode=mode,pattern=pattern,block_bytes=block,
                                               bytes_read=block*len(offsets),read_count=len(offsets),
                                               throughput_GB_s=block*len(offsets)/sum(samples),**stats(samples)))
            except OSError as e:
                result["rows"].append(dict(mode=mode,status="unavailable",reason=str(e)))
            finally:
                if reader: reader.close()
    return result


def shutil_free(path):
    import shutil
    return shutil.disk_usage(path).free
