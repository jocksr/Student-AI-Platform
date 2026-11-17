"""Small CUDA Driver API profiler using the installed driver, without a toolkit.

The PTX GEMM is intentionally a plain SIMT reference, not an MXFP4 Tensor Core
kernel. Its timing must not be used to predict production MoE compute speed.
"""
import ctypes as C
import os
import time
import numpy as np
from .common import provenance, stats

PTX = r'''
.version 7.0
.target sm_80
.address_size 64
.visible .entry gemm(.param .u64 A, .param .u64 B, .param .u64 O,
                    .param .u32 K, .param .u32 N, .param .u32 TOTAL) {
 .reg .pred %p<3>;
 .reg .b32 %r<16>;
 .reg .b64 %d<12>;
 .reg .f32 %f<4>;
 ld.param.u64 %d1,[A]; ld.param.u64 %d2,[B]; ld.param.u64 %d3,[O];
 ld.param.u32 %r1,[K]; ld.param.u32 %r2,[N]; ld.param.u32 %r3,[TOTAL];
 mov.u32 %r4,%ctaid.x; mov.u32 %r5,%ntid.x; mov.u32 %r6,%tid.x;
 mad.lo.u32 %r7,%r4,%r5,%r6;
 setp.ge.u32 %p1,%r7,%r3; @%p1 bra DONE;
 div.u32 %r8,%r7,%r2; rem.u32 %r9,%r7,%r2;
 mul.lo.u32 %r10,%r8,%r1; mov.u32 %r11,0; mov.f32 %f1,0f00000000;
 LOOP:
 add.u32 %r12,%r10,%r11; mul.wide.u32 %d4,%r12,4; add.u64 %d5,%d1,%d4;
 mad.lo.u32 %r13,%r11,%r2,%r9; mul.wide.u32 %d6,%r13,4; add.u64 %d7,%d2,%d6;
 ld.global.f32 %f2,[%d5]; ld.global.f32 %f3,[%d7]; fma.rn.f32 %f1,%f2,%f3,%f1;
 add.u32 %r11,%r11,1; setp.lt.u32 %p2,%r11,%r1; @%p2 bra LOOP;
 mul.wide.u32 %d8,%r7,4; add.u64 %d9,%d3,%d8; st.global.f32 [%d9],%f1;
 DONE: ret;
}
'''


class Driver:
    def __init__(self):
        self.lib=C.WinDLL("nvcuda.dll") if os.name=="nt" else C.CDLL("libcuda.so.1")
        self.allocated=[]; self.pinned=[]; self.modules=[]
        self.call("cuInit",[C.c_uint],0)
        self.device=C.c_int()
        self.call("cuDeviceGet",[C.POINTER(C.c_int),C.c_int],C.byref(self.device),0)
        self.context=C.c_void_p()
        self.call("cuDevicePrimaryCtxRetain",[C.POINTER(C.c_void_p),C.c_int],C.byref(self.context),self.device)
        self.call("cuCtxSetCurrent",[C.c_void_p],self.context)
    def call(self,name,types,*args):
        f=getattr(self.lib,name); f.argtypes=types; f.restype=C.c_int
        status=f(*args)
        if status: raise RuntimeError(f"{name}: CUDA status {status}")
    def alloc(self,n):
        p=C.c_uint64()
        self.call("cuMemAlloc_v2",[C.POINTER(C.c_uint64),C.c_size_t],C.byref(p),n)
        self.allocated.append(p); return p
    def free(self,p):
        self.call("cuMemFree_v2",[C.c_uint64],p); self.allocated.remove(p)
    def host(self,n):
        p=C.c_void_p()
        self.call("cuMemAllocHost_v2",[C.POINTER(C.c_void_p),C.c_size_t],C.byref(p),n)
        self.pinned.append(p); return p
    def copy_h2d(self,d,h,n):
        self.call("cuMemcpyHtoD_v2",[C.c_uint64,C.c_void_p,C.c_size_t],d,h,n)
    def copy_d2h(self,h,d,n):
        self.call("cuMemcpyDtoH_v2",[C.c_void_p,C.c_uint64,C.c_size_t],h,d,n)
    def sync(self): self.call("cuCtxSynchronize",[])
    def memory(self):
        free=C.c_size_t(); total=C.c_size_t()
        self.call("cuMemGetInfo_v2",[C.POINTER(C.c_size_t)]*2,C.byref(free),C.byref(total))
        return dict(free=free.value,total=total.value)
    def attribute(self,attr):
        v=C.c_int()
        self.call("cuDeviceGetAttribute",[C.POINTER(C.c_int),C.c_int,C.c_int],C.byref(v),attr,self.device)
        return v.value
    def gemm_function(self):
        module=C.c_void_p(); code=C.create_string_buffer(PTX.encode())
        self.call("cuModuleLoadData",[C.POINTER(C.c_void_p),C.c_void_p],C.byref(module),C.cast(code,C.c_void_p))
        self.modules.append(module); fn=C.c_void_p()
        self.call("cuModuleGetFunction",[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p],C.byref(fn),module,b"gemm")
        return fn
    def launch(self,fn,a,b,o,k,n,total):
        args=[a,b,o,C.c_uint(k),C.c_uint(n),C.c_uint(total)]
        pointers=(C.c_void_p*len(args))(*(C.cast(C.byref(x),C.c_void_p) for x in args))
        self.call("cuLaunchKernel",[C.c_void_p]+[C.c_uint]*7+[C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p],
                  fn,(total+127)//128,1,1,128,1,1,0,None,pointers,None)
    def close(self):
        for p in list(self.allocated): self.free(p)
        for p in self.pinned: self.call("cuMemFreeHost",[C.c_void_p],p)
        for p in self.modules: self.call("cuModuleUnload",[C.c_void_p],p)
        self.call("cuDevicePrimaryCtxRelease_v2",[C.c_int],self.device)


def benchmark_gpu(mib=32,repeats=7):
    if not 1<=mib<=64: raise ValueError("GPU transfer fixture limited to 1..64 MiB")
    result=provenance("measured-cuda-driver",transfer_rows=[],allocation_rows=[],compute_rows=[],
                      conditions="Synchronous wall-clock operations in a private process; includes API overhead and synchronization; no Tensor Core or MXFP4 claim.")
    driver=None
    try:
        start=time.perf_counter_ns(); driver=Driver()
        result["context_init_ns"]=time.perf_counter_ns()-start
        result["vram_after_init"]=driver.memory()
        result["compute_capability"]=[driver.attribute(75),driver.attribute(76)]
        if result["vram_after_init"]["free"] < 256*2**20: raise RuntimeError("less than 256 MiB free VRAM")
        for n in (4096,mib*2**20):
            times=[]
            for _ in range(repeats):
                t=time.perf_counter_ns(); p=driver.alloc(n); driver.sync(); times.append(time.perf_counter_ns()-t); driver.free(p)
            result["allocation_rows"].append(dict(bytes=n,**stats(times)))
            device=driver.alloc(n)
            pageable=np.full(n,91,dtype=np.uint8)
            pinned=driver.host(n); C.memset(pinned,91,n)
            for kind,host in (("pageable",C.c_void_p(pageable.ctypes.data)),("pinned",pinned)):
                for direction in ("h2d","d2h"):
                    action=(lambda:driver.copy_h2d(device,host,n)) if direction=="h2d" else (lambda:driver.copy_d2h(host,device,n))
                    if direction=="d2h": C.memset(host,0,n)
                    action(); driver.sync(); times=[]
                    for _ in range(repeats):
                        t=time.perf_counter_ns(); action(); driver.sync(); times.append(time.perf_counter_ns()-t)
                    s=stats(times)
                    result["transfer_rows"].append(dict(host_memory=kind,direction=direction,bytes=n,GB_s=n/s["median_ns"],**s))
                restored=np.ctypeslib.as_array((C.c_ubyte*n).from_address(host.value))
                if not np.all(restored==91): raise AssertionError("GPU transfer verification failed")
            driver.free(device)
        fn=driver.gemm_function(); rng=np.random.default_rng(17)
        # One expert projection has the target's 2880 x 2880 dimensions.
        # CPU and GPU both use FP32 here; target weights are MXFP4, so costs are illustrative.
        for k in (512,2880):
            b=rng.normal(0,0.05,(k,k)).astype(np.float32); db=driver.alloc(b.nbytes)
            driver.copy_h2d(db,C.c_void_p(b.ctypes.data),b.nbytes)
            for batch in (1,16):
                a=rng.normal(0,0.05,(batch,k)).astype(np.float32); out=np.empty((batch,k),np.float32)
                da=driver.alloc(a.nbytes); do=driver.alloc(out.nbytes)
                driver.copy_h2d(da,C.c_void_p(a.ctypes.data),a.nbytes)
                expected=a@b; cpu=[]; gpu=[]
                driver.launch(fn,da,db,do,k,k,batch*k); driver.sync()
                for _ in range(5):
                    t=time.perf_counter_ns(); np.matmul(a,b,out=out); cpu.append(time.perf_counter_ns()-t)
                    t=time.perf_counter_ns(); driver.launch(fn,da,db,do,k,k,batch*k); driver.sync(); gpu.append(time.perf_counter_ns()-t)
                driver.copy_d2h(C.c_void_p(out.ctypes.data),do,out.nbytes)
                np.testing.assert_allclose(out,expected,rtol=1e-3,atol=1e-5)
                result["compute_rows"].append(dict(batch=batch,k=k,n=k,dtype="float32",weight_bytes=b.nbytes,
                    cpu_numpy=stats(cpu),gpu_naive_ptx=stats(gpu),max_abs_error=float(np.max(np.abs(out-expected))),
                    caveat="Resident projection only, not complete gated expert. CPU BLAS thread policy is library default. GPU kernel is unoptimized SIMT."))
                driver.free(da); driver.free(do)
            driver.free(db)
        result["status"]="ok"
    except (OSError,RuntimeError) as e:
        result["status"]="unavailable-or-partial"; result["reason"]=str(e)
    finally:
        if driver: driver.close()
    return result
