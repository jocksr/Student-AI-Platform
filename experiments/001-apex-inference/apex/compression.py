"""Independently decodable, checksummed tiles with reversible byte transforms."""
from dataclasses import dataclass
import gzip
import struct
import time
import tracemalloc
import zlib
import numpy as np
from .common import provenance, sha256

HEADER=struct.Struct("<4sBBHIII")
MAX_TILE=16*2**20
CODECS={"raw":0,"gzip":1,"lz4":2,"zstd":3}
TRANSFORMS={"none":0,"shuffle2":1,"bitplane":2,"xor":3}


def available_codecs():
    result=["raw","gzip"]
    try:
        import lz4.block
        result.append("lz4")
    except ImportError: pass
    try:
        import zstandard
        result.append("zstd")
    except ImportError: pass
    return result


def transform(data,name,inverse=False):
    if name=="none": return data
    a=np.frombuffer(data,dtype=np.uint8)
    if name=="xor":
        if inverse: return np.bitwise_xor.accumulate(a).tobytes()
        if not len(a): return b""
        return np.concatenate((a[:1],np.bitwise_xor(a[1:],a[:-1]))).tobytes()
    if name=="shuffle2":
        even=len(a)//2*2
        if inverse: return a[:even].reshape(2,-1).T.copy().tobytes()+a[even:].tobytes()
        return a[:even].reshape(-1,2).T.copy().tobytes()+a[even:].tobytes()
    if name=="bitplane":
        bits=np.unpackbits(a)
        if inverse: bits=bits.reshape(8,-1).T.reshape(-1)
        else: bits=bits.reshape(-1,8).T.reshape(-1)
        return np.packbits(bits).tobytes()
    raise ValueError("unknown transform")


def encode(data,codec="raw",transform_name="none"):
    if len(data)>MAX_TILE: raise ValueError("tile too large")
    if codec not in CODECS or transform_name not in TRANSFORMS: raise ValueError("unknown representation")
    prepared=transform(data,transform_name)
    if codec=="raw": payload=prepared
    elif codec=="gzip": payload=gzip.compress(prepared,compresslevel=1,mtime=0)
    elif codec=="lz4":
        import lz4.block
        payload=lz4.block.compress(prepared,store_size=False)
    else:
        import zstandard as zstd
        payload=zstd.ZstdCompressor(level=1).compress(prepared)
    return HEADER.pack(b"APX1",CODECS[codec],TRANSFORMS[transform_name],0,len(data),len(payload),zlib.crc32(data))+payload


def decode(tile):
    if len(tile)<HEADER.size: raise ValueError("truncated tile")
    magic,cid,tid,reserved,n,payload_n,crc=HEADER.unpack_from(tile)
    if magic!=b"APX1" or reserved or n>MAX_TILE or payload_n!=len(tile)-HEADER.size:
        raise ValueError("invalid tile header")
    codec=next((k for k,v in CODECS.items() if v==cid),None)
    trans=next((k for k,v in TRANSFORMS.items() if v==tid),None)
    if codec is None or trans is None: raise ValueError("unknown codec/transform")
    payload=tile[HEADER.size:]
    try:
        if codec=="raw": prepared=payload
        elif codec=="gzip":
            d=zlib.decompressobj(31); prepared=d.decompress(payload,n+1)
            if not d.eof or d.unused_data: raise ValueError("invalid gzip stream")
        elif codec=="lz4":
            import lz4.block
            prepared=lz4.block.decompress(payload,uncompressed_size=n)
        else:
            import zstandard as zstd
            prepared=zstd.ZstdDecompressor().decompress(payload,max_output_size=max(1,n))
    except Exception as e:
        raise ValueError(f"{codec} decode failed") from e
    if len(prepared)!=n: raise ValueError("decoded length mismatch")
    raw=transform(prepared,trans,True)
    if zlib.crc32(raw)!=crc: raise ValueError("tile checksum mismatch")
    return raw


def synthetic_corpora(n=2**20,seed=17):
    rng=np.random.default_rng(seed)
    # Test controls, never claimed to follow the actual target distribution.
    random_bytes=rng.integers(0,256,n,dtype=np.uint8).tobytes()
    weights=np.array([1,5,12,16,12,7,3,1,1,5,12,16,12,7,3,1])
    nibble=rng.choice(np.arange(16,dtype=np.uint8),size=n*2,p=weights/weights.sum())
    packed=(nibble[::2]|(nibble[1::2]<<4)).tobytes()
    floats=rng.normal(0,0.02,n//2).astype(np.float32).view(np.uint32)
    bf16=(floats>>16).astype("<u2").tobytes()
    return {"random-control":random_bytes,"synthetic-packed-nibbles":packed,
            "synthetic-bf16":bf16,"zero-control":bytes(n)}


def benchmark(corpora=None,tile_sizes=(16384,65536,262144),seed=17,repeats=3):
    if corpora is None: corpora=synthetic_corpora(seed=seed)
    rows=[]
    for corpus,data in corpora.items():
        if not data: raise ValueError("empty benchmark corpus")
        for tile_size in tile_sizes:
            if not 1<=tile_size<=MAX_TILE: raise ValueError("invalid tile size")
            tiles=[data[i:i+tile_size] for i in range(0,len(data),tile_size)]
            for codec in available_codecs():
                for trans in TRANSFORMS:
                    if codec=="raw" and trans!="none": continue
                    encoded=[]; encode_times=[]; decode_times=[]; cpu_times=[]
                    for tile in tiles:
                        e=[]; d=[]; cpu=[]
                        for _ in range(repeats):
                            t=time.perf_counter_ns(); blob=encode(tile,codec,trans); e.append(time.perf_counter_ns()-t)
                            c=time.process_time_ns(); t=time.perf_counter_ns(); restored=decode(blob)
                            d.append(time.perf_counter_ns()-t); cpu.append(time.process_time_ns()-c)
                            if restored!=tile: raise AssertionError("bit-identical round trip failed")
                        encoded.append(blob); encode_times.append(float(np.median(e))); decode_times.append(float(np.median(d))); cpu_times.append(float(np.median(cpu)))
                    # Separate untimed allocation observation; tracemalloc excludes some native allocations.
                    tracemalloc.start(); decode(encoded[0]); _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
                    # Each tile envelope is 20 bytes. Independent offset index uses uint64 per tile plus terminal offset.
                    index_bytes=8*(len(tiles)+1); stored=sum(map(len,encoded))+index_bytes
                    rows.append(dict(corpus=corpus,corpus_sha256=sha256(data),raw_bytes=len(data),tile_size=tile_size,
                        tile_count=len(tiles),codec=codec,transform=trans,stored_bytes=stored,index_bytes=index_bytes,
                        ratio=stored/len(data),encode_ns=sum(encode_times),decode_ns=sum(decode_times),
                        decode_ns_per_byte=sum(decode_times)/len(data),decode_cpu_ns=sum(cpu_times),
                        first_tile_python_peak_bytes=peak,roundtrip_exact=True,gpu_decode_implemented=False))
    return provenance("measured-lossless-compression",seed=seed,repeats=repeats,codecs=available_codecs(),rows=rows,
        conditions="Median per-tile wall-clock, summed across corpus. Decode includes transform and CRC. Envelope and uint64 offset index included in sizes. Allocation probe is separate and excludes some native allocations. Adaptive selection is offline, not free.")


def select_representation(rows,storage_GB_s,h2d_GB_s,expected_reuses=1,memory_penalty_ns_per_byte=0):
    if min(storage_GB_s,h2d_GB_s)<=0 or expected_reuses<1 or memory_penalty_ns_per_byte<0:
        raise ValueError("invalid cost objective")
    if not rows: raise ValueError("no candidates")
    # Decode on CPU: PCIe always carries restored serialized bytes in this track.
    def cost(r):
        return r["stored_bytes"]/storage_GB_s + expected_reuses*(r["decode_ns"]+r["raw_bytes"]/h2d_GB_s) + r["stored_bytes"]*memory_penalty_ns_per_byte
    best=min(rows,key=cost)
    return dict(best,estimated_access_ns=cost(best),selection="offline measured total access cost")
