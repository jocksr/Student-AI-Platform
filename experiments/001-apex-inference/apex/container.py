"""Small indexed tile container. Each read decodes only its requested tile."""
import struct
from pathlib import Path
from .compression import encode,decode,MAX_TILE

PREFIX=struct.Struct("<4sI")
OFFSET=struct.Struct("<Q")


def write_container(path,tiles,codec="raw",transform_name="none"):
    blobs=[encode(tile,codec,transform_name) for tile in tiles]
    offset=PREFIX.size+OFFSET.size*(len(blobs)+1)
    offsets=[offset]
    for blob in blobs: offsets.append(offsets[-1]+len(blob))
    with Path(path).open("wb") as f:
        f.write(PREFIX.pack(b"APC1",len(blobs)))
        for x in offsets: f.write(OFFSET.pack(x))
        for blob in blobs: f.write(blob)
    return offsets[-1]


class TileReader:
    def __init__(self,path):
        self.path=Path(path)
        with self.path.open("rb") as f:
            header=f.read(PREFIX.size)
            if len(header)!=PREFIX.size: raise ValueError("truncated container")
            magic,count=PREFIX.unpack(header)
            size=self.path.stat().st_size
            if magic!=b"APC1" or count>(size-PREFIX.size)//8-1: raise ValueError("invalid container")
            self.offsets=[OFFSET.unpack(f.read(8))[0] for _ in range(count+1)]
            if self.offsets[0]!=PREFIX.size+8*(count+1) or self.offsets[-1]!=size:
                raise ValueError("invalid index extent")
            if any(a>=b or b-a>MAX_TILE*2 for a,b in zip(self.offsets,self.offsets[1:])):
                raise ValueError("invalid tile offsets")
        self.count=count; self.bytes_read=0
    def read(self,index):
        if not 0<=index<self.count: raise IndexError(index)
        start,end=self.offsets[index:index+2]
        with self.path.open("rb") as f:
            f.seek(start); blob=f.read(end-start)
        self.bytes_read+=len(blob)
        return decode(blob)
