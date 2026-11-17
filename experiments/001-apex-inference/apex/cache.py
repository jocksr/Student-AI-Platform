"""Byte-bounded deterministic caches; no global workload lookahead."""
from collections import Counter, OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertRecord:
    layer_id: int
    expert_id: int
    original_size: int
    compressed_size: int
    codec_id: str = "raw"
    representation_id: str = "serialized-model-weight"
    decode_ns: float = 0
    storage_path: str = ""
    storage_offset: int = 0

    def __post_init__(self):
        if min(self.original_size,self.compressed_size)<1 or self.decode_ns<0:
            raise ValueError("invalid expert sizes or decode cost")
    @property
    def key(self): return (self.layer_id,self.expert_id)


class ByteCache:
    def __init__(self,budget,policy="lru"):
        if budget<0: raise ValueError("negative budget")
        if policy not in ("none","lru","lfu","cost"): raise ValueError("unknown policy")
        self.budget=budget; self.policy=policy; self.entries=OrderedDict()
        self.frequency=Counter(); self.clock=0; self.used=0; self.peak=0; self.evictions=0

    def contains(self,key): return key in self.entries

    def access(self,key):
        self.clock+=1; self.frequency[key]+=1
        if key not in self.entries: return False
        self.entries[key][1]=self.clock; self.entries.move_to_end(key)
        return True

    def insert(self,key,size,reload_ns=1.0):
        if size<=0: raise ValueError("nonpositive object size")
        if self.policy=="none" or size>self.budget: return False
        if key in self.entries: return True
        while self.used+size>self.budget:
            if self.policy=="lru": victim=next(iter(self.entries))
            else:
                def score(k):
                    s,last,cost=self.entries[k]
                    value=self.frequency[k] if self.policy=="lfu" else self.frequency[k]*cost/s
                    return (value,last,k)
                victim=min(self.entries,key=score)
            self.used-=self.entries.pop(victim)[0]; self.evictions+=1
        self.entries[key]=[size,self.clock,reload_ns]; self.used+=size; self.peak=max(self.peak,self.used)
        assert 0<=self.used<=self.budget
        return True
