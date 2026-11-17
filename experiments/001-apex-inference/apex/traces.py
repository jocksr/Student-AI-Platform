from collections import Counter
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
import random
from .common import sha256, write_json


@dataclass(frozen=True)
class TraceEvent:
    request_id: str
    phase: str
    step: int
    layer: int
    selected_experts: tuple[int,...]
    timestamp_ns: int | None = None
    batch_size: int = 1
    context_length: int = 0
    source: str = "synthetic"
    measurements: dict = field(default_factory=dict)

    def validate(self,layers,experts,top_k=None):
        if not isinstance(self.request_id,str) or not self.request_id: raise ValueError("request_id required")
        if self.phase not in ("prefill","decode"): raise ValueError("invalid phase")
        if self.source not in ("synthetic","real-small-model","gpt-oss-120b"): raise ValueError("invalid source")
        for x in (self.step,self.layer,self.batch_size,self.context_length,*self.selected_experts):
            if type(x) is not int: raise ValueError("trace indices must be integers")
        if self.step<0 or not 0<=self.layer<layers or self.batch_size<1 or self.context_length<0:
            raise ValueError("invalid event dimensions")
        if self.timestamp_ns is not None and (type(self.timestamp_ns) is not int or self.timestamp_ns<0):
            raise ValueError("invalid timestamp")
        if not self.selected_experts or len(set(self.selected_experts))!=len(self.selected_experts):
            raise ValueError("experts must be nonempty and unique")
        if any(not 0<=x<experts for x in self.selected_experts): raise ValueError("expert out of bounds")
        if top_k is not None and len(self.selected_experts)!=top_k: raise ValueError("incorrect top-k")
        return self


def generate(steps=64,layers=36,experts=128,top_k=4,seed=17,workload="hotset",requests=1):
    if min(steps,layers,experts,top_k,requests)<1 or top_k>experts: raise ValueError("invalid trace configuration")
    if workload not in ("uniform","hotset","phase-shift"): raise ValueError("unknown workload")
    rng=random.Random(seed)
    hot_size=min(experts,max(top_k,experts//8))
    for step in range(steps):
        for request in range(requests):
            for layer in range(layers):
                shift=experts//2 if workload=="phase-shift" and step>=steps//2 else 0
                hot=[(layer*7+shift+i)%experts for i in range(hot_size)]
                selected=[]
                while len(selected)<top_k:
                    e=rng.randrange(experts) if workload=="uniform" or rng.random()>0.85 else rng.choice(hot)
                    if e not in selected: selected.append(e)
                yield TraceEvent(f"synthetic-{request}","decode",step,layer,tuple(selected),
                                 context_length=128+step)


def save_trace(path,events,metadata):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="\n") as f:
        for e in events: f.write(json.dumps(asdict(e),sort_keys=True,separators=(",",":"))+"\n")
    metadata=dict(metadata,sha256=sha256(path.read_bytes()),schema_version=1)
    write_json(str(path)+".meta.json",metadata)
    return metadata


def load_trace(path,layers,experts,top_k=None):
    with Path(path).open(encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            try:
                row=json.loads(line); row["selected_experts"]=tuple(row["selected_experts"])
                yield TraceEvent(**row).validate(layers,experts,top_k)
            except (TypeError,KeyError,ValueError) as e:
                raise ValueError(f"{path}:{line_no}: {e}") from e


def analyze(events):
    frequency=Counter(); coactivation=Counter(); count=0; sources=set()
    for e in events:
        count+=1; sources.add(e.source)
        frequency.update((e.layer,x) for x in e.selected_experts)
        for i,a in enumerate(e.selected_experts):
            for b in e.selected_experts[i+1:]: coactivation[(e.layer,*sorted((a,b)))]+=1
    return dict(events=count,sources=sorted(sources),unique_experts=len(frequency),
                top_experts=[dict(layer=k[0],expert=k[1],count=v) for k,v in frequency.most_common(20)],
                top_pairs=[dict(layer=k[0],experts=list(k[1:]),count=v) for k,v in coactivation.most_common(20)])
