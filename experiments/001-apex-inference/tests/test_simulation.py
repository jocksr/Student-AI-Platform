import tempfile
from pathlib import Path
import unittest
from apex.cache import ExpertRecord
from apex.traces import TraceEvent,generate,save_trace,load_trace
from apex.simulation import CostModel,replay


def event(step,expert=0): return TraceEvent("a","decode",step,0,(expert,))


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.records={(0,i):ExpertRecord(0,i,100,50,decode_ns=10) for i in range(3)}
        self.cost=CostModel(1,2)

    def test_hand_calculated_no_cache(self):
        r=replay([event(0),event(1)],self.records,100,100,"none",self.cost)
        self.assertEqual(r["metrics"]["ssd_bytes"],100)
        self.assertEqual(r["metrics"]["h2d_bytes"],200)
        self.assertEqual(r["metrics"]["estimated_serial_ns"],220)

    def test_gpu_hit_removes_storage_decode_transfer(self):
        r=replay([event(0),event(1)],self.records,100,100,"lru",self.cost)
        self.assertEqual(r["metrics"]["gpu_hits"],1)
        self.assertEqual(r["metrics"]["estimated_serial_ns"],110)

    def test_host_hit_still_transfers(self):
        r=replay([event(0),event(1)],self.records,100,0,"lru",self.cost)
        self.assertEqual(r["metrics"]["ram_hits"],1)
        self.assertEqual(r["metrics"]["estimated_serial_ns"],160)

    def test_compressed_host_trades_capacity_for_decode(self):
        events=[event(0,0),event(1,1),event(2,0)]
        raw=replay(events,self.records,100,0,"lru",self.cost,"raw")
        comp=replay(events,self.records,100,0,"lru",self.cost,"compressed")
        self.assertEqual(raw["metrics"]["ssd_misses"],3)
        self.assertEqual(comp["metrics"]["ssd_misses"],2)
        self.assertEqual(comp["ram_peak"],100)
        self.assertEqual(comp["metrics"]["decode_ns"],30)

    def test_cpu_can_avoid_weight_upload(self):
        cost=CostModel(1,2,cpu_expert_ns=20,gpu_expert_ns=1,activation_roundtrip_ns=5)
        r=replay([event(0)],self.records,100,100,"lru",cost,placement="adaptive")
        self.assertEqual(r["metrics"]["h2d_bytes"],0)
        self.assertEqual(r["metrics"]["estimated_serial_ns"],85)

    def test_reject_unknown_compute_costs(self):
        with self.assertRaises(ValueError): replay([event(0)],self.records,1,1,"lru",self.cost,placement="adaptive")
        for value in (0,-1,float("nan"),float("inf")):
            with self.assertRaises(ValueError): CostModel(value,1)

    def test_request_identity_kept_separate(self):
        events=[event(0),TraceEvent("b","decode",0,0,(0,))]
        r=replay(events,self.records,0,0,"none",self.cost)
        self.assertEqual(r["token_steps"],2)

    def test_seed_and_model_dimensions(self):
        a=list(generate(3,2,8,2,17)); b=list(generate(3,2,8,2,17))
        self.assertEqual(a,b); self.assertNotEqual(a,list(generate(3,2,8,2,18)))
        for e in a: e.validate(2,8,2)

    def test_trace_serialization_and_bad_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"trace.jsonl"; events=list(generate(2,2,8,2))
            save_trace(path,events,{"source":"synthetic"})
            self.assertEqual(events,list(load_trace(path,2,8,2)))
            path.write_text('{"broken":true}\n')
            with self.assertRaisesRegex(ValueError,":1:"): list(load_trace(path,2,8))

    def test_invalid_experts(self):
        for experts in ((),(0,0),(8,),(-1,),(True,)):
            with self.assertRaises(ValueError): TraceEvent("a","decode",0,0,experts).validate(1,8)
