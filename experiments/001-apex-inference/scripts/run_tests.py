import json
import argparse
import os
from pathlib import Path
import sys
import time
import unittest
experiment_root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(experiment_root))
from apex.common import provenance,write_json

parser=argparse.ArgumentParser(description="Run the APEX experiment correctness suite")
parser.add_argument("--out",default="results/tests.json")
args=parser.parse_args()
os.chdir(experiment_root)
start=time.perf_counter()
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover("tests"))
write_json(args.out,provenance("correctness-tests",tests=result.testsRun,
    failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
    elapsed_seconds=time.perf_counter()-start,passed=result.wasSuccessful()))
sys.exit(not result.wasSuccessful())
