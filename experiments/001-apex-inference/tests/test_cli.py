import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CliTests(unittest.TestCase):
    def call(self,*args):
        return subprocess.run([sys.executable,"-m","apex",*args],capture_output=True,text=True,timeout=30)

    def test_generate_replay_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace=str(Path(tmp)/"trace.jsonl"); result=str(Path(tmp)/"result.json")
            p=self.call("generate-trace","--out",trace,"--steps","3","--layers","2","--experts","8","--top-k","2","--requests","2")
            self.assertEqual(p.returncode,0,p.stderr)
            p=self.call("simulate","--trace",trace,"--out",result,"--layers","2","--experts","8","--policy","lru","--expert-bytes","128","--gpu-mib","0")
            self.assertEqual(p.returncode,0,p.stderr)
            data=json.loads(Path(result).read_text())
            self.assertEqual(data["metrics"]["accesses"],24)
            self.assertEqual(data["token_steps"],6)
            self.assertEqual(data["source_categories"],["synthetic"])

    def test_invalid_configuration_exits_nonzero(self):
        p=self.call("generate-trace","--top-k","129")
        self.assertEqual(p.returncode,2)
        self.assertIn("invalid trace configuration",p.stderr)
