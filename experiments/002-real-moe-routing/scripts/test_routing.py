"""Checks for trace corruption and misattributed request boundaries."""
import struct
import unittest

from routing import read_mct, cycles, convert
from benchmark import check_answer


def fixture():
    data = bytearray(b"MCT1")
    data.extend(struct.pack("<3I",2,2,1))
    data.extend(struct.pack("<3I",0,8,2))
    data.extend(struct.pack("<2I",0,1))
    data.extend(struct.pack("<2d3QI2Q",1,1,4096,4096,256,2,128,128))
    for tokens in (4,1,1,4,1,1,4,1):
        for layer in range(2):
            ids = [1,3]*tokens
            data.extend(struct.pack("<3i",layer,2,len(ids)))
            data.extend(struct.pack("<"+"i"*len(ids),*ids))
    return bytes(data)


class RoutingTests(unittest.TestCase):
    def test_header_records_and_partial_tail(self):
        full = read_mct(fixture())
        self.assertEqual(full["groups"][0]["expert_bytes"],256)
        self.assertEqual(len(cycles(full)[0]),8)
        partial = read_mct(fixture()[:-1])
        self.assertGreater(partial["trailing_bytes"],0)
        self.assertEqual(cycles(partial)[1],1)

    def test_coverage_and_missing_tail(self):
        request = {"response":{"timings":{"predicted_n":3,"prompt_n":100}}}
        run = {"warmup":request,"requests":[dict(request,task_id="test")]}
        parsed = read_mct(fixture())
        events, coverage = convert(run,parsed)
        self.assertEqual(len(events),24)
        self.assertEqual(coverage["requests"][1]["observed_decode_steps"],2)
        parsed["records"] = parsed["records"][:12]
        with self.assertRaisesRegex(ValueError,"flush-workload"): convert(run,parsed)

    def test_corrupt_expert_and_layer_order(self):
        bad = bytearray(fixture()); struct.pack_into("<i",bad,len(bad)-4,99)
        with self.assertRaisesRegex(ValueError,"expert IDs"): read_mct(bytes(bad))
        parsed = read_mct(fixture()); parsed["records"][1]["layer"] = 0
        with self.assertRaisesRegex(ValueError,"layer cycle"): cycles(parsed)

    def test_reject_wrong_request_length(self):
        request = {"response":{"timings":{"predicted_n":4,"prompt_n":100}}}
        with self.assertRaisesRegex(ValueError,"decode coverage"):
            convert({"warmup":request,"requests":[]},read_mct(fixture()))

    def test_answer_checks_reject_extra_claims(self):
        self.assertTrue(check_answer({"check":"exact","expected":"391"},"391."))
        self.assertFalse(check_answer({"check":"exact","expected":"391"},"391 or 392"))
        self.assertTrue(check_answer({"check":"json","expected":[1,2]},"```json\n[1, 2]\n```"))
        self.assertFalse(check_answer({"check":"json","expected":[1,2]},"[2,1]"))


if __name__ == "__main__": unittest.main()
