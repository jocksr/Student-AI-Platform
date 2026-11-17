import os
from pathlib import Path
import random
import tempfile
import unittest
from apex.compression import (encode,decode,transform,TRANSFORMS,available_codecs,
                              HEADER,MAX_TILE,select_representation,synthetic_corpora)
from apex.container import write_container,TileReader


class CompressionTests(unittest.TestCase):
    def test_roundtrip_every_codec_transform_edge_and_random(self):
        rng=random.Random(12)
        for codec in available_codecs():
            for trans in TRANSFORMS:
                for size in (0,1,2,3,31,127,16384,65537):
                    with self.subTest(codec=codec,transform=trans,size=size):
                        data=rng.randbytes(size)
                        self.assertEqual(decode(encode(data,codec,trans)),data)

    def test_roundtrip_constant_and_all_bit_patterns(self):
        for data in (bytes(65536),bytes([255])*1025,bytes(range(256))*17):
            for codec in available_codecs():
                for trans in TRANSFORMS: self.assertEqual(decode(encode(data,codec,trans)),data)

    def test_transform_independent_inverse(self):
        data=bytes(range(256))+b"odd"
        for name in TRANSFORMS: self.assertEqual(transform(transform(data,name),name,True),data)

    def test_corruption_detected(self):
        tile=bytearray(encode(b"correctness"*100)); tile[-1]^=1
        with self.assertRaises(ValueError): decode(tile)

    def test_truncation_unknown_and_size_guard(self):
        for bad in (b"",b"APX1",encode(b"abc")[:-1],HEADER.pack(b"APX1",99,0,0,0,0,0),
                    HEADER.pack(b"APX1",0,0,0,MAX_TILE+1,0,0)):
            with self.assertRaises(ValueError): decode(bad)

    def test_container_reverse_random_access(self):
        tiles=[random.Random(i).randbytes(16384+i) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"tiles.apc"; total=write_container(p,tiles,"gzip","bitplane")
            reader=TileReader(p)
            self.assertEqual(reader.read(4),tiles[4]); self.assertLess(reader.bytes_read,total)
            for i in (2,0,3,1): self.assertEqual(reader.read(i),tiles[i])
            with self.assertRaises(IndexError): reader.read(5)

    def test_container_rejects_bad_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"bad"; p.write_bytes(b"APC1"+bytes([255])*4)
            with self.assertRaises(ValueError): TileReader(p)

    def test_cost_selection_can_reject_smaller_file(self):
        raw=dict(codec="raw",stored_bytes=1000,raw_bytes=1000,decode_ns=10)
        small=dict(codec="slow",stored_bytes=100,raw_bytes=1000,decode_ns=10000)
        self.assertEqual(select_representation([small,raw],1,10)["codec"],"raw")
        with self.assertRaises(ValueError): select_representation([],1,10)

    def test_seeded_corpora(self):
        self.assertEqual(synthetic_corpora(1024,5),synthetic_corpora(1024,5))
