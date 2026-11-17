import random
import unittest
from apex.cache import ByteCache,ExpertRecord


class CacheTests(unittest.TestCase):
    def test_lru_eviction_order(self):
        c=ByteCache(20,"lru")
        for k in ("a","b"): c.access(k); c.insert(k,10)
        c.access("a"); c.access("c"); c.insert("c",10)
        self.assertTrue(c.contains("a")); self.assertFalse(c.contains("b")); self.assertEqual(c.evictions,1)

    def test_lfu_eviction(self):
        c=ByteCache(20,"lfu")
        for k in ("a","b"): c.access(k); c.insert(k,10)
        c.access("a"); c.access("b"); c.access("a"); c.insert("c",10)
        self.assertTrue(c.contains("a")); self.assertFalse(c.contains("b"))

    def test_cost_policy_preserves_expensive_item(self):
        c=ByteCache(20,"cost")
        c.access("expensive"); c.insert("expensive",10,1000)
        c.access("cheap"); c.insert("cheap",10,1)
        c.insert("next",10,1)
        self.assertTrue(c.contains("expensive")); self.assertFalse(c.contains("cheap"))

    def test_oversized_entry_does_not_flush_cache(self):
        c=ByteCache(10); c.insert("a",10)
        self.assertFalse(c.insert("b",11)); self.assertTrue(c.contains("a"))

    def test_disabled_and_zero_cache(self):
        for c in (ByteCache(0),ByteCache(100,"none")):
            self.assertFalse(c.insert("a",1)); self.assertFalse(c.access("a")); self.assertEqual(c.used,0)

    def test_random_byte_budget_invariant(self):
        rng=random.Random(4); sizes={k:rng.randrange(1,20) for k in range(30)}
        for policy in ("none","lru","lfu","cost"):
            c=ByteCache(73,policy)
            for _ in range(500):
                k=rng.randrange(30); c.access(k); c.insert(k,sizes[k],k+1)
                self.assertEqual(c.used,sum(v[0] for v in c.entries.values()))
                self.assertLessEqual(c.used,73)

    def test_invalid_budgets_and_records(self):
        with self.assertRaises(ValueError): ByteCache(-1)
        with self.assertRaises(ValueError): ExpertRecord(0,0,0,1)
