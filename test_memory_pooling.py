#!/usr/bin/env python3
"""
Tests für Memory Pooling.
"""

import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from memory_pooling import MemoryPool, KVCachePool, MemoryPoolManager


class TestMemoryPool(unittest.TestCase):
    """Tests für MemoryPool."""
    
    def setUp(self):
        """Setup für Tests."""
        self.pool = MemoryPool(
            block_size=1024,
            max_blocks=10,
            dtype="float16",
        )
        
    def test_init(self):
        """Test Initialisierung."""
        self.assertEqual(self.pool.block_size, 1024)
        self.assertEqual(self.pool.max_blocks, 10)
        self.assertEqual(self.pool.size(), 10)
        self.assertEqual(self.pool.in_use_count(), 0)
        
    def test_acquire_release(self):
        """Test Acquire und Release."""
        block = self.pool.acquire()
        self.assertIsNotNone(block)
        self.assertEqual(self.pool.size(), 9)
        self.assertEqual(self.pool.in_use_count(), 1)
        
        self.pool.release(block)
        self.assertEqual(self.pool.size(), 10)
        self.assertEqual(self.pool.in_use_count(), 0)
        
    def test_acquire_empty(self):
        """Test Acquire von leerem Pool."""
        # Alle Blöcke allozieren
        blocks = []
        for _ in range(10):
            block = self.pool.acquire()
            if block:
                blocks.append(block)
                
        # Pool sollte leer sein
        block = self.pool.acquire()
        self.assertIsNone(block)
        
        # Blöcke zurückgeben
        for block in blocks:
            self.pool.release(block)
            
    def test_utilization(self):
        """Test Auslastung."""
        self.assertEqual(self.pool.utilization(), 0.0)
        
        block = self.pool.acquire()
        self.assertGreater(self.pool.utilization(), 0.0)
        
        self.pool.release(block)
        self.assertEqual(self.pool.utilization(), 0.0)


class TestKVCachePool(unittest.TestCase):
    """Tests für KVCachePool."""
    
    def setUp(self):
        """Setup für Tests."""
        self.pool = KVCachePool(
            max_batch_size=4,
            max_seq_length=512,
            n_layers=4,
            n_heads=8,
            head_dim=64,
        )
        
    def test_init(self):
        """Test Initialisierung."""
        self.assertEqual(self.pool.max_batch_size, 4)
        self.assertEqual(self.pool.n_layers, 4)
        
    def test_allocate_free_cache(self):
        """Test KV-Cache allozieren und freigeben."""
        cache = self.pool.allocate_cache("req1")
        self.assertIsNotNone(cache)
        self.assertEqual(cache["request_id"], "req1")
        self.assertEqual(len(cache["layers"]), 4)
        
        stats = self.pool.get_stats()
        self.assertGreater(stats["k_pool_in_use"], 0)
        
        self.pool.free_cache("req1")
        stats = self.pool.get_stats()
        self.assertEqual(stats["active_caches"], 0)
        
    def test_duplicate_allocate(self):
        """Test doppelte Allokation."""
        cache1 = self.pool.allocate_cache("req1")
        cache2 = self.pool.allocate_cache("req1")
        
        # Sollte gleiche Cache zurückgeben
        self.assertEqual(cache1, cache2)
        
        self.pool.free_cache("req1")
        
    def test_pool_exhaustion(self):
        """Test Pool-Erschöpfung."""
        # Alloziere mehrere Caches
        caches = []
        for i in range(4):
            cache = self.pool.allocate_cache(f"req{i}")
            if cache:
                caches.append(cache)
                
        # Pool sollte erschöpft sein
        cache = self.pool.allocate_cache("req_overflow")
        self.assertIsNone(cache)
        
        # Caches freigeben
        for cache in caches:
            self.pool.free_cache(cache["request_id"])


class TestMemoryPoolManager(unittest.TestCase):
    """Tests für MemoryPoolManager."""
    
    def setUp(self):
        """Setup für Tests."""
        self.manager = MemoryPoolManager()
        
    def test_init(self):
        """Test Initialisierung."""
        self.assertEqual(len(self.manager.pools), 0)
        
    def test_create_pool(self):
        """Test Pool erstellen."""
        pool = self.manager.create_pool(
            pool_name="test_pool",
            block_size=512,
            max_blocks=10,
        )
        
        self.assertIn("test_pool", self.manager.pools)
        self.assertEqual(pool.max_blocks, 10)
        
    def test_get_pool(self):
        """Test Pool abrufen."""
        self.manager.create_pool("pool1", 512, 10)
        pool = self.manager.get_pool("pool1")
        self.assertIsNotNone(pool)
        
        pool = self.manager.get_pool("nonexistent")
        self.assertIsNone(pool)
        
    def test_remove_pool(self):
        """Test Pool entfernen."""
        self.manager.create_pool("pool1", 512, 10)
        self.manager.remove_pool("pool1")
        self.assertNotIn("pool1", self.manager.pools)
        
    def test_get_stats(self):
        """Test Statistiken."""
        self.manager.create_pool("pool1", 512, 10)
        stats = self.manager.get_stats()
        
        self.assertIn("pool1", stats)
        self.assertEqual(stats["pool1"]["size"], 10)
        
    def test_reset_all(self):
        """Test Reset aller Pools."""
        pool = self.manager.create_pool("pool1", 512, 10)
        block = pool.acquire()
        
        self.manager.reset_all()
        self.assertEqual(pool.size(), 10)
        self.assertEqual(pool.in_use_count(), 0)


if __name__ == "__main__":
    unittest.main()
