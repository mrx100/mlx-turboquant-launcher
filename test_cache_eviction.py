#!/usr/bin/env python3
"""
Tests für Cache Eviction Policies.
"""

import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cache_eviction import (
    CacheEntry,
    LRUEvictionPolicy,
    LFUEvictionPolicy,
    AttentionScoreEvictionPolicy,
    SlidingWindowEvictionPolicy,
    CacheEvictionManager,
)


class TestLRUEvictionPolicy(unittest.TestCase):
    """Tests für LRUEvictionPolicy."""
    
    def setUp(self):
        """Setup für Tests."""
        self.policy = LRUEvictionPolicy(max_size=5)
        
    def test_init(self):
        """Test Initialisierung."""
        self.assertEqual(self.policy.max_size, 5)
        self.assertEqual(self.policy.size(), 0)
        
    def test_put_get(self):
        """Test Einfügen und Abrufen."""
        entry = CacheEntry(key="k1", value="v1", position=0)
        self.policy.put(0, entry)
        
        retrieved = self.policy.get(0)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.key, "k1")
        
    def test_eviction(self):
        """Test Eviction."""
        for i in range(7):
            entry = CacheEntry(key=f"k{i}", value=f"v{i}", position=i)
            self.policy.put(i, entry)
            
        self.assertLessEqual(self.policy.size(), 5)
        
    def test_lru_order(self):
        """Test LRU-Reihenfolge."""
        for i in range(5):
            entry = CacheEntry(key=f"k{i}", value=f"v{i}", position=i)
            self.policy.put(i, entry)
            
        # Access key 0 again
        self.policy.get(0)
        
        # Add new entry, should evict key 1 (least recently used)
        entry = CacheEntry(key="k5", value="v5", position=5)
        self.policy.put(5, entry)
        
        # Key 0 should still be there
        self.assertIsNotNone(self.policy.get(0))
        
    def test_evict_method(self):
        """Test evict() Methode."""
        for i in range(5):
            entry = CacheEntry(key=f"k{i}", value=f"v{i}", position=i)
            self.policy.put(i, entry)
            
        evicted = self.policy.evict(2)
        self.assertEqual(len(evicted), 2)
        self.assertEqual(self.policy.size(), 3)


class TestLFUEvictionPolicy(unittest.TestCase):
    """Tests für LFUEvictionPolicy."""
    
    def setUp(self):
        """Setup für Tests."""
        self.policy = LFUEvictionPolicy(max_size=5)
        
    def test_init(self):
        """Test Initialisierung."""
        self.assertEqual(self.policy.max_size, 5)
        self.assertEqual(self.policy.size(), 0)
        
    def test_put_get(self):
        """Test Einfügen und Abrufen."""
        entry = CacheEntry(key="k1", value="v1", position=0)
        self.policy.put(0, entry)
        
        retrieved = self.policy.get(0)
        self.assertIsNotNone(retrieved)
        
    def test_lfu_eviction(self):
        """Test LFU Eviction."""
        for i in range(5):
            entry = CacheEntry(key=f"k{i}", value=f"v{i}", position=i)
            self.policy.put(i, entry)
            
        # Access some keys more often
        self.policy.get(0)
        self.policy.get(0)
        self.policy.get(1)
        
        # Add new entry, should evict least frequently used
        entry = CacheEntry(key="k5", value="v5", position=5)
        self.policy.put(5, entry)
        
        self.assertLessEqual(self.policy.size(), 5)
        
    def test_evict_method(self):
        """Test evict() Methode."""
        for i in range(5):
            entry = CacheEntry(key=f"k{i}", value=f"v{i}", position=i)
            self.policy.put(i, entry)
            
        evicted = self.policy.evict(2)
        self.assertEqual(len(evicted), 2)


class TestAttentionScoreEvictionPolicy(unittest.TestCase):
    """Tests für AttentionScoreEvictionPolicy."""
    
    def setUp(self):
        """Setup für Tests."""
        self.policy = AttentionScoreEvictionPolicy(max_size=10, window_size=4)
        
    def test_init(self):
        """Test Initialisierung."""
        self.assertEqual(self.policy.max_size, 10)
        self.assertEqual(self.policy.window_size, 4)
        
    def test_put_get(self):
        """Test Einfügen und Abrufen."""
        entry = CacheEntry(key="k1", value="v1", position=0, attention_score=0.8)
        self.policy.put(0, entry)
        
        retrieved = self.policy.get(0)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.attention_score, 0.8)
        
    def test_attention_eviction(self):
        """Test Attention-Score Eviction."""
        for i in range(10):
            entry = CacheEntry(
                key=f"k{i}",
                value=f"v{i}",
                position=i,
                attention_score=0.1 * i,
            )
            self.policy.put(i, entry)
            
        self.assertLessEqual(self.policy.size(), 10)
        
    def test_update_scores(self):
        """Test Score-Update."""
        for i in range(5):
            entry = CacheEntry(key=f"k{i}", value=f"v{i}", position=i, attention_score=0.5)
            self.policy.put(i, entry)
            
        self.policy.update_scores({0: 0.9, 1: 0.1})
        
        entry0 = self.policy.get(0)
        entry1 = self.policy.get(1)
        self.assertEqual(entry0.attention_score, 0.9)
        self.assertEqual(entry1.attention_score, 0.1)


class TestSlidingWindowEvictionPolicy(unittest.TestCase):
    """Tests für SlidingWindowEvictionPolicy."""
    
    def setUp(self):
        """Setup für Tests."""
        self.policy = SlidingWindowEvictionPolicy(
            max_size=10,
            window_size=4,
            important_ratio=0.2,
        )
        
    def test_init(self):
        """Test Initialisierung."""
        self.assertEqual(self.policy.max_size, 10)
        self.assertEqual(self.policy.window_size, 4)
        self.assertEqual(self.policy.important_count, 2)
        
    def test_put_get(self):
        """Test Einfügen und Abrufen."""
        entry = CacheEntry(key="k1", value="v1", position=0)
        self.policy.put(0, entry)
        
        retrieved = self.policy.get(0)
        self.assertIsNotNone(retrieved)
        
    def test_sliding_window(self):
        """Test Sliding Window Verhalten."""
        for i in range(10):
            entry = CacheEntry(key=f"k{i}", value=f"v{i}", position=i)
            self.policy.put(i, entry)
            
        self.assertLessEqual(self.policy.size(), 10)
        
    def test_mark_important(self):
        """Test Important-Markierung."""
        for i in range(5):
            entry = CacheEntry(key=f"k{i}", value=f"v{i}", position=i)
            self.policy.put(i, entry)
            
        self.policy.mark_important([0, 1])
        self.assertIn(0, self.policy.important_keys)
        self.assertIn(1, self.policy.important_keys)


class TestCacheEvictionManager(unittest.TestCase):
    """Tests für CacheEvictionManager."""
    
    def test_init_lru(self):
        """Test Initialisierung mit LRU."""
        manager = CacheEvictionManager(max_size=10, policy="lru")
        self.assertEqual(manager.policy_name, "lru")
        
    def test_init_lfu(self):
        """Test Initialisierung mit LFU."""
        manager = CacheEvictionManager(max_size=10, policy="lfu")
        self.assertEqual(manager.policy_name, "lfu")
        
    def test_init_attention(self):
        """Test Initialisierung mit Attention."""
        manager = CacheEvictionManager(max_size=10, policy="attention")
        self.assertEqual(manager.policy_name, "attention")
        
    def test_init_sliding_window(self):
        """Test Initialisierung mit Sliding Window."""
        manager = CacheEvictionManager(max_size=10, policy="sliding_window")
        self.assertEqual(manager.policy_name, "sliding_window")
        
    def test_default_policy(self):
        """Test Default-Policy."""
        manager = CacheEvictionManager(max_size=10, policy="invalid")
        self.assertEqual(manager.policy_name, "invalid")
        
    def test_get_stats(self):
        """Test Statistiken."""
        manager = CacheEvictionManager(max_size=10, policy="lru")
        stats = manager.get_stats()
        
        self.assertEqual(stats["policy"], "lru")
        self.assertEqual(stats["max_size"], 10)
        self.assertEqual(stats["size"], 0)


if __name__ == "__main__":
    unittest.main()
