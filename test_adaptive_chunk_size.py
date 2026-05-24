#!/usr/bin/env python3
"""
Tests für Adaptive Chunk-Size.
"""

import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from adaptive_chunk_size import AdaptiveChunkSizer, AdaptiveChunkSizeManager


class TestAdaptiveChunkSizer(unittest.TestCase):
    """Tests für AdaptiveChunkSizer."""
    
    def setUp(self):
        """Setup für Tests."""
        self.sizer = AdaptiveChunkSizer()
        
    def test_init(self):
        """Test Initialisierung."""
        self.assertEqual(self.sizer.min_chunk_size, 64)
        self.assertEqual(self.sizer.max_chunk_size, 2048)
        self.assertEqual(self.sizer.base_chunk_size, 512)
        
    def test_compute_basic(self):
        """Test grundlegende Berechnung."""
        chunk_size = self.sizer.compute_chunk_size(
            prompt_length=512,
            model_size_gb=10.0,
            available_memory_gb=20.0,
        )
        self.assertGreater(chunk_size, 0)
        self.assertLessEqual(chunk_size, 2048)
        self.assertGreaterEqual(chunk_size, 64)
        
    def test_compute_long_prompt(self):
        """Test mit langem Prompt."""
        chunk_size = self.sizer.compute_chunk_size(
            prompt_length=8192,
            model_size_gb=10.0,
            available_memory_gb=20.0,
        )
        self.assertGreaterEqual(chunk_size, 256)
        
    def test_compute_short_prompt(self):
        """Test mit kurzem Prompt."""
        chunk_size = self.sizer.compute_chunk_size(
            prompt_length=128,
            model_size_gb=10.0,
            available_memory_gb=20.0,
        )
        self.assertLess(chunk_size, 512)
        
    def test_compute_low_memory(self):
        """Test mit wenig Speicher."""
        chunk_size = self.sizer.compute_chunk_size(
            prompt_length=512,
            model_size_gb=20.0,
            available_memory_gb=5.0,
        )
        self.assertLessEqual(chunk_size, 512)
        
    def test_compute_long_sequence(self):
        """Test mit langer Sequenz."""
        chunk_size = self.sizer.compute_chunk_size(
            prompt_length=512,
            model_size_gb=10.0,
            available_memory_gb=20.0,
            sequence_length=16384,
        )
        self.assertLess(chunk_size, 512)
        
    def test_compute_large_model(self):
        """Test mit großem Modell."""
        chunk_size = self.sizer.compute_chunk_size(
            prompt_length=512,
            model_size_gb=40.0,
            available_memory_gb=20.0,
        )
        self.assertLess(chunk_size, 512)
        
    def test_cache(self):
        """Test Cache-Funktionalität."""
        chunk1 = self.sizer.compute_chunk_size(512, 10.0, 20.0)
        chunk2 = self.sizer.compute_chunk_size(512, 10.0, 20.0)
        self.assertEqual(chunk1, chunk2)
        
    def test_reset_cache(self):
        """Test Cache zurücksetzen."""
        self.sizer.compute_chunk_size(512, 10.0, 20.0)
        self.sizer.reset_cache()
        self.assertEqual(len(self.sizer._cache), 0)
        
    def test_get_profile(self):
        """Test Profil abrufen."""
        profile = self.sizer.get_profile("fast")
        self.assertEqual(profile["base_chunk_size"], 1024)
        
        profile = self.sizer.get_profile("memory_efficient")
        self.assertEqual(profile["base_chunk_size"], 256)
        
        profile = self.sizer.get_profile("invalid")
        self.assertEqual(profile["base_chunk_size"], 512)
        
    def test_apply_profile(self):
        """Test Profil anwenden."""
        self.sizer.apply_profile("fast")
        self.assertEqual(self.sizer.base_chunk_size, 1024)
        self.assertEqual(self.sizer.max_chunk_size, 2048)
        
        self.sizer.apply_profile("low_latency")
        self.assertEqual(self.sizer.base_chunk_size, 128)
        self.assertEqual(self.sizer.max_chunk_size, 256)
        
    def test_chunk_size_is_power_of_2(self):
        """Test dass Chunk-Size Potenz von 2 ist."""
        for prompt_length in [128, 512, 2048, 8192]:
            chunk_size = self.sizer.compute_chunk_size(
                prompt_length=prompt_length,
                model_size_gb=10.0,
                available_memory_gb=20.0,
            )
            # Prüfe ob Potenz von 2
            self.assertEqual(chunk_size & (chunk_size - 1), 0)


class TestAdaptiveChunkSizeManager(unittest.TestCase):
    """Tests für AdaptiveChunkSizeManager."""
    
    def setUp(self):
        """Setup für Tests."""
        self.manager = AdaptiveChunkSizeManager()
        
    def test_init(self):
        """Test Initialisierung."""
        self.assertEqual(self.manager.current_profile, "balanced")
        self.assertEqual(len(self.manager.request_history), 0)
        
    def test_record_request(self):
        """Test Request aufzeichnen."""
        self.manager.record_request(512, 512, 1.0)
        self.assertEqual(len(self.manager.request_history), 1)
        self.assertEqual(self.manager.request_history[0]["prompt_length"], 512)
        
    def test_record_request_limit(self):
        """Test Request-Limit."""
        for i in range(150):
            self.manager.record_request(512, 512, 1.0)
        self.assertLessEqual(len(self.manager.request_history), 100)
        
    def test_optimize_profile(self):
        """Test Profil-Optimierung."""
        # Zeichne einige Requests auf
        for i in range(20):
            self.manager.record_request(4096, 512, 2.0)
            
        # Optimiere
        self.manager.optimize_profile()
        
        # Profil sollte sich geändert haben oder gleich bleiben
        self.assertIn(self.manager.current_profile, 
                     ["fast", "balanced", "memory_efficient", "low_latency"])


if __name__ == "__main__":
    unittest.main()
