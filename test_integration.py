#!/usr/bin/env python3
"""
Integrationstest für alle Optimierungen.

Testet das Zusammenspiel aller implementierten Optimierungen:
- Speculative Decoding
- Adaptive Chunk-Size
- Layer-specific Quantization
- Cache Eviction Policies
- Memory Pooling
"""

import sys
import os
import unittest
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from speculative_decoding import SpeculativeDecoder, SpeculativePrefillPipeline
from adaptive_chunk_size import AdaptiveChunkSizer, AdaptiveChunkSizeManager
from layer_specific_quantization import LayerSpecificQuantizer, ProfileGuidedOptimizer
from cache_eviction import CacheEvictionManager, CacheEntry
from memory_pooling import MemoryPoolManager, KVCachePool


class TestIntegrationSpeculativeChunkSize(unittest.TestCase):
    """Test Speculative Decoding + Adaptive Chunk-Size."""
    
    def test_combined_workflow(self):
        """Test kombinierter Workflow."""
        # Mock models
        class MockModel:
            def __call__(self, x):
                return [0.1] * 1000
                
        draft_model = MockModel()
        target_model = MockModel()
        
        # Speculative Decoder
        decoder = SpeculativeDecoder(
            draft_model=draft_model,
            target_model=target_model,
            gamma=4,
        )
        
        # Adaptive Chunk-Size
        chunk_sizer = AdaptiveChunkSizer()
        chunk_size = chunk_sizer.compute_chunk_size(
            prompt_length=1024,
            model_size_gb=10.0,
            available_memory_gb=20.0,
        )
        
        # Verify both funktionieren
        self.assertGreater(chunk_size, 0)
        self.assertIsNotNone(decoder)
        
    def test_pipeline_with_chunk_size(self):
        """Test Pipeline mit Chunk-Size."""
        class MockModel:
            def __call__(self, x):
                return [0.1] * 1000
                
        pipeline = SpeculativePrefillPipeline(
            draft_model=MockModel(),
            target_model=MockModel(),
            gamma=4,
            chunk_size=512,
        )
        
        chunk_sizer = AdaptiveChunkSizer()
        optimal_chunk = chunk_sizer.compute_chunk_size(
            prompt_length=2048,
            model_size_gb=20.0,
            available_memory_gb=16.0,
        )
        
        # Update pipeline chunk size
        pipeline.chunk_size = optimal_chunk
        
        self.assertEqual(pipeline.chunk_size, optimal_chunk)


class TestIntegrationQuantizationEviction(unittest.TestCase):
    """Test Layer-specific Quantization + Cache Eviction."""
    
    def test_combined_workflow(self):
        """Test kombinierter Workflow."""
        # Layer-specific Quantization
        quantizer = LayerSpecificQuantizer(n_layers=32)
        for i in range(32):
            quantizer.analyzer.record_score(i, i / 32.0)
            
        configs = quantizer.auto_configure(
            target_avg_bits=3.5,
            min_bits=2,
            max_bits=6,
        )
        
        # Cache Eviction
        eviction_manager = CacheEvictionManager(
            max_size=1024,
            policy="attention",
            window_size=256,
        )
        
        # Verify beide funktionieren zusammen
        self.assertEqual(len(configs), 32)
        self.assertGreater(eviction_manager.max_size, 0)
        
    def test_memory_efficient_config(self):
        """Test memory-effiziente Konfiguration."""
        quantizer = LayerSpecificQuantizer(n_layers=32)
        for i in range(32):
            quantizer.analyzer.record_score(i, 0.5)
            
        configs = quantizer.auto_configure(
            target_avg_bits=2.5,  # Niedrig für Memory-Effizienz
            min_bits=2,
            max_bits=4,
        )
        
        eviction_manager = CacheEvictionManager(
            max_size=512,  # Kleiner Cache
            policy="sliding_window",
            window_size=128,
        )
        
        # Verify memory-effiziente Konfiguration
        avg_bits = sum(c["k_bits"] for c in configs.values()) / len(configs)
        self.assertLessEqual(avg_bits, 3.5)
        self.assertLessEqual(eviction_manager.max_size, 512)


class TestIntegrationMemoryPoolEviction(unittest.TestCase):
    """Test Memory Pooling + Cache Eviction."""
    
    def test_combined_workflow(self):
        """Test kombinierter Workflow."""
        # Memory Pool
        pool_manager = MemoryPoolManager()
        kv_pool = pool_manager.create_pool(
            pool_name="kv_cache",
            block_size=4096,
            max_blocks=32,
        )
        
        # Cache Eviction
        eviction_manager = CacheEvictionManager(
            max_size=1024,
            policy="lru",
        )
        
        # Simuliere Cache-Einträge mit Memory-Pool
        for i in range(10):
            entry = CacheEntry(
                key=f"key{i}",
                value=f"value{i}",
                position=i,
                attention_score=0.1 * i,
            )
            eviction_manager.put(i, entry)
            
        # Verify beide funktionieren
        self.assertGreater(kv_pool.size(), 0)
        self.assertEqual(eviction_manager.size(), 10)
        
    def test_kv_cache_pool_with_eviction(self):
        """Test KV-Cache Pool mit Eviction."""
        kv_pool = KVCachePool(
            max_batch_size=4,
            max_seq_length=512,
            n_layers=4,
            n_heads=8,
            head_dim=64,
        )
        
        eviction_manager = CacheEvictionManager(
            max_size=16,
            policy="attention",
            window_size=4,
        )
        
        # Alloziere Cache
        cache = kv_pool.allocate_cache("req1")
        self.assertIsNotNone(cache)
        
        # Füge Einträge zum Eviction-Manager hinzu
        for i in range(8):
            entry = CacheEntry(
                key=f"token{i}",
                value=f"embedding{i}",
                position=i,
                attention_score=0.5,
            )
            eviction_manager.put(i, entry)
            
        # Verify
        stats = kv_pool.get_stats()
        self.assertGreater(stats["active_caches"], 0)
        self.assertEqual(eviction_manager.size(), 8)
        
        # Cleanup
        kv_pool.free_cache("req1")


class TestAllOptimizationsCombined(unittest.TestCase):
    """Test alle Optimierungen zusammen."""
    
    def test_full_system(self):
        """Test vollständiges System."""
        # 1. Speculative Decoding
        class MockModel:
            def __call__(self, x):
                return [0.1] * 1000
                
        decoder = SpeculativeDecoder(
            draft_model=MockModel(),
            target_model=MockModel(),
            gamma=4,
        )
        
        # 2. Adaptive Chunk-Size
        chunk_sizer = AdaptiveChunkSizer()
        chunk_size = chunk_sizer.compute_chunk_size(
            prompt_length=2048,
            model_size_gb=20.0,
            available_memory_gb=16.0,
        )
        
        # 3. Layer-specific Quantization
        quantizer = LayerSpecificQuantizer(n_layers=32)
        for i in range(32):
            quantizer.analyzer.record_score(i, i / 32.0)
        quant_configs = quantizer.auto_configure()
        
        # 4. Cache Eviction
        eviction_manager = CacheEvictionManager(
            max_size=1024,
            policy="attention",
            window_size=256,
        )
        
        # 5. Memory Pooling
        pool_manager = MemoryPoolManager()
        kv_pool = pool_manager.create_pool(
            pool_name="kv_cache",
            block_size=4096,
            max_blocks=32,
        )
        
        # Verify alle Komponenten initialisiert
        self.assertIsNotNone(decoder)
        self.assertGreater(chunk_size, 0)
        self.assertEqual(len(quant_configs), 32)
        self.assertGreater(eviction_manager.max_size, 0)
        self.assertGreater(kv_pool.size(), 0)
        
    def test_system_stats(self):
        """Test System-Statistiken."""
        # Setup
        class MockModel:
            def __call__(self, x):
                return [0.1] * 1000
                
        decoder = SpeculativeDecoder(
            draft_model=MockModel(),
            target_model=MockModel(),
            gamma=4,
        )
        
        chunk_sizer = AdaptiveChunkSizer()
        quantizer = LayerSpecificQuantizer(n_layers=32)
        for i in range(32):
            quantizer.analyzer.record_score(i, i / 32.0)
        quantizer.auto_configure()  # Konfiguriere Layers
        eviction_manager = CacheEvictionManager(max_size=1024, policy="lru")
        pool_manager = MemoryPoolManager()
        kv_pool = pool_manager.create_pool("kv", 4096, 32)
        
        # Sammle Statistiken
        stats = {
            "speculative": decoder.get_stats(),
            "chunk_size": chunk_sizer.base_chunk_size,
            "quantization": quantizer.get_summary(),
            "eviction": eviction_manager.get_stats(),
            "memory_pool": pool_manager.get_stats(),
        }
        
        # Verify Statistiken vorhanden
        self.assertIn("acceptance_rate", stats["speculative"])
        self.assertIn("avg_k_bits", stats["quantization"])
        self.assertIn("policy", stats["eviction"])
        self.assertIn("kv", stats["memory_pool"])


if __name__ == "__main__":
    unittest.main()
