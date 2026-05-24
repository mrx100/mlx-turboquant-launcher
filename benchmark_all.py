#!/usr/bin/env python3
"""
Benchmark-Suite für alle MLX Optimierungen.

Misst Performance aller implementierten Optimierungen:
- Speculative Decoding
- Adaptive Chunk-Size
- Layer-specific Quantization
- Cache Eviction Policies
- Memory Pooling

Usage:
    python3 benchmark_all.py
    python3 benchmark_all.py --quick
    python3 benchmark_all.py --verbose
"""

import sys
import os
import time
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from speculative_decoding import SpeculativeDecoder, SpeculativePrefillPipeline
from adaptive_chunk_size import AdaptiveChunkSizer, AdaptiveChunkSizeManager
from layer_specific_quantization import LayerSpecificQuantizer, ProfileGuidedOptimizer
from cache_eviction import CacheEvictionManager, CacheEntry
from memory_pooling import MemoryPoolManager, KVCachePool


class MockModel:
    """Mock model für Benchmarks."""
    
    def __init__(self, vocab_size=1000, latency_ms=10):
        self.vocab_size = vocab_size
        self.latency_ms = latency_ms
        self.call_count = 0
        self.total_latency = 0
        
    def __call__(self, x):
        start = time.time()
        self.call_count += 1
        # Simuliere Modell-Latenz
        time.sleep(self.latency_ms / 1000.0)
        self.total_latency += time.time() - start
        return [random.gauss(0, 1) for _ in range(self.vocab_size)]


class BenchmarkResult:
    """Benchmark-Ergebnis."""
    
    def __init__(self, name, metrics):
        self.name = name
        self.metrics = metrics
        
    def __str__(self):
        lines = [f"=== {self.name} ==="]
        for key, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.2f}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def benchmark_speculative_decoding(quick=False, verbose=False):
    """Benchmark Speculative Decoding."""
    results = []
    
    gamma_values = [2, 4, 6, 8] if not quick else [4]
    
    for gamma in gamma_values:
        # Target model langsamer als draft
        draft_model = MockModel(vocab_size=1000, latency_ms=5)
        target_model = MockModel(vocab_size=1000, latency_ms=15)
        
        decoder = SpeculativeDecoder(
            draft_model=draft_model,
            target_model=target_model,
            gamma=gamma,
        )
        
        sampler = lambda logits: 0  # Immer token 0
        
        start = time.time()
        tokens_generated = 0
        
        for token_id, _ in decoder.generate_step(
            prompt_ids=list(range(10)),
            cache=[],
            max_tokens=50,
            sampler=sampler,
        ):
            tokens_generated += 1
            if tokens_generated >= 20:
                break
                
        duration = time.time() - start
        stats = decoder.get_stats()
        
        result = BenchmarkResult(
            f"Speculative Decoding (γ={gamma})",
            {
                "tokens_generated": tokens_generated,
                "duration_s": duration,
                "tokens_per_second": tokens_generated / max(duration, 0.001),
                "acceptance_rate": stats["acceptance_rate"],
                "speedup": stats["speedup"],
                "draft_steps": stats["draft_steps"],
                "target_steps": stats["target_steps"],
            }
        )
        results.append(result)
        
    # Vergleich: Ohne Speculative Decoding
    baseline_model = MockModel(vocab_size=1000, latency_ms=15)
    start = time.time()
    tokens_baseline = 0
    for _ in range(20):
        baseline_model(list(range(10 + tokens_baseline)))
        tokens_baseline += 1
    baseline_duration = time.time() - start
    
    results.append(BenchmarkResult(
        "Baseline (No Speculative)",
        {
            "tokens_generated": tokens_baseline,
            "duration_s": baseline_duration,
            "tokens_per_second": tokens_baseline / max(baseline_duration, 0.001),
        }
    ))
    
    return results


def benchmark_adaptive_chunk_size(quick=False, verbose=False):
    """Benchmark Adaptive Chunk-Size."""
    results = []
    
    sizer = AdaptiveChunkSizer()
    
    test_cases = [
        ("Short Prompt", 128, 10.0, 20.0),
        ("Medium Prompt", 1024, 10.0, 20.0),
        ("Long Prompt", 4096, 10.0, 20.0),
        ("Very Long Prompt", 8192, 10.0, 20.0),
        ("Low Memory", 1024, 20.0, 5.0),
        ("Large Model", 1024, 40.0, 20.0),
        ("Long Sequence", 1024, 10.0, 20.0, 16384),
    ]
    
    if quick:
        test_cases = test_cases[:3]
    
    for case in test_cases:
        name = case[0]
        prompt_len = case[1]
        model_size = case[2]
        mem_avail = case[3]
        seq_len = case[4] if len(case) > 4 else 0
        
        start = time.time()
        iterations = 1000
        for _ in range(iterations):
            chunk_size = sizer.compute_chunk_size(
                prompt_length=prompt_len,
                model_size_gb=model_size,
                available_memory_gb=mem_avail,
                sequence_length=seq_len,
            )
        duration = time.time() - start
        
        results.append(BenchmarkResult(
            f"Adaptive Chunk-Size: {name}",
            {
                "prompt_length": prompt_len,
                "chunk_size": chunk_size,
                "compute_time_us": (duration / iterations) * 1_000_000,
                "iterations": iterations,
            }
        ))
    
    # Profile-Vergleich
    for profile in ["fast", "balanced", "memory_efficient", "low_latency"]:
        sizer.apply_profile(profile)
        chunk_size = sizer.compute_chunk_size(1024, 10.0, 20.0)
        
        results.append(BenchmarkResult(
            f"Profile: {profile}",
            {
                "base_chunk_size": sizer.base_chunk_size,
                "computed_chunk_size": chunk_size,
            }
        ))
    
    return results


def benchmark_layer_quantization(quick=False, verbose=False):
    """Benchmark Layer-specific Quantization."""
    results = []
    
    n_layers_values = [16, 32, 48] if not quick else [32]
    
    for n_layers in n_layers_values:
        quantizer = LayerSpecificQuantizer(n_layers=n_layers)
        
        # Sensitivitäts-Scores setzen
        for i in range(n_layers):
            quantizer.analyzer.record_score(i, i / n_layers)
            
        start = time.time()
        configs = quantizer.auto_configure(
            target_avg_bits=3.5,
            min_bits=2,
            max_bits=6,
        )
        config_time = time.time() - start
        
        summary = quantizer.get_summary()
        
        results.append(BenchmarkResult(
            f"Layer Quantization ({n_layers} layers)",
            {
                "n_layers": n_layers,
                "config_time_ms": config_time * 1000,
                "avg_k_bits": summary["avg_k_bits"],
                "avg_v_bits": summary["avg_v_bits"],
                "compression_ratio": summary["compression_ratio"],
            }
        ))
    
    # Profile-Guided Optimization
    quantizer = LayerSpecificQuantizer(n_layers=32)
    optimizer = ProfileGuidedOptimizer(quantizer)
    
    for i in range(10):
        configs = {j: {"k_bits": 4 + (i % 3), "v_bits": 2} for j in range(32)}
        optimizer.record_profiling_run(
            layer_configs=configs,
            throughput=100 + i * 5,
            quality_score=0.95 - i * 0.005,
            memory_usage_mb=500 - i * 10,
        )
        
    start = time.time()
    optimized = optimizer.optimize(target_quality=0.90)
    optimize_time = time.time() - start
    
    results.append(BenchmarkResult(
        "Profile-Guided Optimization",
        {
            "profiling_runs": 10,
            "optimize_time_ms": optimize_time * 1000,
            "optimized_configs": len(optimized),
        }
    ))
    
    return results


def benchmark_cache_eviction(quick=False, verbose=False):
    """Benchmark Cache Eviction Policies."""
    results = []
    
    policies = ["lru", "lfu", "attention", "sliding_window"]
    if quick:
        policies = ["lru", "attention"]
    
    for policy in policies:
        manager = CacheEvictionManager(
            max_size=1024,
            policy=policy,
            window_size=256,
        )
        
        # Benchmark: Einfügen
        start = time.time()
        n_entries = 500 if quick else 2000
        for i in range(n_entries):
            entry = CacheEntry(
                key=f"key{i}",
                value=f"value{i}",
                position=i,
                attention_score=random.random(),
            )
            manager.put(i, entry)
        insert_duration = time.time() - start
        
        # Benchmark: Abrufen
        start = time.time()
        for i in range(0, n_entries, 2):
            manager.get(i)
        get_duration = time.time() - start
        
        # Benchmark: Eviction
        start = time.time()
        evicted = manager.evict(100)
        eviction_duration = time.time() - start
        
        stats = manager.get_stats()
        
        results.append(BenchmarkResult(
            f"Cache Eviction: {policy}",
            {
                "policy": policy,
                "entries_inserted": n_entries,
                "insert_time_ms": insert_duration * 1000,
                "insert_per_second": n_entries / max(insert_duration, 0.001),
                "get_time_ms": get_duration * 1000,
                "get_per_second": (n_entries // 2) / max(get_duration, 0.001),
                "evicted": len(evicted),
                "eviction_time_ms": eviction_duration * 1000,
                "final_size": stats["size"],
                "utilization": stats["utilization"],
            }
        ))
    
    return results


def benchmark_memory_pooling(quick=False, verbose=False):
    """Benchmark Memory Pooling."""
    results = []
    
    kv_pool = KVCachePool(
        max_batch_size=32,
        max_seq_length=4096,
        n_layers=4,
        n_heads=8,
        head_dim=64,
    )
    
    # Benchmark: Allocate/Free
    start = time.time()
    n_ops = 100 if quick else 500
    for i in range(n_ops):
        cache = kv_pool.allocate_cache(f"req{i % 32}")
        if cache:
            kv_pool.free_cache(f"req{i % 32}")
    duration = time.time() - start
    
    stats = kv_pool.get_stats()
    
    results.append(BenchmarkResult(
        "Memory Pooling: KV-Cache",
        {
            "operations": n_ops,
            "duration_ms": duration * 1000,
            "ops_per_second": n_ops / max(duration, 0.001),
            "pool_size": stats["k_pool_size"],
            "pool_in_use": stats["k_pool_in_use"],
            "pool_utilization": stats["k_pool_utilization"],
        }
    ))
    
    # Vergleich: Verschiedene Pool-Größen
    for max_batch in [4, 8, 16, 32]:
        if quick and max_batch > 8:
            break
            
        pool = KVCachePool(
            max_batch_size=max_batch,
            max_seq_length=512,
            n_layers=2,
            n_heads=4,
            head_dim=32,
        )
        
        start = time.time()
        for i in range(min(max_batch, 10)):
            pool.allocate_cache(f"req{i}")
        alloc_duration = time.time() - start
        
        results.append(BenchmarkResult(
            f"Memory Pool: batch={max_batch}",
            {
                "max_batch": max_batch,
                "alloc_time_ms": alloc_duration * 1000,
                "alloc_per_second": min(max_batch, 10) / max(alloc_duration, 0.001),
            }
        ))
    
    return results


def run_all_benchmarks(quick=False, verbose=False):
    """Alle Benchmarks ausführen."""
    all_results = []
    
    benchmarks = [
        ("Speculative Decoding", benchmark_speculative_decoding),
        ("Adaptive Chunk-Size", benchmark_adaptive_chunk_size),
        ("Layer Quantization", benchmark_layer_quantization),
        ("Cache Eviction", benchmark_cache_eviction),
        ("Memory Pooling", benchmark_memory_pooling),
    ]
    
    for name, func in benchmarks:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            
        results = func(quick=quick, verbose=verbose)
        all_results.extend(results)
        
        if verbose:
            for result in results:
                print(result)
                print()
        else:
            print(f"  {name}: {len(results)} benchmarks done")
    
    return all_results


def print_summary(results):
    """Zusammenfassung ausgeben."""
    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    
    for result in results:
        print(f"\n{result.name}")
        for key, value in result.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLX Optimizations Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print("MLX Optimizations Benchmark Suite")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print()
    
    results = run_all_benchmarks(quick=args.quick, verbose=args.verbose)
    print_summary(results)
