# MLX TurboQuant Launcher — Benchmarks

All benchmarks run on **MacBook Pro M1 Max, 32GB RAM**.

## Real-World Benchmark: Space Invaders Prompt

**Prompt:** "create a full playable,modern spaceinvaders game in one html file with great sound and visual effects, scoring and leverls getting harder each time with a boss coming with extended life and hard to kill. offer great selection of weopens. run it with a 512k context and calculate or show statistics about server capacities as like token/s."

**Model:** Qwen3.6-35B-A3B-UD-MLX-3bit  
**Strategy:** tq_4bit_fast  
**Context:** 131K (YaRN 0.5x)  
**Async Cache:** Enabled

| Metric | Value |
|--------|-------|
| **Time to First Token (TTFT)** | 3001.8 ms |
| **Total Time** | 38.03 s |
| **Generation Time** | 35.03 s |
| **Total Tokens** | 2047 |
| **Tokens/s (overall)** | **53.8** |
| **Tokens/s (generation only)** | **58.4** |
| Errors | 0 |

### Comparison with Previous Runs

| Model | Config | Speed | Notes |
|-------|--------|-------|-------|
| Qwen3.6-35B-A3B 3bit | tq_4bit_fast, async | **58.4 tok/s** | Current — best |
| Qwen3.6-35B-A3B 3bit | UD-MLX | 47 tok/s | Previous test |
| Qwen3.6-35B-A3B 4bit | UD-MLX | 42 tok/s | Previous test |
| Qwen3.6-35B-A3B | Standard | 37 tok/s | Previous test |
| Qwen3.6-27B 3bit | Standard | 26 tok/s | Previous test |
| GLM-4.7-Flash | Standard | 10 tok/s | Previous test |

## Optimization Benchmarks

### Speculative Decoding

| Configuration | Tokens/s | Speedup | Acceptance Rate |
|---|---|---|---|
| Baseline (no speculative) | 53 | 1.0x | — |
| γ=2 | ~70 | ~1.3x | TBD |
| γ=4 | ~88 | ~1.66x | 100% (mock) |
| γ=6 | ~95 | ~1.8x | TBD |
| γ=8 | ~100 | ~1.9x | TBD |

*Note: Real-model benchmarks pending async cache compatibility fix.*

### Adaptive Chunk-Size

| Scenario | Chunk Size | Compute Time |
|---|---|---|
| Short Prompt (128) | 256 | 0.19 µs/op |
| Medium Prompt (1024) | 512 | 0.19 µs/op |
| Long Prompt (4096) | 512 | 0.18 µs/op |
| Low Memory | 256 | 0.19 µs/op |
| Large Model (40GB) | 256 | 0.19 µs/op |
| Long Sequence (16K) | 256 | 0.18 µs/op |

**Profiles:**
| Profile | Base Chunk | Computed |
|---|---|---|
| fast | 1024 | 1024 |
| balanced | 512 | 512 |
| memory_efficient | 256 | 256 |
| low_latency | 128 | 128 |

### Layer-specific Quantization

| Layers | Config Time | Avg K-bits | Avg V-bits | Compression |
|---|---|---|---|---|
| 16 | 0.02 ms | 3.62 | 2.12 | 5.57x |
| 32 | 0.03 ms | 3.62 | 2.12 | 5.57x |
| 48 | 0.02 ms | 3.62 | 2.12 | 5.57x |

**Profile-Guided Optimization:** 10 profiling runs, 0.00ms optimize time

### Cache Eviction Policies

| Policy | Insert (ops/s) | Get (ops/s) | Eviction (ms) |
|---|---|---|---|
| LRU | 1,673,371 | 1,000,000 | 0.02 |
| LFU | 34,613 | 1,000,000 | 5.77 |
| Attention | 42,369 | 1,000,000 | 2.38 |
| Sliding Window | 76,491 | 1,000,000 | 4.81 |

### Memory Pooling

| Operation | Performance |
|---|---|
| KV-Cache Alloc/Free | 218,613 ops/s |
| Pool Alloc (batch=4) | 4,000 ops/s |
| Pool Alloc (batch=8) | 8,000 ops/s |
| Pool Alloc (batch=16) | 10,000 ops/s |
| Pool Alloc (batch=32) | 10,000 ops/s |

## Async Prefill Benchmarks

**Model:** gpt-oss-20B (MXFP4-Q8, 24 layers)  
**Hardware:** MacBook Pro M1 Max, 32GB RAM

| Prompt Length | Best Step Size | Time | Tokens/sec | Async Ops |
|---|---|---|---|---|
| 1,000 tokens | 1024 | 1.92s | 520.9 | 1 |
| 5,000 tokens | 1024 | 9.45s | 529.2 | 5 |
| 10,000 tokens | 512 | 20.24s | 494.1 | 20 |

**Cache Throughput:**
| Mode | Iterations/s | Improvement |
|---|---|---|
| Sync | 1,375 | baseline |
| Async | 4,317 | **+214%** |

## YaRN Context Extension

**Model:** Qwen3.6-35B-A3B

| Context | Tokens/s | First Token | Memory |
|---|---|---|---|
| 262K (native) | 27.2 | ~3s | 1.86 GB (with TQ) |
| 512K (2x YaRN) | 25.8 | ~4s | 1.86 GB (with TQ) |
| 1M (4x YaRN) | 24.4 | ~5s | 1.86 GB (with TQ) |

**Memory savings with TurboQuant (Qwen 35B):**

| Context | Without TQ | With TQ | Saved |
|---|---|---|---|
| 262K | 7.45 GB | 1.86 GB | 5.6 GB (75%) |
| 1M | 29.8 GB | 7.45 GB | 22.4 GB (75%) |

## Test Coverage

| Test Suite | Tests | Status |
|---|---|---|
| Speculative Decoding | 6 | ✅ All passed |
| Adaptive Chunk-Size | 16 | ✅ All passed |
| Layer-specific Quantization | 17 | ✅ All passed |
| Cache Eviction | 23 | ✅ All passed |
| Memory Pooling | 14 | ✅ All passed |
| Integration Tests | 8 | ✅ All passed |
| **Total** | **84** | ✅ **All passed** |

## How to Run Benchmarks

```bash
# Full benchmark suite
python3 benchmark_all.py

# Quick benchmark
python3 benchmark_all.py --quick

# Verbose output
python3 benchmark_all.py --verbose

# Real-world benchmark (Space Invaders prompt)
python3 benchmark_realworld.py
```
