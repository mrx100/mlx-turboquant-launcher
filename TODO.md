# MLX Server Optimierungen - Implementierungsplan

## Bereits implementiert ✅

- ✅ **Async Prefill mit `mx.async_eval`** — +214% Cache throughput
- ✅ **TurboQuant KV-Cache Kompression** — 3.6-5.5x Kompression
- ✅ **Hybrid GDN+SDPA Support** — Qwen3.6 optimiert
- ✅ **YaRN Context Extension** — Bis 1M+ Tokens
- ✅ **Persistent Prompt Caching** — Cache über Sessions hinweg
- ✅ **Asymmetric K/V Bits** — K=4-bit, V=2-bit
- ✅ **Logging & Statistics** — Server-Metriken mit Rotation

---

## Optimierungen nach Impact sortiert

### 1. Speculative Decoding (30-50% Speedup)
**Aufwand:** 3-5 Tage  
**Priorität:** HIGH  
**Beschreibung:** Kleineres Draft-Model generiert Tokens, Hauptmodell verifiziert. 3-5x weniger Forward-Passes.  
**Ansatz:** 
- Draft model: Qwen2.5-1.5B oder gpt-oss-2B
- Target model: Qwen3.6-35B oder gpt-oss-20B
- γ=4-6 Tokens pro Draft-Schritt
- Acceptance rate ~60-80%

### 2. Continuous Batching (20-40% Speedup)
**Aufwand:** 5-7 Tage  
**Priorität:** HIGH  
**Beschreibung:** Wie vLLM: Multiple Requests parallel, dynamisches Scheduling.  
**Ansatz:**
- Iteration-level scheduling
- Dynamic batch formation
- Request queue management

### 3. PagedAttention (15-30% Speedup)
**Aufwand:** 3-5 Tage  
**Priorität:** HIGH  
**Beschreibung:** Memory Management wie vLLM: KV-Cache in Pages, weniger Fragmentation.  
**Ansatz:**
- Block-based KV cache allocation
- Page table management
- Dynamic memory reuse

### 4. Cache Eviction Policies (10-20% Speedup)
**Aufwand:** 2-3 Tage  
**Priorität:** MEDIUM  
**Beschreibung:** Intelligente Cache-Verwaltung für lange Kontexte (SnapKV, H2O).  
**Ansatz:**
- Attention-score based eviction
- Sliding window + important tokens
- LRU/LFU fallback

### 5. Layer-spezifische Quantisierung (10-15% Speedup)
**Aufwand:** 2-3 Tage  
**Priorität:** MEDIUM  
**Beschreibung:** Sensible Layer mit mehr Bits, robuste Layer mit weniger Bits.  
**Ansatz:**
- Sensitivity analysis pro Layer
- Adaptive bit allocation
- Profile-guided optimization

### 6. Adaptive Chunk-Size (10-15% Speedup)
**Aufwand:** 1-2 Tage  
**Priorität:** MEDIUM  
**Beschreibung:** Auto-Tuning der Chunk-Size basierend auf Prompt-Länge und Modell.  
**Ansatz:**
- Dynamic chunk sizing based on sequence length
- Hardware-aware optimization
- Profile-based heuristics

### 7. Memory Pooling (5-10% Speedup)
**Aufwand:** 1-2 Tage  
**Priorität:** LOW  
**Beschreibung:** Vor-allozierte Memory-Pools, weniger Allokationen.  
**Ansatz:**
- Pre-allocated KV cache buffers
- Memory reuse across requests
- Pool-based allocation

---

## Test-Strategie

Jede Optimierung wird einzeln implementiert und getestet:
1. Unit Tests für Kernlogik
2. Benchmark-Vergleich (vorher/nachher)
3. Integrationstest mit realem Modell
4. Performance-Metriken dokumentieren

## Benchmark-Suite

- **Token Throughput:** tokens/s bei verschiedenen Prompt-Längen
- **Latency:** Time-to-first-token (TTFT), inter-token latency
- **Memory Usage:** Peak memory, cache efficiency
- **Accuracy:** Perplexity-Vergleich bei quantisierten Caches
