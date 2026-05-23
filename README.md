# MLX TurboQuant Launcher

Scripts zum Starten von MLX-Modellen mit TurboQuant KV-Cache Compression und YaRN Context Extension.

## Features

- **TurboQuant KV-Cache Compression** — 4-bit/3-bit Quantisierung auf Softmax-Layern
- **Hybrid Architecture Support** — Qwen 3.5/3.6 (GDN+SDPA) automatisch erkannt
- **YaRN Context Extension** — Context über native Limits hinaus (bis 1M+ Tokens)
- **OpenAI-kompatibler Server** — `/v1/chat/completions` Endpoint
- **Automatische Architekturerkennung** — Pure SDPA vs Hybrid

## Scripts

### `mlx-turboquant.py`
Python Launcher mit eingebettetem HTTP-Server und TurboQuant Integration.

```bash
# Interaktiv starten
mlx-turboquant.py

# Mit gespeicherten Defaults
mlx-turboquant.py --serve

# Modelle auflisten
mlx-turboquant.py --list
```

### `mlx-server.sh`
Bash Launcher mit `vmlx-engine serve` (ohne TurboQuant).

### `opencode-mlx-turboquant.sh`
Launcher der opencode mit dem TurboQuant-Server verbindet.

## YaRN Context Extension

Für Context über das native Limit hinaus wird YaRN (Yet another RoPE extensioN) verwendet:

| Modell | Native Max | YaRN 2x | YaRN 4x |
|--------|-----------|---------|---------|
| Qwen 3.6-35B | 262K | 512K | 1M |
| gpt-oss-20B | 131K | 256K | 512K |

**Memory-Ersparnis mit TurboQuant (Qwen 35B):**
- 262K: 5.6 GB gespart (75%)
- 1M: 5.6 GB gespart (75% vom KV-Cache)

## Performance

| Metrik | Qwen 3.6-35B (262K) | Qwen 3.6-35B (1M YaRN) |
|--------|---------------------|------------------------|
| Tokens/s | 27.2 | 24.4 |
| First Token | ~3s | ~5s |

## Setup

Benötigt:
- Python 3.13+ mit MLX
- turboquant-mlx: `~/workspace/turboquant-mlx/`
- MLX-Modelle in `~/.lmstudio/models/`
