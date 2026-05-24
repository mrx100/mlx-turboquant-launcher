#!/Users/hasan/workspace/mlx-studio/.venv/bin/python3
"""TurboQuant MLX Launcher & Server — single script.

Supports:
- Pure SDPA models (Llama, Mistral, Gemma): full TurboQuant KV-cache compression
- Hybrid GDN+SDPA models (Qwen 3.5, Qwen 3.6): TurboQuant on softmax layers only
- YaRN context extension beyond native limits (up to 1M+ tokens)

Usage:
  mlx-turboquant.py              interactive launcher
  mlx-turboquant.py --list       list available models
  mlx-turboquant.py --serve      start with saved defaults (non-interactive)
"""

# Auto-switch to venv if mlx_lm not available
try:
    import mlx_lm
except ImportError:
    import os, subprocess, sys
    _venv = os.path.expanduser("~/workspace/mlx-studio/.venv/bin/python3")
    if os.path.exists(_venv):
        os.execv(_venv, [_venv] + sys.argv)
    else:
        print("✗ mlx_lm not found. Install it or set the venv path in the script.")
        sys.exit(1)

import os, sys, json, time, uuid, asyncio, re, shutil, atexit, signal, logging, traceback
from pathlib import Path
from datetime import datetime

# ── Logging Setup ───────────────────────────────────────────────────

_LOG_DIR = Path.home() / ".mlx-turboquant"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "server.log"
_STATS_FILE = _LOG_DIR / "stats.json"

# Rotating file handler (10MB max, 5 backups)
class RotatingFileHandler(logging.FileHandler):
    def __init__(self, filename, max_bytes=10*1024*1024, backup_count=5):
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        super().__init__(filename, mode='a')

    def emit(self, record):
        if self.stream and hasattr(self.stream, 'tell'):
            try:
                self.stream.seek(0, 2)
                if self.stream.tell() >= self.max_bytes:
                    self.stream.close()
                    self._rotate()
                    self.stream = self._open()
            except: pass
        super().emit(record)

    def _rotate(self):
        for i in range(self.backup_count, 0, -1):
            src = Path(f"{self.baseFilename}.{i}")
            dst = Path(f"{self.baseFilename}.{i+1}")
            if src.exists():
                src.rename(dst)
        Path(self.baseFilename).rename(Path(f"{self.baseFilename}.1"))

logger = logging.getLogger("turboquant")
logger.setLevel(logging.DEBUG)
fh = RotatingFileHandler(str(_LOG_FILE), max_bytes=10*1024*1024, backup_count=5)
fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(fh)

# Console handler for important messages
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ch)

# ── Server Statistics ──────────────────────────────────────────────

class ServerStats:
    """Tracks server performance metrics."""

    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_errors = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.connection_resets = 0
        self.bad_requests = 0
        self.avg_tokens_per_second = 0.0
        self.last_request_time = None
        self.last_error = None
        self.requests_by_status = {"200": 0, "400": 0, "500": 0}

    def record_request(self, status_code, tokens=0, duration=0):
        self.total_requests += 1
        self.last_request_time = time.time()
        status_key = str(status_code)
        if status_key in self.requests_by_status:
            self.requests_by_status[status_key] += 1
        if tokens > 0:
            self.total_tokens_generated += tokens
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.avg_tokens_per_second = self.total_tokens_generated / elapsed

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_cache_miss(self):
        self.cache_misses += 1

    def record_error(self, error_msg):
        self.total_errors += 1
        self.last_error = error_msg
        self.requests_by_status["500"] += 1
        logger.error(f"Error #{self.total_errors}: {error_msg}")

    def record_bad_request(self):
        self.bad_requests += 1
        self.requests_by_status["400"] += 1

    def record_connection_reset(self):
        self.connection_resets += 1

    def get_summary(self):
        uptime = time.time() - self.start_time
        hours = int(uptime // 3600)
        mins = int((uptime % 3600) // 60)
        secs = int(uptime % 60)
        return {
            "uptime": f"{hours}h {mins}m {secs}s",
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "avg_tokens_per_second": round(self.avg_tokens_per_second, 1),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{self.cache_hits / max(self.cache_hits + self.cache_misses, 1) * 100:.1f}%",
            "errors": self.total_errors,
            "connection_resets": self.connection_resets,
            "bad_requests": self.bad_requests,
            "requests_by_status": dict(self.requests_by_status),
            "last_error": self.last_error,
        }

    def save(self):
        try:
            _STATS_FILE.write_text(json.dumps(self.get_summary(), indent=2))
        except: pass

    def print_summary(self):
        s = self.get_summary()
        logger.info("=" * 60)
        logger.info("  SERVER STATISTICS")
        logger.info("=" * 60)
        logger.info(f"  Uptime:          {s['uptime']}")
        logger.info(f"  Total Requests:  {s['total_requests']}")
        logger.info(f"  Tokens Generated: {s['total_tokens_generated']}")
        logger.info(f"  Avg Tokens/s:    {s['avg_tokens_per_second']}")
        logger.info(f"  Cache Hits:      {s['cache_hits']} ({s['cache_hit_rate']})")
        logger.info(f"  Cache Misses:    {s['cache_misses']}")
        logger.info(f"  Errors:          {s['errors']}")
        logger.info(f"  Connection Resets: {s['connection_resets']}")
        logger.info(f"  Bad Requests:    {s['bad_requests']}")
        if s.get('last_error'):
            logger.info(f"  Last Error:      {s['last_error']}")
        logger.info("=" * 60)


_server_stats = ServerStats()

_MODELS_DIR = Path.home() / ".lmstudio/models"
_CONFIG_FILE = Path.home() / ".tq_defaults.json"
_TQ_DIR = Path.home() / "workspace" / "turboquant-mlx"

_DEFAULTS = {
    "model": None, "port": 8081, "host": "0.0.0.0",
    "max_tokens": 2048, "temperature": 0.7,
    "tq_strategy": "v2_4bit_rotated", "ctx_size": 8192,
    "k_bits": 4, "v_bits": 2, "kv_group_size": 64,
    "quantized_kv_start": 512,
    "prompt_cache_dir": None,
    "use_rotation": True, "use_normalization": True,
}

_CONTEXT_PRESETS = [
    ("8192", "8K"),
    ("16384", "16K"),
    ("32768", "32K"),
    ("65536", "64K"),
    ("131072", "128K"),
    ("262144", "262K (native max)"),
    ("524288", "512K (2x YaRN)"),
    ("1048576", "1M (4x YaRN)"),
    ("2097152", "2M (8x YaRN)"),
]

_NATIVE_MAX_CONTEXT = 262144

_STRATEGIES_PURE = {
    "1": ("none", "No KV-Cache quantization (standard mlx-lm)"),
    "2": ("v2_4bit_lean", "V2 4-bit LEAN — fastest, 3.6x compression"),
    "3": ("v2_4bit_rotated", "V2 4-bit rotated — best 4-bit quality, 3.6x compression (Recommended)"),
    "4": ("v2_3bit_qjl", "V2 3-bit rot+QJL — best for D=256 models, 4.7x compression"),
    "5": ("v3_35bit_mixed", "V3 3.5-bit mixed — near-lossless, 4.1x compression"),
    "6": ("v3_3bit_lloyd", "V3 3-bit Lloyd-Max — balanced, 4.7x compression"),
    "7": ("v3_25bit_mixed", "V3 2.5-bit mixed — aggressive, 5.5x compression"),
}

_STRATEGIES_HYBRID = {
    "1": ("none", "No KV-Cache quantization (standard mlx-lm)"),
    "2": ("tq_4bit", "TurboQuant 4-bit + rotation on softmax layers (Recommended)"),
    "3": ("tq_4bit_fast", "TurboQuant 4-bit LEAN on softmax layers (faster)"),
    "4": ("tq_3bit", "TurboQuant 3-bit + rotation + QJL on softmax layers"),
}


# ── GDN Pass-Through Cache ─────────────────────────────────────────

class GDNPassThroughCache:
    """Stub cache for Gated DeltaNet (linear attention) layers.

    GDN layers manage their own conv state and recurrent state internally.
    This stub satisfies the cache interface expected by mlx-lm's generation
    loop while allocating zero memory.
    """
    offset = 0
    lengths = None

    def __init__(self):
        self._state = [None, None]

    def __getitem__(self, idx):
        return self._state[idx]

    def __setitem__(self, idx, val):
        self._state[idx] = val

    def advance(self, n):
        self.offset += n

    @property
    def state(self):
        return []

    @state.setter
    def state(self, v):
        pass

    @property
    def nbytes(self):
        return 0

    def empty(self):
        return True

    def is_trimmable(self):
        return False

    def trim(self, n):
        return 0

    @property
    def nbytes_equivalent_fp16(self):
        return 0

    def make_mask(self, *args, **kwargs):
        return None


# ── Architecture Detection ──────────────────────────────────────────

def _detect_architecture(model):
    """Detect model attention architecture.

    Returns: (is_hybrid, softmax_indices, gdn_indices)
    """
    layers = model.layers
    softmax_indices, gdn_indices = [], []

    for i, layer in enumerate(layers):
        if hasattr(layer, 'is_linear'):
            if layer.is_linear:
                gdn_indices.append(i)
            else:
                softmax_indices.append(i)
        else:
            softmax_indices.append(i)

    is_hybrid = len(gdn_indices) > 0
    return is_hybrid, softmax_indices, gdn_indices


# ── Hybrid Cache Factory ────────────────────────────────────────────

def _create_hybrid_cache(model, strategy, head_dim, k_bits=4, v_bits=2):
    """Create cache list appropriate for the model's architecture.

    Pure SDPA: all layers get TurboQuant caches.
    Hybrid GDN+SDPA: softmax layers get TurboQuant, GDN layers get stubs.

    Supports asymmetric K/V bit allocation for optimized memory/quality.
    """
    import mlx.core as mx
    import mlx_lm
    from mlx_lm.models.cache import make_prompt_cache
    from turboquant.cache import TurboQuantKVCache
    from turboquant.cache_v2 import TurboQuantKVCacheV2
    from turboquant.cache_v3 import TurboQuantKVCacheV3

    is_hybrid, softmax_indices, gdn_indices = _detect_architecture(model)
    n_layers = len(model.layers)

    if strategy == "none":
        return make_prompt_cache(model), is_hybrid, softmax_indices, gdn_indices

    if is_hybrid:
        # Hybrid model: TurboQuant on softmax layers, stubs on GDN layers
        cache_list = []
        for i in range(n_layers):
            if i in softmax_indices:
                if strategy == "tq_4bit":
                    cache = TurboQuantKVCacheV2(
                        head_dim=head_dim, bits=4, group_size=64,
                        use_rotation=True, use_normalization=True,
                        k_bits=k_bits, v_bits=v_bits,
                    )
                elif strategy == "tq_4bit_fast":
                    cache = TurboQuantKVCacheV2(
                        head_dim=head_dim, bits=4, group_size=64,
                        use_rotation=False, use_normalization=False,
                        k_bits=k_bits, v_bits=v_bits,
                    )
                elif strategy == "tq_3bit":
                    cache = TurboQuantKVCacheV2(
                        head_dim=head_dim, bits=3, group_size=64,
                        use_rotation=True, use_normalization=True,
                        use_qjl=True,
                        k_bits=k_bits, v_bits=v_bits,
                    )
                else:
                    cache = TurboQuantKVCacheV2(
                        head_dim=head_dim, bits=4, group_size=64,
                        use_rotation=True, use_normalization=True,
                        k_bits=k_bits, v_bits=v_bits,
                    )
            else:
                cache = GDNPassThroughCache()
            cache_list.append(cache)
        return cache_list, is_hybrid, softmax_indices, gdn_indices

    # Pure SDPA model: all layers get TurboQuant
    opts = {
        "v2_4bit_lean": lambda: [TurboQuantKVCache(head_dim=head_dim, mse_bits=4, use_qjl=False) for _ in range(n_layers)],
        "v2_4bit_rotated": lambda: [TurboQuantKVCache(head_dim=head_dim, mse_bits=4, use_qjl=True) for _ in range(n_layers)],
        "v2_3bit_qjl": lambda: [TurboQuantKVCache(head_dim=head_dim, mse_bits=3, use_qjl=True) for _ in range(n_layers)],
        "v3_35bit_mixed": lambda: [TurboQuantKVCacheV3(head_dim=head_dim, bits=3, n_outlier=64, outlier_bits=4) for _ in range(n_layers)],
        "v3_3bit_lloyd": lambda: [TurboQuantKVCacheV3(head_dim=head_dim, bits=3) for _ in range(n_layers)],
        "v3_25bit_mixed": lambda: [TurboQuantKVCacheV3(head_dim=head_dim, bits=2, n_outlier=64, outlier_bits=3) for _ in range(n_layers)],
    }
    fn = opts.get(strategy)
    return (fn() if fn else make_prompt_cache(model)), is_hybrid, softmax_indices, gdn_indices


# ── Patch model modules (fix import aliasing bug in mlx-lm 0.31+) ───

def _patch_model_modules(patched_sdpa):
    import mlx_lm.models
    for name in dir(mlx_lm.models):
        if name.startswith('_'):
            continue
        mod = getattr(mlx_lm.models, name, None)
        if mod is None:
            continue
        if hasattr(mod, 'scaled_dot_product_attention'):
            if mod.scaled_dot_product_attention is not patched_sdpa:
                mod.scaled_dot_product_attention = patched_sdpa


# ── helpers ────────────────────────────────────────────────────────

def _fmt_size(b):
    if b >= 1_073_741_824: return f"{b/1_073_741_824:.1f} GB"
    if b >= 1_048_576: return f"{b/1_048_576:.0f} MB"
    return f"{b} B"


def _scan_models():
    models, seen = [], set()
    if not _MODELS_DIR.exists(): return models
    for sft in sorted(_MODELS_DIR.rglob("*.safetensors")):
        d = sft.parent
        if d == _MODELS_DIR or d.parent == _MODELS_DIR or str(d) in seen: continue
        seen.add(str(d))
        org, name = d.parent.name, d.name
        sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        models.append({"label": f"{org}/{name}", "path": str(d), "size": sz})
    return models


def _read_config():
    cfg = dict(_DEFAULTS)
    if _CONFIG_FILE.exists():
        try: cfg.update(json.loads(_CONFIG_FILE.read_text()))
        except Exception: pass
    return cfg


def _save_config(cfg):
    _CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    print(f"✓ defaults saved to {_CONFIG_FILE}")


def _pick_model(models, current=None):
    paths = [m["path"] for m in models]
    if current and current in paths:
        i = paths.index(current)
        print(f"  Saved default: {models[i]['label']}")
        if input("  Use this model? [Y/n]: ").strip().lower() != "n":
            return models[i]
    print()
    for i, m in enumerate(models, 1):
        print(f"  {i:2d}) {m['label']} ({_fmt_size(m['size'])})")
    while True:
        try:
            idx = int(input(f"\n  Select model [1-{len(models)}]: ")) - 1
            if 0 <= idx < len(models): return models[idx]
        except ValueError: pass
        print("  Invalid selection")


def _pick_strategy(cur=None, is_hybrid=False):
    strategies = _STRATEGIES_HYBRID if is_hybrid else _STRATEGIES_PURE
    print()
    for k, (n, d) in strategies.items():
        m = " (current)" if n == cur else ""
        print(f"  {k}) [{n}] — {d}{m}")
    default = "2"
    while True:
        s = input(f"  Strategy [{default}]: ").strip() or default
        if s in strategies: return strategies[s][0]
        print("  Invalid selection")


def _pick_int(prompt, default):
    v = input(f"  {prompt} [{default}]: ").strip()
    try: return int(v) if v else default
    except: return default


def _pick_float(prompt, default):
    v = input(f"  {prompt} [{default}]: ").strip()
    try: return float(v) if v else default
    except: return default


def _pick_kv_bits(cfg, is_turboquant=False):
    """Select asymmetric K/V cache quantization bits.

    For TurboQuant strategies: K and V must use same bits (mx.quantized_matmul requirement).
    For MLX native (strategy=none): K and V can differ (K=4-bit, V=2-bit recommended).
    """
    k_bits = cfg.get("k_bits", 4)
    v_bits = cfg.get("v_bits", 2)

    if is_turboquant:
        # TurboQuant requires symmetric K/V bits for mx.quantized_matmul
        print()
        print("  KV-Cache quantization (TurboQuant — K/V must match for quantized_matmul):")
        kv_options = {"1": (2, "2-bit — 8x compression (aggressive)"), "2": (3, "3-bit — 5.3x compression"), "3": (4, "4-bit — 4x compression (Recommended)"), "4": (8, "8-bit — 2x compression (highest quality)")}
        for k, (v, d) in kv_options.items():
            m = " (current)" if v == k_bits else ""
            print(f"  {k}) [{v}-bit] — {d}{m}")
        kv_choice = input(f"  KV bits [3]: ").strip() or "3"
        bits = kv_options.get(kv_choice, (4, ""))[0]
        cfg["k_bits"] = bits
        cfg["v_bits"] = bits
    else:
        # MLX native supports asymmetric K/V
        print()
        print("  Asymmetric KV-Cache quantization (MLX native):")
        print("  Keys (K) are sensitive to quantization; Values (V) tolerate lower bits.")
        print()
        print("  K-bit options (Keys — used in softmax Q·K^T, needs precision):")
        k_options = {"1": (2, "2-bit — 8x compression (aggressive)"), "2": (4, "4-bit — 4x compression (Recommended)"), "3": (8, "8-bit — 2x compression (highest quality)")}
        for k, (v, d) in k_options.items():
            m = " (current)" if v == k_bits else ""
            print(f"  {k}) [{v}-bit K] — {d}{m}")
        k_choice = input(f"  K bits [2]: ").strip() or "2"
        cfg["k_bits"] = k_options.get(k_choice, (4, ""))[0]

        print()
        print("  V-bit options (Values — weighted sums, more tolerant):")
        v_options = {"1": (2, "2-bit — 8x compression (Recommended)"), "2": (3, "3-bit — 5.3x compression"), "3": (4, "4-bit — 4x compression"), "4": (8, "8-bit — 2x compression (highest quality)")}
        for k, (v, d) in v_options.items():
            m = " (current)" if v == v_bits else ""
            print(f"  {k}) [{v}-bit V] — {d}{m}")
        v_choice = input(f"  V bits [1]: ").strip() or "1"
        cfg["v_bits"] = v_options.get(v_choice, (2, ""))[0]

    cfg["kv_group_size"] = cfg.get("kv_group_size", 64)


def _pick_quantized_kv_start(cfg):
    """Select when to start KV quantization (delays compression for better prefill)."""
    qkv_start = cfg.get("quantized_kv_start", 512)

    print()
    print("  Quantized KV Start (delays compression to stabilize prefill):")
    print("  First N tokens stay in FP16, then quantization kicks in.")
    qkv_options = {"1": (0, "0 — Quantize from token 1 (max compression)"), "2": (256, "256 — Short system prompts stay precise"), "3": (512, "512 — Recommended for long contexts"), "4": (1024, "1024 — Maximum prefill quality")}
    for k, (v, d) in qkv_options.items():
        m = " (current)" if v == qkv_start else ""
        print(f"  {k}) [{v}] — {d}{m}")
    qkv_choice = input(f"  Quantized KV start [3]: ").strip() or "3"
    cfg["quantized_kv_start"] = qkv_options.get(qkv_choice, (512, ""))[0]


def _pick_prompt_cache(cfg):
    """Select persistent prompt cache directory for long sessions."""
    cache_dir = cfg.get("prompt_cache_dir", None)
    print()
    print("  Persistent prompt cache (survives server restarts, near-zero TTFT):")
    default_dir = str(Path.home() / ".mlx_prompt_cache")
    print(f"  Cache directory: {default_dir}")
    print(f"  Current: {cache_dir or 'disabled'}")
    choice = input(f"  Enable prompt cache? [Y/n]: ").strip().lower()
    if choice != "n":
        cfg["prompt_cache_dir"] = default_dir
    else:
        cfg["prompt_cache_dir"] = None


def _pick_context_size(cfg, model_path):
    """Select context size with YaRN extension support."""
    import json
    from pathlib import Path

    config_path = Path(model_path) / "config.json"
    native_max = _NATIVE_MAX_CONTEXT
    if config_path.exists():
        try:
            cfg_data = json.loads(config_path.read_text())
            native_max = cfg_data.get("max_position_embeddings", _NATIVE_MAX_CONTEXT)
        except: pass

    print()
    print("  Context size presets:")
    current = cfg.get("ctx_size", 8192)
    for i, (val, label) in enumerate(_CONTEXT_PRESETS, 1):
        v = int(val)
        marker = " (current)" if v == current else ""
        yaarn_note = " — YaRN extended" if v > native_max else ""
        print(f"  {i:2d}) [{label}]{yaarn_note}{marker}")

    while True:
        try:
            idx = int(input(f"  Select context size [6]: ").strip() or "6") - 1
            if 0 <= idx < len(_CONTEXT_PRESETS):
                return int(_CONTEXT_PRESETS[idx][0])
        except ValueError: pass
        print("  Invalid selection")


def _apply_yarn_override(model_path, desired_ctx):
    """Create a temporary config with YaRN rope_scaling if context > native max."""
    import json, tempfile
    from pathlib import Path

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return model_path, None

    original_cfg = json.loads(config_path.read_text())
    native_max = original_cfg.get("max_position_embeddings", _NATIVE_MAX_CONTEXT)

    if desired_ctx <= native_max:
        return model_path, None

    yarn_factor = desired_ctx / native_max
    rope_params = original_cfg.get("rope_parameters", {})
    rope_params["type"] = "yarn"
    rope_params["factor"] = yarn_factor
    rope_params["original_max_position_embeddings"] = native_max
    rope_params.setdefault("beta_fast", 32)
    rope_params.setdefault("beta_slow", 1)
    rope_params.setdefault("mscale", 1.0)
    rope_params.setdefault("mscale_all_dim", 0.0)

    # Also update rope_scaling if present (for models with existing YaRN)
    if "rope_scaling" in original_cfg and original_cfg["rope_scaling"]:
        original_cfg["rope_scaling"]["factor"] = yarn_factor
        original_cfg["rope_scaling"]["original_max_position_embeddings"] = native_max

    original_cfg["max_position_embeddings"] = desired_ctx
    original_cfg["rope_parameters"] = rope_params

    tmp_dir = Path(tempfile.mkdtemp(prefix="yarn_config_"))

    def cleanup():
        try: shutil.rmtree(tmp_dir, ignore_errors=True)
        except: pass
    atexit.register(cleanup)

    for f in config_path.parent.iterdir():
        if f.name != "config.json" and f.is_file():
            (tmp_dir / f.name).symlink_to(f)

    tmp_config = tmp_dir / "config.json"
    tmp_config.write_text(json.dumps(original_cfg, indent=2))

    print(f"  YaRN override: factor={yarn_factor:.1f}x, context={desired_ctx}")
    return str(tmp_dir), str(tmp_dir)


def _kill_existing():
    import subprocess
    for proc in ("mlx-turboquant.py",):
        try:
            r = subprocess.run(["pgrep", "-f", proc], capture_output=True, text=True)
            if r.stdout.strip():
                pids = r.stdout.strip().split()
                my_pid = str(os.getpid())
                pids = [p for p in pids if p != my_pid]
                if pids:
                    print(f"  Killing existing {proc} instances: {' '.join(pids)}")
                    subprocess.run(["kill"] + pids, capture_output=True)
                    time.sleep(0.5)
                    subprocess.run(["kill", "-9"] + pids, capture_output=True)
        except: pass


def _check_port(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def _strip_thinking(text):
    """Remove thinking blocks from Qwen-style output."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    if '</think>' in text:
        text = text.split('</think>', 1)[-1].strip()
    match = re.search(
        r"(?:Here's\s+a\s+)?[Tt]hinking\s+[Pp]rocess:\s*\n((?:\s*\d+\.\s+.*\n)+)(.*)",
        text, re.DOTALL
    )
    if match:
        rest = match.group(2).strip()
        if rest:
            return rest
    return text.strip()


# ── server mode ────────────────────────────────────────────────────

def _start_server(cfg):
    sys.path.insert(0, str(_TQ_DIR))
    sys.path.insert(0, str(_TQ_DIR / "turboquant"))

    import mlx.core as mx
    import mlx_lm
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    import mlx_lm.models.base as _base
    import turboquant.patch as tq_patch
    from turboquant.cache import TurboQuantKVCache
    from turboquant.cache_v2 import TurboQuantKVCacheV2
    from turboquant.cache_v3 import TurboQuantKVCacheV3

    # Apply TurboQuant patch with v2 fix (patch model modules too)
    tq_patch.apply()
    _patch_model_modules(_base.scaled_dot_product_attention)

    model_path = cfg["model"]
    strategy = cfg["tq_strategy"]
    host, port = cfg["host"], cfg["port"]
    max_tokens, temperature = cfg["max_tokens"], cfg["temperature"]
    k_bits = cfg.get("k_bits", 4)
    v_bits = cfg.get("v_bits", 2)
    kv_group_size = cfg.get("kv_group_size", 64)
    quantized_kv_start = cfg.get("quantized_kv_start", 512)
    prompt_cache_dir = cfg.get("prompt_cache_dir", None)

    # Persistent prompt cache setup
    _prompt_cache = {"dir": None, "loaded": False}
    if prompt_cache_dir:
        _prompt_cache["dir"] = Path(prompt_cache_dir)
        _prompt_cache["dir"].mkdir(parents=True, exist_ok=True)
        print(f"Prompt cache enabled: {_prompt_cache['dir']}")

    print(f"Loading model: {model_path}")
    model, tokenizer = mlx_lm.load(model_path)
    n_layers = len(model.layers)

    is_hybrid, softmax_indices, gdn_indices = _detect_architecture(model)

    # Get head_dim from softmax layer (layer 0 for pure SDPA, first softmax for hybrid)
    head_dim = None
    if is_hybrid and softmax_indices:
        attn = model.layers[softmax_indices[0]].self_attn
        head_dim = getattr(attn, 'head_dim', None)
    elif not is_hybrid:
        try:
            head_dim = model.layers[0].self_attn.head_dim
        except AttributeError:
            args = getattr(model, 'args', None)
            tc = getattr(args, 'text_config', None)
            if tc and isinstance(tc, dict):
                head_dim = tc.get('head_dim')
            elif args:
                head_dim = getattr(args, 'head_dim', None)
            if head_dim is None:
                cfg_path = Path(model_path) / "config.json"
                if cfg_path.exists():
                    cfg_data = json.loads(cfg_path.read_text())
                    head_dim = cfg_data.get('head_dim')
                    if head_dim is None:
                        tc = cfg_data.get('text_config', {})
                        if isinstance(tc, dict):
                            head_dim = tc.get('head_dim')
                    # GLM/DeepSeek: qk_nope_head_dim + qk_rope_head_dim
                    if head_dim is None:
                        nope = cfg_data.get('qk_nope_head_dim')
                        rope = cfg_data.get('qk_rope_head_dim')
                        if nope and rope:
                            head_dim = nope + rope
                    if head_dim is None:
                        hidden = cfg_data.get('hidden_size')
                        n_heads = cfg_data.get('num_attention_heads')
                        if hidden and n_heads:
                            head_dim = hidden // n_heads

    if head_dim is None:
        print("✗ Could not determine head_dim")
        sys.exit(1)

    arch_label = "Hybrid GDN+SDPA" if is_hybrid else "Pure SDPA"
    ctx_size = cfg.get("ctx_size", 8192)
    if ctx_size > _NATIVE_MAX_CONTEXT:
        arch_label += f" + YaRN ({ctx_size/_NATIVE_MAX_CONTEXT:.1f}x)"
    elif ctx_size > 0 and ctx_size != _NATIVE_MAX_CONTEXT:
        pass  # Within native range

    print(f"Model loaded: {n_layers} layers, head_dim={head_dim}")
    if is_hybrid:
        print(f"  Architecture: {arch_label}")
        print(f"  Softmax layers: {len(softmax_indices)} (compressed)")
        print(f"  GDN layers: {len(gdn_indices)} (skipped)")

    async def safe_write(w, data):
        """Write to client with connection error handling."""
        try:
            w.write(data)
            await w.drain()
            return True
        except (ConnectionResetError, BrokenPipeError, OSError):
            return False

    async def handle(r, w):
        req_start = time.time()
        try:
            data = await asyncio.wait_for(r.read(65536), timeout=300)
            if not data: return
            req = data.decode("utf-8", errors="replace")
            lines = req.split("\r\n")
            parts = lines[0].split()
            method, path = (parts[0], parts[1]) if len(parts) >= 2 else ("", "")
            body_start = None
            for i, line in enumerate(lines[1:], 1):
                if line == "":
                    body_start = i + 1
                    break
            body = "\r\n".join(lines[body_start:]) if body_start else ""

            if path == "/health" or path == "/":
                resp = json.dumps({
                    "status": "ok",
                    "model": Path(model_path).name,
                    "strategy": strategy,
                    "architecture": arch_label,
                    "is_hybrid": is_hybrid,
                    "softmax_layers": len(softmax_indices) if is_hybrid else n_layers,
                    "gdn_layers": len(gdn_indices),
                    "head_dim": head_dim,
                    "k_bits": k_bits,
                    "v_bits": v_bits,
                    "quantized_kv_start": quantized_kv_start,
                    "prompt_cache": bool(_prompt_cache["dir"]),
                    "stats": _server_stats.get_summary(),
                })
                rdata = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(resp)}\r\n\r\n{resp}"
                w.write(rdata.encode()); await w.drain()
                _server_stats.record_request(200)
                return

            if path == "/v1/models":
                resp = json.dumps({"object": "list", "data": [{
                    "id": Path(model_path).name, "object": "model", "owned_by": "turboquant",
                }]})
                rdata = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(resp)}\r\n\r\n{resp}"
                w.write(rdata.encode()); await w.drain()
                _server_stats.record_request(200)
                return

            if path == "/stats" and method == "GET":
                resp = json.dumps(_server_stats.get_summary(), indent=2)
                rdata = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(resp)}\r\n\r\n{resp}"
                w.write(rdata.encode()); await w.drain()
                return

            if path == "/v1/chat/completions" and method == "POST":
                try: payload = json.loads(body)
                except:
                    _server_stats.record_bad_request()
                    logger.warning(f"Bad request: Invalid JSON")
                    w.write(b"HTTP/1.1 400 Bad Request\r\n\r\nInvalid JSON"); await w.drain(); return

                msgs = payload.get("messages", [])
                if not msgs:
                    _server_stats.record_bad_request()
                    logger.warning(f"Bad request: No messages")
                    w.write(b"HTTP/1.1 400 Bad Request\r\n\r\nNo messages"); await w.drain(); return

                mt = payload.get("max_tokens") or max_tokens
                tp = payload.get("temperature") or temperature
                stream = payload.get("stream", False)
                cache, _, _, _ = _create_hybrid_cache(model, strategy, head_dim, k_bits, v_bits)

                # Apply MLX native KV quantization (works with ALL models)
                if strategy == "none" and k_bits > 0:
                    try:
                        from mlx_lm.generate import maybe_quantize_kv_cache
                        maybe_quantize_kv_cache(cache, quantized_kv_start, kv_group_size, k_bits)
                    except Exception as e:
                        print(f"  ⚠ KV quantization not supported for this model: {e}")
                        cache, _, _, _ = _create_hybrid_cache(model, strategy, head_dim, k_bits, v_bits)

                formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                prompt_ids = mx.array(tokenizer.encode(formatted))

                # Prompt cache: try to load persisted cache
                if _prompt_cache["dir"]:
                    import hashlib
                    prompt_hash = hashlib.sha256(formatted.encode()).hexdigest()[:16]
                    cache_file = _prompt_cache["dir"] / f"{prompt_hash}.safetensors"
                    if cache_file.exists():
                        try:
                            from mlx_lm.models.cache import load_prompt_cache
                            loaded = load_prompt_cache(str(cache_file))
                            if loaded and len(loaded) == len(cache):
                                for i, c in enumerate(loaded):
                                    if hasattr(c, 'offset') and hasattr(cache[i], 'offset'):
                                        cache[i].offset = c.offset
                                        if hasattr(c, 'keys') and c.keys is not None:
                                            cache[i].keys = c.keys
                                            cache[i].values = c.values
                                logger.info(f"Loaded prompt cache: {cache_file.name}")
                                _server_stats.record_cache_hit()
                                _prompt_cache["loaded"] = True
                        except Exception as e:
                            logger.warning(f"Failed to load prompt cache: {e}")
                            _server_stats.record_cache_miss()
                    else:
                        _server_stats.record_cache_miss()

                def _sampler(logits):
                    if tp == 0:
                        return mx.argmax(logits, axis=-1)
                    return mx.random.categorical(logits * (1.0 / tp))

                cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                eos_ids = {tokenizer.eos_token_id}

                if stream:
                    header = f"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n"
                    if not await safe_write(w, header.encode()): return

                    tokens_out = []
                    thinking_done = False
                    thinking_buffer = ""

                    for tok, _ in generate_step(
                        prompt=prompt_ids, model=model, max_tokens=mt, sampler=_sampler,
                        prompt_cache=cache, quantized_kv_start=quantized_kv_start
                    ):
                        tid = int(tok)
                        if tid in eos_ids: break
                        tokens_out.append(tid)
                        chunk_text = tokenizer.decode([tid])

                        if not thinking_done:
                            thinking_buffer += chunk_text
                            if '</think>' in thinking_buffer:
                                thinking_done = True
                                chunk_text = thinking_buffer.split('</think>', 1)[-1]
                                if chunk_text.strip():
                                    chunk = json.dumps({
                                        "id": cmpl_id, "object": "chat.completion.chunk",
                                        "created": int(time.time()), "model": Path(model_path).name,
                                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": chunk_text}, "finish_reason": None}],
                                    })
                                    if not await safe_write(w, f"data: {chunk}\n\n".encode()): break
                            elif re.match(r"^(?:.*\n)?(?:Here's\s+a\s+)?[Tt]hinking\s+[Pp]rocess:\s*\n", thinking_buffer):
                                lines_think = thinking_buffer.split('\n')
                                for line in lines_think:
                                    s = line.strip()
                                    if re.match(r'^\d+\.\s+', s):
                                        continue
                                    if s == '' and thinking_buffer.count('\n') < 3:
                                        continue
                                    if s and not re.match(r'^(?:Here\'s\s+a\s+)?[Tt]hinking\s+[Pp]rocess:', s) and not re.match(r'^\d+\.\s+', s):
                                        thinking_done = True
                                        chunk_text = line + '\n'
                                        chunk = json.dumps({
                                            "id": cmpl_id, "object": "chat.completion.chunk",
                                            "created": int(time.time()), "model": Path(model_path).name,
                                            "choices": [{"index": 0, "delta": {"role": "assistant", "content": chunk_text}, "finish_reason": None}],
                                        })
                                        if not await safe_write(w, f"data: {chunk}\n\n".encode()): break
                                        break
                                if not thinking_done:
                                    continue

                        chunk = json.dumps({
                            "id": cmpl_id, "object": "chat.completion.chunk",
                            "created": int(time.time()), "model": Path(model_path).name,
                            "choices": [{"index": 0, "delta": {"role": "assistant", "content": chunk_text}, "finish_reason": None}],
                        })
                        if not await safe_write(w, f"data: {chunk}\n\n".encode()): break

                    final = json.dumps({
                        "id": cmpl_id, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": Path(model_path).name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    })
                    await safe_write(w, f"data: {final}\n\ndata: [DONE]\n\n".encode())
                    w.close(); await w.wait_closed()
                    duration = time.time() - req_start
                    _server_stats.record_request(200, tokens=len(tokens_out), duration=duration)
                    logger.info(f"Stream completed: {len(tokens_out)} tokens in {duration:.2f}s ({len(tokens_out)/max(duration,0.01):.1f} tok/s)")
                    return

                # Non-streaming mode
                tokens_out = []
                for tok, _ in generate_step(
                    prompt=prompt_ids, model=model, max_tokens=mt, sampler=_sampler,
                    prompt_cache=cache, quantized_kv_start=quantized_kv_start
                ):
                    tid = int(tok)
                    if tid in eos_ids: break
                    tokens_out.append(tid)

                text = _strip_thinking(tokenizer.decode(tokens_out))
                resp = json.dumps({
                    "id": cmpl_id, "object": "chat.completion",
                    "created": int(time.time()), "model": Path(model_path).name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(tokens_out), "total_tokens": len(prompt_ids) + len(tokens_out)},
                })
                rdata = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(resp)}\r\n\r\n{resp}"
                w.write(rdata.encode()); await w.drain()
                duration = time.time() - req_start
                _server_stats.record_request(200, tokens=len(tokens_out), duration=duration)
                logger.info(f"Request completed: {len(tokens_out)} tokens in {duration:.2f}s ({len(tokens_out)/max(duration,0.01):.1f} tok/s)")

                # Save prompt cache for future requests
                if _prompt_cache["dir"] and not _prompt_cache["loaded"]:
                    try:
                        import hashlib
                        from mlx_lm.models.cache import save_prompt_cache
                        prompt_hash = hashlib.sha256(formatted.encode()).hexdigest()[:16]
                        cache_file = _prompt_cache["dir"] / f"{prompt_hash}.safetensors"
                        if not cache_file.exists():
                            save_prompt_cache(str(cache_file), cache, {"prompt": formatted[:100]})
                            logger.info(f"Saved prompt cache: {cache_file.name}")
                            _server_stats.record_cache_hit()
                    except Exception as e:
                        logger.warning(f"Failed to save prompt cache: {e}")

                return

            w.write(b"HTTP/1.1 404 Not Found\r\n\r\n"); await w.drain()
        except (ConnectionResetError, BrokenPipeError):
            _server_stats.record_connection_reset()
            logger.debug("Client disconnected")
        except Exception as e:
            tb = traceback.format_exc()
            _server_stats.record_error(str(e))
            logger.error(f"Request error: {e}\n{tb}")
            try:
                eb = json.dumps({"error": str(e)})
                w.write(f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {len(eb)}\r\n\r\n{eb}".encode())
                await w.drain()
            except: pass
        finally:
            try: w.close(); await w.wait_closed()
            except: pass

    async def serve():
        loop = asyncio.get_event_loop()
        shutdown_event = asyncio.Event()

        def _signal_handler():
            print("\n  Shutting down gracefully...")
            _server_stats.print_summary()
            _server_stats.save()
            shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try: loop.add_signal_handler(sig, _signal_handler)
            except: pass

        print()
        print("=" * 54)
        print("  TurboQuant MLX Server")
        print("=" * 54)
        print(f"  Model:       {Path(model_path).name}")
        print(f"  Architecture: {arch_label}")
        print(f"  Strategy:    {strategy}")
        print(f"  Endpoint:    http://{host}:{port}")
        print(f"  Chat API:    http://{host}:{port}/v1/chat/completions")
        if is_hybrid:
            print(f"  Compressed:  {len(softmax_indices)}/{n_layers} layers (softmax only)")
        print("=" * 54)
        print()
        srv = await asyncio.start_server(handle, host, port)
        addr = srv.sockets[0].getsockname()
        print(f"  Listening on {addr[0]}:{addr[1]}")
        print("  Press Ctrl+C to stop\n")
        try:
            await shutdown_event.wait()
        except asyncio.CancelledError:
            pass
        srv.close()
        await srv.wait_closed()
        print("  Server stopped.")

    asyncio.run(serve())


# ── main (launcher) ────────────────────────────────────────────────

def main():
    if "--list" in sys.argv:
        models = _scan_models()
        if not models: print("No MLX models found"); return
        print(f"\nFound {len(models)} model(s):\n")
        for m in models: print(f"  {m['label']} ({_fmt_size(m['size'])})")
        return

    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__); return

    if "--serve" in sys.argv:
        cfg = _read_config()
        if not cfg.get("model") or not Path(cfg["model"]).exists():
            print("✗ No valid saved model. Run without --serve first.")
            sys.exit(1)

        # Apply YaRN override if context > native max (for --serve mode too)
        ctx_size = cfg.get("ctx_size", 8192)
        yarn_path, yarn_tmpdir = _apply_yarn_override(
            cfg.get("_original_model_path", cfg["model"]), ctx_size
        )
        if yarn_tmpdir:
            cfg["model"] = yarn_path
            cfg["_yarn_tmpdir"] = yarn_tmpdir

        if not _check_port(cfg["port"]):
            print(f"✗ Port {cfg['port']} is already in use. Kill the other process or change the port in {_CONFIG_FILE}")
            sys.exit(1)
        _start_server(cfg)
        return

    models = _scan_models()
    if not models:
        print("✗ No MLX models found in ~/.lmstudio/models"); sys.exit(1)

    cfg = _read_config()
    _kill_existing()

    print()
    print("=" * 54)
    print("  TurboQuant MLX Server Launcher")
    print("=" * 54)

    model = _pick_model(models, cfg.get("model"))
    cfg["model"] = model["path"]
    cfg["_original_model_path"] = model["path"]
    cfg["model_label"] = model["label"]
    cfg["host"] = input(f"  Server host [{cfg['host']}]: ").strip() or cfg["host"]

    # Detect architecture before asking for strategy
    sys.path.insert(0, str(_TQ_DIR))
    sys.path.insert(0, str(_TQ_DIR / "turboquant"))
    import mlx_lm
    _tmp_model, _ = mlx_lm.load(model["path"])
    is_hybrid, _, _ = _detect_architecture(_tmp_model)
    del _tmp_model

    if is_hybrid:
        print("  Detected: Hybrid GDN+SDPA (Qwen 3.5/3.6)")
        print("  TurboQuant will compress only the softmax layers")
    else:
        print("  Detected: Pure SDPA model")

    # Port selection with conflict detection
    while True:
        cfg["port"] = _pick_int("Server port", cfg["port"])
        if _check_port(cfg["port"]):
            break
        print(f"  ⚠ Port {cfg['port']} is already in use by another process.")
        if input(f"  Try a different port? [Y/n]: ").strip().lower() == "n":
            print("✗ Aborted"); sys.exit(1)

    cfg["tq_strategy"] = _pick_strategy(cfg.get("tq_strategy"), is_hybrid=is_hybrid)
    cfg["ctx_size"] = _pick_context_size(cfg, model["path"])

    # KV quantization for non-TurboQuant strategies (works with ALL models)
    if cfg["tq_strategy"] == "none":
        _pick_kv_bits(cfg, is_turboquant=False)
        _pick_quantized_kv_start(cfg)
        _pick_prompt_cache(cfg)
    else:
        # TurboQuant strategies: symmetric K/V bits required
        _pick_kv_bits(cfg, is_turboquant=True)
        _pick_quantized_kv_start(cfg)

    # Apply YaRN override if context > native max
    yarn_path, yarn_tmpdir = _apply_yarn_override(model["path"], cfg["ctx_size"])
    if yarn_tmpdir:
        cfg["model"] = yarn_path
        cfg["_yarn_tmpdir"] = yarn_tmpdir
    else:
        cfg["model"] = model["path"]

    cfg["max_tokens"] = _pick_int("Max tokens per request", cfg["max_tokens"])
    cfg["temperature"] = _pick_float("Default temperature", cfg["temperature"])

    print()
    print("=" * 54)
    print("  Launch configuration")
    print("=" * 54)
    print(f"  Model:       {cfg['model_label']}")
    print(f"  Architecture: {'Hybrid GDN+SDPA' if is_hybrid else 'Pure SDPA'}")
    print(f"  Host:Port:   {cfg['host']}:{cfg['port']}")
    print(f"  Strategy:    {cfg['tq_strategy']}")
    ctx = cfg['ctx_size']
    ctx_label = " (YaRN extended)" if ctx > _NATIVE_MAX_CONTEXT else ""
    print(f"  Context:     {ctx}{ctx_label}")
    print(f"  KV Quant:    K={cfg['k_bits']}-bit / V={cfg['v_bits']}-bit (asymmetric)")
    print(f"  Quant Start: {cfg.get('quantized_kv_start', 512)} tokens")
    if cfg.get("prompt_cache_dir"):
        print(f"  Prompt Cache: {cfg['prompt_cache_dir']} (persistent)")
    print(f"  Max Tokens:  {cfg['max_tokens']}")
    print(f"  Temperature: {cfg['temperature']}")
    print("=" * 54)

    if _CONFIG_FILE.exists():
        if input("\n  Save as defaults? [Y/n]: ").strip().lower() != "n":
            _save_config(cfg)
    else:
        _save_config(cfg)

    if not Path(cfg["model"]).exists():
        print(f"✗ Model not found: {cfg['model']}"); sys.exit(1)

    print()
    print(f"  Starting TurboQuant server on {cfg['host']}:{cfg['port']} ...")
    print(f"  Model: {cfg['model_label']}, Strategy: {cfg['tq_strategy']}")
    print()

    cfg.pop("model_label", None)
    _start_server(cfg)


if __name__ == "__main__":
    main()
