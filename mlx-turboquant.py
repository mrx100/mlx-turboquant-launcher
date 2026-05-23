#!/usr/bin/env python3
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

import os, sys, json, time, uuid, asyncio, re, shutil, atexit, signal
from pathlib import Path

_MODELS_DIR = Path.home() / ".lmstudio/models"
_CONFIG_FILE = Path.home() / ".tq_defaults.json"
_TQ_DIR = Path.home() / "workspace" / "turboquant-mlx"

_DEFAULTS = {
    "model": None, "port": 8081, "host": "0.0.0.0",
    "max_tokens": 2048, "temperature": 0.7,
    "tq_strategy": "v2_4bit_rotated", "ctx_size": 8192,
    "kv_bits": 4, "kv_group_size": 64,
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

def _create_hybrid_cache(model, strategy, head_dim):
    """Create cache list appropriate for the model's architecture.

    Pure SDPA: all layers get TurboQuant caches.
    Hybrid GDN+SDPA: softmax layers get TurboQuant, GDN layers get stubs.
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
                    )
                elif strategy == "tq_4bit_fast":
                    cache = TurboQuantKVCacheV2(
                        head_dim=head_dim, bits=4, group_size=64,
                        use_rotation=False, use_normalization=False,
                    )
                elif strategy == "tq_3bit":
                    cache = TurboQuantKVCacheV2(
                        head_dim=head_dim, bits=3, group_size=64,
                        use_rotation=True, use_normalization=True,
                        use_qjl=True,
                    )
                else:
                    cache = TurboQuantKVCacheV2(
                        head_dim=head_dim, bits=4, group_size=64,
                        use_rotation=True, use_normalization=True,
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
                    head_dim = json.loads(cfg_path.read_text()).get('head_dim') or (json.loads(cfg_path.read_text()).get('text_config') or {}).get('head_dim')

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
                })
                rdata = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(resp)}\r\n\r\n{resp}"
                w.write(rdata.encode()); await w.drain(); return

            if path == "/v1/models":
                resp = json.dumps({"object": "list", "data": [{
                    "id": Path(model_path).name, "object": "model", "owned_by": "turboquant",
                }]})
                rdata = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(resp)}\r\n\r\n{resp}"
                w.write(rdata.encode()); await w.drain(); return

            if path == "/v1/chat/completions" and method == "POST":
                try: payload = json.loads(body)
                except: w.write(b"HTTP/1.1 400 Bad Request\r\n\r\nInvalid JSON"); await w.drain(); return

                msgs = payload.get("messages", [])
                if not msgs: w.write(b"HTTP/1.1 400 Bad Request\r\n\r\nNo messages"); await w.drain(); return

                mt = payload.get("max_tokens") or max_tokens
                tp = payload.get("temperature") or temperature
                stream = payload.get("stream", False)
                cache, _, _, _ = _create_hybrid_cache(model, strategy, head_dim)
                formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                prompt_ids = mx.array(tokenizer.encode(formatted))

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

                    for tok, _ in generate_step(prompt=prompt_ids, model=model, max_tokens=mt, sampler=_sampler, prompt_cache=cache):
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
                    return

                # Non-streaming mode
                tokens_out = []
                for tok, _ in generate_step(prompt=prompt_ids, model=model, max_tokens=mt, sampler=_sampler, prompt_cache=cache):
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
                w.write(rdata.encode()); await w.drain(); return

            w.write(b"HTTP/1.1 404 Not Found\r\n\r\n"); await w.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass  # Client disconnected — normal, don't log
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"ERROR: {e}\n{tb}", flush=True)
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
