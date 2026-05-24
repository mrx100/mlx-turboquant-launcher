#!/usr/bin/env python3
"""
Real-World Benchmark: Space Invaders Prompt

Send a complex prompt to the MLX server and measure:
- Time to first token (TTFT)
- Tokens per second
- Total tokens generated
- Speculative decoding stats
"""

import json
import time
import urllib.request
import sys

SERVER_URL = "http://localhost:8084"
PROMPT = """create a full playable,modern spaceinvaders game in one html file with great sound and visual effects, scoring and  leverls getting harder each time with a boss coming with extended life and hard to kill. offer great selection of weopens. run it with a 512k context and calculate or show statistics about server capacities as like token/s."""


def send_request(prompt, stream=True, max_tokens=2048):
    """Send request to server and measure performance."""
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": stream,
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    start_time = time.time()
    tokens = []
    first_token_time = None
    
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            if stream:
                for line in response:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                if first_token_time is None:
                                    first_token_time = time.time() - start_time
                                tokens.append(content)
                        except json.JSONDecodeError:
                            continue
            else:
                result = json.loads(response.read().decode())
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                tokens = list(content)
                first_token_time = time.time() - start_time
                
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    end_time = time.time()
    total_time = end_time - start_time
    total_tokens = len(tokens)
    
    return {
        "ttft_ms": first_token_time * 1000 if first_token_time else 0,
        "total_time_s": total_time,
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / max(total_time, 0.001),
        "generation_time_s": total_time - (first_token_time or 0),
        "gen_tokens_per_second": total_tokens / max(total_time - (first_token_time or 0), 0.001),
    }


def get_server_stats():
    """Get server statistics."""
    try:
        req = urllib.request.Request(f"{SERVER_URL}/stats")
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode())
    except:
        return None


def get_health():
    """Get server health info."""
    try:
        req = urllib.request.Request(f"{SERVER_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode())
    except:
        return None


def main():
    print("=" * 60)
    print("  REAL-WORLD BENCHMARK: Space Invaders Prompt")
    print("=" * 60)
    print()
    
    # Check server
    health = get_health()
    if not health:
        print("✗ Server not running. Start with:")
        print("  cd ~/workspace/mlx-turboquant-launcher && python3 mlx-turboquant.py --serve")
        sys.exit(1)
        
    print(f"Model: {health.get('model', 'unknown')}")
    print(f"Strategy: {health.get('strategy', 'unknown')}")
    print(f"Architecture: {health.get('architecture', 'unknown')}")
    print(f"KV Quant: K={health.get('k_bits', '?')}-bit / V={health.get('v_bits', '?')}-bit")
    print(f"Async Cache: {health.get('use_async_cache', '?')}")
    print()
    
    # Run benchmark
    print(f"Prompt length: {len(PROMPT)} chars")
    print(f"Max tokens: 2048")
    print()
    print("Sending request...")
    print()
    
    result = send_request(PROMPT, stream=True, max_tokens=2048)
    
    if result:
        print("=" * 60)
        print("  RESULTS")
        print("=" * 60)
        print(f"  Time to First Token (TTFT):  {result['ttft_ms']:.1f} ms")
        print(f"  Total Time:                  {result['total_time_s']:.2f} s")
        print(f"  Generation Time:             {result['generation_time_s']:.2f} s")
        print(f"  Total Tokens:                {result['total_tokens']}")
        print(f"  Tokens/s (overall):          {result['tokens_per_second']:.1f}")
        print(f"  Tokens/s (generation only):  {result['gen_tokens_per_second']:.1f}")
        print("=" * 60)
        
        # Server stats
        stats = get_server_stats()
        if stats:
            print()
            print("  SERVER STATISTICS")
            print("=" * 60)
            print(f"  Uptime:           {stats.get('uptime', 'N/A')}")
            print(f"  Total Requests:   {stats.get('total_requests', 0)}")
            print(f"  Tokens Generated: {stats.get('total_tokens_generated', 0)}")
            print(f"  Avg Tokens/s:     {stats.get('avg_tokens_per_second', 0):.1f}")
            print(f"  Cache Hit Rate:   {stats.get('cache_hit_rate', 'N/A')}")
            print(f"  Errors:           {stats.get('errors', 0)}")
            print("=" * 60)
    else:
        print("✗ Request failed")


if __name__ == "__main__":
    main()
