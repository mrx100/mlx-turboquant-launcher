#!/usr/bin/env python3
"""
Download Draft-Modell für Speculative Decoding.

Lädt ein kleineres Modell (Qwen2.5-1.5B oder gpt-oss-2B) für speculative decoding.
"""

import os
import subprocess
import sys
from pathlib import Path


def download_draft_model(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """
    Lädt ein Draft-Modell herunter.
    
    Args:
        model_name: HuggingFace Modell-Name
    """
    models_dir = Path(os.path.expanduser("~/.lmstudio/models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading draft model: {model_name}")
    print(f"Target directory: {models_dir}")
    
    # Versuche mlx-lm zu verwenden
    try:
        result = subprocess.run(
            ["mlx-lm", "convert", "--hf-path", model_name, "--mlx-path", str(models_dir / model_name.split("/")[-1])],
            capture_output=True,
            text=True,
            timeout=600,
        )
        
        if result.returncode == 0:
            print(f"✓ Draft model downloaded: {models_dir / model_name.split('/')[-1]}")
            return str(models_dir / model_name.split("/")[-1])
        else:
            print(f"✗ Download failed: {result.stderr}")
            
    except FileNotFoundError:
        print("mlx-lm not found, trying huggingface-cli...")
        
    # Fallback: huggingface-cli
    try:
        result = subprocess.run(
            ["huggingface-cli", "download", model_name, "--local-dir", str(models_dir / model_name.split("/")[-1])],
            capture_output=True,
            text=True,
            timeout=600,
        )
        
        if result.returncode == 0:
            print(f"✓ Draft model downloaded: {models_dir / model_name.split('/')[-1]}")
            return str(models_dir / model_name.split("/")[-1])
        else:
            print(f"✗ Download failed: {result.stderr}")
            
    except FileNotFoundError:
        print("huggingface-cli not found")
        
    print("\nManual download required:")
    print(f"  mlx-lm convert --hf-path {model_name} --mlx-path {models_dir / model_name.split('/')[-1]}")
    return None


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
    download_draft_model(model)
