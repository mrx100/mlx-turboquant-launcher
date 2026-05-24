#!/usr/bin/env python3
"""
Tests für Speculative Decoding.
"""

import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from speculative_decoding import SpeculativeDecoder, SpeculativePrefillPipeline


class MockSampler:
    """Mock sampler für Tests."""
    
    def __init__(self, fixed_token=None):
        self.fixed_token = fixed_token
        self.call_count = 0
        
    def __call__(self, logits):
        self.call_count += 1
        if self.fixed_token is not None:
            return self.fixed_token
        return 0


class MockModel:
    """Mock model für Tests."""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.call_count = 0
        
    def __call__(self, x, cache=None):
        self.call_count += 1
        import random
        seq_len = x.shape[1] if hasattr(x, 'shape') else len(x)
        # Return (1, seq_len, vocab_size)
        return [[[random.gauss(0, 1) for _ in range(self.vocab_size)] for _ in range(seq_len)] for _ in range(1)]


class TestSpeculativeDecoder(unittest.TestCase):
    """Tests für SpeculativeDecoder."""
    
    def setUp(self):
        self.draft_model = MockModel(vocab_size=1000)
        self.target_model = MockModel(vocab_size=1000)
        self.decoder = SpeculativeDecoder(
            draft_model=self.draft_model,
            target_model=self.target_model,
            gamma=4,
        )
        self.sampler = MockSampler(fixed_token=42)
        
    def test_init(self):
        self.assertEqual(self.decoder.gamma, 4)
        self.assertEqual(self.decoder.total_tokens, 0)
        
    def test_reset_stats(self):
        self.decoder.total_tokens = 100
        self.decoder.accepted_tokens = 50
        self.decoder.draft_steps = 10
        self.decoder.target_steps = 5
        self.decoder.reset_stats()
        self.assertEqual(self.decoder.total_tokens, 0)
        self.assertEqual(self.decoder.accepted_tokens, 0)
        self.assertEqual(self.decoder.draft_steps, 0)
        self.assertEqual(self.decoder.target_steps, 0)
        
    def test_get_stats(self):
        self.decoder.total_tokens = 100
        self.decoder.accepted_tokens = 75
        self.decoder.draft_steps = 10
        self.decoder.target_steps = 10
        
        stats = self.decoder.get_stats()
        self.assertAlmostEqual(stats["acceptance_rate"], 0.75)
        self.assertAlmostEqual(stats["speedup"], 10.0)
        self.assertEqual(stats["total_tokens"], 100)
        
    def test_generate_step_yields_tokens(self):
        prompt = [1, 2, 3, 4, 5]
        cache = []
        max_tokens = 20
        
        tokens = []
        for token_id, log_prob in self.decoder.generate_step(
            prompt, cache, max_tokens, self.sampler
        ):
            tokens.append(token_id)
            if len(tokens) >= 10:
                break
        
        self.assertGreater(len(tokens), 0)


class TestSpeculativePrefillPipeline(unittest.TestCase):
    """Tests für SpeculativePrefillPipeline."""
    
    def setUp(self):
        self.draft_model = MockModel(vocab_size=1000)
        self.target_model = MockModel(vocab_size=1000)
        self.pipeline = SpeculativePrefillPipeline(
            draft_model=self.draft_model,
            target_model=self.target_model,
            gamma=4,
            chunk_size=512,
        )
        self.sampler = MockSampler(fixed_token=42)
        
    def test_init(self):
        self.assertEqual(self.pipeline.chunk_size, 512)
        self.assertEqual(self.pipeline.decoder.gamma, 4)
        
    def test_generate_yields_tokens(self):
        prompt = [1, 2, 3, 4, 5]
        cache = []
        max_tokens = 20
        
        tokens = []
        for token_id, log_prob in self.pipeline.generate(
            prompt, cache, max_tokens, self.sampler
        ):
            tokens.append(token_id)
            if len(tokens) >= 5:
                break
        
        self.assertGreater(len(tokens), 0)


if __name__ == "__main__":
    unittest.main()
