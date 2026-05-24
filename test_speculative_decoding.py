#!/usr/bin/env python3
"""
Tests für Speculative Decoding.
"""

import sys
import os
import unittest
import random
from pathlib import Path

# Füge das Projekt-Verzeichnis zum Pfad hinzu
sys.path.insert(0, str(Path(__file__).parent))

from speculative_decoding import SpeculativeDecoder, SpeculativePrefillPipeline, ArgMaxSampler


class MockModel:
    """Mock model für Tests."""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.call_count = 0
        
    def __call__(self, x):
        self.call_count += 1
        # Generiere zufällige logits als List
        return [random.gauss(0, 1) for _ in range(self.vocab_size)]


class TestSpeculativeDecoder(unittest.TestCase):
    """Tests für SpeculativeDecoder."""
    
    def setUp(self):
        """Setup für Tests."""
        self.draft_model = MockModel(vocab_size=1000)
        self.target_model = MockModel(vocab_size=1000)
        self.decoder = SpeculativeDecoder(
            draft_model=self.draft_model,
            target_model=self.target_model,
            gamma=4,
        )
        self.sampler = ArgMaxSampler()
        
    def test_init(self):
        """Test Initialisierung."""
        self.assertEqual(self.decoder.gamma, 4)
        self.assertEqual(self.decoder.acceptance_threshold, 0.0)
        self.assertEqual(self.decoder.total_tokens, 0)
        
    def test_reset_stats(self):
        """Test Stats zurücksetzen."""
        self.decoder.total_tokens = 100
        self.decoder.accepted_tokens = 50
        self.decoder.draft_steps = 10
        self.decoder.target_steps = 5
        self.decoder.reset_stats()
        self.assertEqual(self.decoder.total_tokens, 0)
        self.assertEqual(self.decoder.accepted_tokens, 0)
        self.assertEqual(self.decoder.draft_steps, 0)  # Wird zurückgesetzt
        self.assertEqual(self.decoder.target_steps, 0)  # Wird zurückgesetzt
        
    def test_get_stats(self):
        """Test Stats abrufen."""
        self.decoder.total_tokens = 100
        self.decoder.accepted_tokens = 75
        self.decoder.draft_steps = 10
        self.decoder.target_steps = 10
        
        stats = self.decoder.get_stats()
        self.assertAlmostEqual(stats["acceptance_rate"], 0.75)
        self.assertAlmostEqual(stats["speedup"], 10.0)
        self.assertEqual(stats["total_tokens"], 100)
        self.assertEqual(stats["accepted_tokens"], 75)
        
    def test_generate_draft_tokens(self):
        """Test Draft-Token Generierung."""
        prompt = [1, 2, 3, 4, 5]
        cache = []
        max_tokens = 100
        
        tokens, logits = self.decoder.generate_draft_tokens(
            prompt, cache, max_tokens, self.sampler
        )
        
        self.assertEqual(len(tokens), 4)  # gamma=4
        self.assertEqual(len(logits), 4)
        self.assertGreater(self.decoder.draft_steps, 0)
        
    def test_generate_draft_tokens_respects_max(self):
        """Test dass max_tokens respektiert wird."""
        prompt = [1, 2, 3, 4, 5]
        cache = []
        max_tokens = 7  # Nur 2 Tokens möglich
        
        tokens, logits = self.decoder.generate_draft_tokens(
            prompt, cache, max_tokens, self.sampler
        )
        
        self.assertLessEqual(len(tokens), 2)
        
    def test_verify_tokens_empty(self):
        """Test Token-Verifizierung mit leerer Liste."""
        tokens, count = self.decoder.verify_tokens(
            [1, 2, 3], [], [], self.sampler
        )
        self.assertEqual(tokens, [])
        self.assertEqual(count, 0)
        
    def test_verify_tokens_accepts_some(self):
        """Test Token-Verifizierung akzeptiert einige Tokens."""
        draft_tokens = [10, 20, 30, 40]
        prompt = [1, 2, 3]
        
        tokens, count = self.decoder.verify_tokens(
            prompt, draft_tokens, [], self.sampler
        )
        
        self.assertGreaterEqual(count, 0)
        self.assertGreaterEqual(len(tokens), 0)
        self.assertGreater(self.decoder.target_steps, 0)
        
    def test_generate_step_yields_tokens(self):
        """Test dass generate_step Tokens yieldet."""
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
        """Setup für Tests."""
        self.draft_model = MockModel(vocab_size=1000)
        self.target_model = MockModel(vocab_size=1000)
        self.pipeline = SpeculativePrefillPipeline(
            draft_model=self.draft_model,
            target_model=self.target_model,
            gamma=4,
            chunk_size=512,
        )
        self.sampler = ArgMaxSampler()
        
    def test_init(self):
        """Test Initialisierung."""
        self.assertEqual(self.pipeline.chunk_size, 512)
        self.assertEqual(self.pipeline.decoder.gamma, 4)
        
    def test_generate_yields_tokens(self):
        """Test dass generate Tokens yieldet."""
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
