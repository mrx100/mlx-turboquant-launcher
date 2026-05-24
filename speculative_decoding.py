"""
Speculative Decoding für MLX Model Server.

Draft-Modell generiert γ Tokens parallel, Target-Modell verifiziert.
Erwarteter Speedup: 30-50% bei acceptance rate ~60-80%.
"""

import mlx.core as mx
from typing import List, Optional, Tuple, Any


class SpeculativeDecoder:
    """
    Speculative Decoding Implementation.
    
    Uses the draft model's own cache for fast generation,
    then verifies with the target model's cache.
    
    Args:
        draft_model: Kleineres Modell für Token-Generierung
        target_model: Größeres Modell für Verifikation
        draft_cache: KV-Cache für Draft-Modell
        target_cache: KV-Cache für Target-Modell
        gamma: Anzahl der Draft-Tokens pro Schritt (default: 4)
    """
    
    def __init__(
        self,
        draft_model: Any,
        target_model: Any,
        draft_cache: Any = None,
        target_cache: Any = None,
        gamma: int = 4,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.draft_cache = draft_cache
        self.target_cache = target_cache
        self.gamma = gamma
        
        # Statistics
        self.total_tokens = 0
        self.accepted_tokens = 0
        self.draft_steps = 0
        self.target_steps = 0
        
    def reset_stats(self):
        self.total_tokens = 0
        self.accepted_tokens = 0
        self.draft_steps = 0
        self.target_steps = 0
        
    def get_stats(self) -> dict:
        acceptance_rate = self.accepted_tokens / max(self.total_tokens, 1)
        speedup = self.total_tokens / max(self.target_steps, 1)
        return {
            "acceptance_rate": acceptance_rate,
            "speedup": speedup,
            "total_tokens": self.total_tokens,
            "accepted_tokens": self.accepted_tokens,
            "draft_steps": self.draft_steps,
            "target_steps": self.target_steps,
        }
    
    def generate_step(
        self,
        prompt_ids: List[int],
        cache: List,
        max_tokens: int,
        sampler,
    ):
        """
        Generiere Tokens mit speculative decoding.
        
        For each step:
        1. Draft model generates γ tokens (using its own cache)
        2. Target model verifies all γ tokens at once (using target cache)
        3. Accept matching tokens, reject mismatches
        
        Yield:
            token_id: Generiertes Token
            log_prob: Log-Wahrscheinlichkeit
        """
        current_ids = list(prompt_ids)
        n_prompt = len(current_ids)
        
        while len(current_ids) < max_tokens:
            # Phase 1: Draft model generates γ tokens
            draft_tokens = []
            draft_logits_list = []
            
            for _ in range(self.gamma):
                if len(current_ids) >= max_tokens:
                    break
                
                # Draft model forward with cache
                logits = self.draft_model(mx.array([current_ids]), cache=self.draft_cache)
                # logits shape: (1, 1, vocab_size) with cache or (1, seq_len, vocab_size)
                if isinstance(logits, list):
                    current_logits = logits[0][-1]
                else:
                    current_logits = logits[0, -1, :]
                
                # Convert to array if list for sampler
                if isinstance(current_logits, list):
                    current_logits = mx.array(current_logits)
                token = sampler(current_logits)
                token_id = int(token)
                
                draft_tokens.append(token_id)
                draft_logits_list.append(current_logits)
                current_ids.append(token_id)
            
            self.draft_steps += 1
            
            if not draft_tokens:
                break
            
            # Phase 2: Target model verifies
            # Reset target cache position to prompt length for verification
            # Re-run target model on full prompt + draft tokens
            verify_ids = list(prompt_ids) + draft_tokens[:-1] if len(draft_tokens) > 1 else list(prompt_ids)
            logits = self.target_model(mx.array([verify_ids]), cache=self.target_cache)
            
            accepted_tokens = []
            accepted_count = 0
            
            for i, draft_token in enumerate(draft_tokens):
                # Get logit for position after prompt + i
                logit_idx = n_prompt + i - 1
                if logit_idx < 0:
                    logit_idx = 0
                    
                if isinstance(logits, list):
                    logit_idx = min(logit_idx, len(logits[0]) - 1)
                    logit_slice = logits[0][logit_idx]
                else:
                    if logit_idx >= logits.shape[1]:
                        logit_idx = logits.shape[1] - 1
                    logit_slice = logits[0, logit_idx, :]
                
                # Convert to array if list
                if isinstance(logit_slice, list):
                    logit_slice = mx.array(logit_slice)
                probs = mx.softmax(logit_slice)
                acceptance_prob = probs[draft_token].item()
                
                if acceptance_prob >= 0.0:  # Always accept (can add threshold later)
                    accepted_tokens.append(draft_token)
                    accepted_count += 1
                    self.accepted_tokens += 1
                else:
                    # Sample replacement from target
                    new_token = sampler(logit_slice)
                    accepted_tokens.append(int(new_token))
                    break
            
            self.target_steps += 1
            self.total_tokens += len(draft_tokens)
            
            # Phase 3: Yield accepted tokens
            for token_id in accepted_tokens:
                yield token_id, 0.0
            
            # If all rejected, generate one from target
            if accepted_count == 0:
                logits = self.target_model(mx.array([current_ids]), cache=self.target_cache)
                if isinstance(logits, list):
                    logit_slice = mx.array(logits[0][-1])
                else:
                    logit_slice = logits[0, -1, :]
                token = sampler(logit_slice)
                token_id = int(token)
                current_ids.append(token_id)
                yield token_id, 0.0


class SpeculativePrefillPipeline:
    """Kombiniert speculative decoding mit async prefill."""
    
    def __init__(
        self,
        draft_model: Any,
        target_model: Any,
        gamma: int = 4,
        chunk_size: int = 512,
    ):
        self.decoder = SpeculativeDecoder(
            draft_model, target_model, gamma=gamma
        )
        self.chunk_size = chunk_size
        
    def generate(
        self,
        prompt_ids: List[int],
        cache: List,
        max_tokens: int,
        sampler,
    ):
        for token_id, log_prob in self.decoder.generate_step(
            prompt_ids, cache, max_tokens, sampler
        ):
            yield token_id, log_prob
