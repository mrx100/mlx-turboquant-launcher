"""
Speculative Decoding für MLX Model Server.

Draft-Modell generiert γ Tokens parallel, Target-Modell verifiziert.
Erwarteter Speedup: 30-50% bei acceptance rate ~60-80%.
"""

import mlx.core as mx
import random
from typing import List, Optional, Tuple, Any


class MockSampler:
    """Sampler interface für Tests."""
    
    def __call__(self, logits: List[float]) -> int:
        raise NotImplementedError


class ArgMaxSampler(MockSampler):
    """Sampler der das Token mit dem höchsten Logit wählt."""
    
    def __call__(self, logits: List[float]) -> int:
        return logits.index(max(logits))


class RandomSampler(MockSampler):
    """Sampler der zufällig ein Token wählt."""
    
    def __call__(self, logits: List[float]) -> int:
        return random.randint(0, len(logits) - 1)


class SpeculativeDecoder:
    """
    Speculative Decoding Implementation.
    
    Args:
        draft_model: Kleineres Modell für Token-Generierung
        target_model: Größeres Modell für Verifikation
        gamma: Anzahl der Draft-Tokens pro Schritt (default: 4)
        acceptance_threshold: Minimum acceptance probability (default: 0.0)
    """
    
    def __init__(
        self,
        draft_model: Any,
        target_model: Any,
        gamma: int = 4,
        acceptance_threshold: float = 0.0,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.gamma = gamma
        self.acceptance_threshold = acceptance_threshold
        
        # Statistics
        self.total_tokens = 0
        self.accepted_tokens = 0
        self.draft_steps = 0
        self.target_steps = 0
        
    def reset_stats(self):
        """Setze alle Statistiken zurück."""
        self.total_tokens = 0
        self.accepted_tokens = 0
        self.draft_steps = 0
        self.target_steps = 0
        
    def get_stats(self) -> dict:
        """Hole aktuelle Statistiken."""
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
    
    def generate_draft_tokens(
        self,
        prompt_ids: List[int],
        cache: List,
        max_tokens: int,
        sampler,
    ) -> Tuple[List[int], List]:
        """
        Generiere γ Draft-Tokens mit dem Draft-Modell.
        
        Returns:
            draft_tokens: Liste der generierten Token-IDs
            draft_logits: Logits für jedes Token (für Verifikation)
        """
        draft_tokens = []
        draft_logits = []
        current_ids = list(prompt_ids)
        
        for _ in range(self.gamma):
            if len(current_ids) >= max_tokens:
                break
                
            # Draft model forward pass
            logits = self.draft_model(mx.array([current_ids]))
            # logits shape: (1, seq_len, vocab_size)
            current_logits = logits[0, -1, :]
                
            token = sampler(current_logits)
            token_id = int(token)
            
            draft_tokens.append(token_id)
            draft_logits.append(current_logits)
            current_ids.append(token_id)
            
        self.draft_steps += 1
        return draft_tokens, draft_logits
    
    def verify_tokens(
        self,
        prompt_ids: List[int],
        draft_tokens: List[int],
        cache: List,
        sampler,
    ) -> Tuple[List[int], int]:
        """
        Verifiziere Draft-Tokens mit Target-Modell.
        
        Returns:
            accepted_tokens: Liste der akzeptierten Tokens
            accepted_count: Anzahl der akzeptierten Tokens
        """
        if not draft_tokens:
            return [], 0
            
        # Target model forward pass für alle Draft-Tokens parallel
        input_ids = prompt_ids + draft_tokens[:-1]
        logits = self.target_model(mx.array([input_ids]))
        
        # logits shape: (1, seq_len, vocab_size)
        # Wir brauchen die logits für die Positionen nach dem Prompt
        
        # Verifiziere jedes Draft-Token
        accepted_tokens = []
        accepted_count = 0
        
        for i, draft_token in enumerate(draft_tokens):
            # Hole die Wahrscheinlichkeit für das Draft-Token
            # logits shape: (1, seq_len, vocab_size)
            # Wir wollen die Position nach dem Prompt + i
            logit_index = len(prompt_ids) + i - 1
            if logit_index < 0:
                logit_index = 0
            if logit_index >= logits.shape[1]:
                logit_index = logits.shape[1] - 1
            logit_slice = logits[0, logit_index, :]
            
            # Convert to list for compatibility
            logit_list = logit_slice.tolist()
            
            # Berechne softmax probability
            max_logit = max(logit_list)
            exp_values = [max(0, l - max_logit) for l in logit_list]
            sum_exp = sum(exp_values)
            if sum_exp == 0:
                acceptance_prob = 0.0
            else:
                acceptance_prob = exp_values[draft_token] / sum_exp
            
            # Akzeptiere mit Wahrscheinlichkeit acceptance_prob
            if acceptance_prob >= self.acceptance_threshold:
                accepted_tokens.append(draft_token)
                accepted_count += 1
                self.accepted_tokens += 1
            else:
                # Sample neues Token vom Target-Modell
                new_token = sampler(logit_slice)
                accepted_tokens.append(int(new_token))
                break
                
        self.target_steps += 1
        self.total_tokens += len(draft_tokens)
        
        return accepted_tokens, accepted_count
    
    def generate_step(
        self,
        prompt_ids: List[int],
        cache: List,
        max_tokens: int,
        sampler,
    ):
        """
        Generiere Tokens mit speculative decoding.
        
        Yield:
            token_id: Generiertes Token
            log_prob: Log-Wahrscheinlichkeit
        """
        current_ids = list(prompt_ids)
        
        while len(current_ids) < max_tokens:
            # Phase 1: Draft-Modell generiert γ Tokens
            draft_tokens, draft_logits = self.generate_draft_tokens(
                current_ids, cache, max_tokens, sampler
            )
            
            if not draft_tokens:
                break
                
            # Phase 2: Target-Modell verifiziert
            accepted_tokens, accepted_count = self.verify_tokens(
                current_ids, draft_tokens, cache, sampler
            )
            
            # Phase 3: Akzeptierte Tokens yielden
            for token_id in accepted_tokens:
                current_ids.append(token_id)
                yield token_id, 0.0  # log_prob wird später berechnet
                
            # Wenn keine Tokens akzeptiert wurden, generiere eins vom Target
            if accepted_count == 0:
                # Generiere ein Token vom Target-Modell
                logits = self.target_model(mx.array([current_ids]))
                token = sampler(logits[0, -1, :])
                token_id = int(token)
                current_ids.append(token_id)
                yield token_id, 0.0


class SpeculativePrefillPipeline:
    """
    Kombiniert speculative decoding mit async prefill.
    """
    
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
        """Generiere Tokens mit speculative decoding pipeline."""
        for token_id, log_prob in self.decoder.generate_step(
            prompt_ids, cache, max_tokens, sampler
        ):
            yield token_id, log_prob
