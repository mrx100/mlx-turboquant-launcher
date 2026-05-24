"""
Cache Eviction Policies für MLX Model Server.

Intelligente Cache-Verwaltung für lange Kontexte.
Implementierte Policies:
- LRU (Least Recently Used)
- LFU (Least Frequently Used)
- Attention-Score based (SnapKV, H2O)
- Sliding Window + Important Tokens

Erwarteter Speedup: 10-20% bei langen Kontexten.
"""

import math
import heapq
from typing import List, Dict, Optional, Tuple, Any
from collections import OrderedDict


class CacheEntry:
    """Eintrag im KV-Cache."""
    
    def __init__(
        self,
        key: Any,
        value: Any,
        position: int,
        attention_score: float = 0.0,
        access_count: int = 0,
        last_access: int = 0,
    ):
        self.key = key
        self.value = value
        self.position = position
        self.attention_score = attention_score
        self.access_count = access_count
        self.last_access = last_access
        
    def __lt__(self, other):
        return self.attention_score < other.attention_score


class LRUEvictionPolicy:
    """
    Least Recently Used Eviction Policy.
    """
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = OrderedDict()
        
    def get(self, key: int) -> Optional[CacheEntry]:
        """Hole Entry aus Cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
        
    def put(self, key: int, entry: CacheEntry):
        """Füge Entry in Cache ein."""
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = entry
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = entry
            
    def evict(self, n: int = 1) -> List[int]:
        """Entferne n Entries."""
        evicted = []
        for _ in range(min(n, len(self.cache))):
            if self.cache:
                key, _ = self.cache.popitem(last=False)
                evicted.append(key)
        return evicted
        
    def size(self) -> int:
        return len(self.cache)


class LFUEvictionPolicy:
    """
    Least Frequently Used Eviction Policy.
    """
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = {}
        self.frequency = {}
        
    def get(self, key: int) -> Optional[CacheEntry]:
        """Hole Entry aus Cache."""
        if key in self.cache:
            self.frequency[key] = self.frequency.get(key, 0) + 1
            return self.cache[key]
        return None
        
    def put(self, key: int, entry: CacheEntry):
        """Füge Entry in Cache ein."""
        if key in self.cache:
            self.cache[key] = entry
            self.frequency[key] = self.frequency.get(key, 0) + 1
        else:
            if len(self.cache) >= self.max_size:
                self._evict_one()
            self.cache[key] = entry
            self.frequency[key] = 1
            
    def _evict_one(self):
        """Entferne am wenigsten verwendeten Entry."""
        if not self.cache:
            return
            
        min_key = min(self.cache.keys(), key=lambda k: self.frequency.get(k, 0))
        del self.cache[min_key]
        del self.frequency[min_key]
        
    def evict(self, n: int = 1) -> List[int]:
        """Entferne n Entries."""
        evicted = []
        for _ in range(min(n, len(self.cache))):
            if self.cache:
                min_key = min(self.cache.keys(), key=lambda k: self.frequency.get(k, 0))
                del self.cache[min_key]
                del self.frequency[min_key]
                evicted.append(min_key)
        return evicted
        
    def size(self) -> int:
        return len(self.cache)


class AttentionScoreEvictionPolicy:
    """
    Attention-Score based Eviction Policy (SnapKV/H2O).
    """
    
    def __init__(self, max_size: int, window_size: int = 256):
        self.max_size = max_size
        self.window_size = window_size
        self.cache = {}
        self.scores = {}
        
    def get(self, key: int) -> Optional[CacheEntry]:
        """Hole Entry aus Cache."""
        return self.cache.get(key)
        
    def put(self, key: int, entry: CacheEntry):
        """Füge Entry in Cache ein."""
        if key in self.cache:
            self.cache[key] = entry
            self.scores[key] = entry.attention_score
        else:
            if len(self.cache) >= self.max_size:
                self._evict_one()
            self.cache[key] = entry
            self.scores[key] = entry.attention_score
            
    def _evict_one(self):
        """Entferne Entry mit niedrigstem Attention-Score."""
        if not self.cache:
            return
            
        # Behalte window_size wichtigste Tokens
        if len(self.cache) <= self.window_size:
            return
            
        # Finde Entry mit niedrigstem Score außerhalb des Fensters
        min_key = None
        min_score = float('inf')
        
        for key, entry in self.cache.items():
            if entry.position < len(self.cache) - self.window_size:
                if entry.attention_score < min_score:
                    min_score = entry.attention_score
                    min_key = key
                    
        if min_key is not None:
            del self.cache[min_key]
            del self.scores[min_key]
            
    def evict(self, n: int = 1) -> List[int]:
        """Entferne n Entries."""
        evicted = []
        for _ in range(min(n, len(self.cache))):
            if len(self.cache) <= self.window_size:
                break
                
            min_key = None
            min_score = float('inf')
            
            for key, entry in self.cache.items():
                if entry.position < len(self.cache) - self.window_size:
                    if entry.attention_score < min_score:
                        min_score = entry.attention_score
                        min_key = key
                        
            if min_key is not None:
                del self.cache[min_key]
                del self.scores[min_key]
                evicted.append(min_key)
                
        return evicted
        
    def size(self) -> int:
        return len(self.cache)
    
    def update_scores(self, attention_weights: Dict[int, float]):
        """
        Aktualisiere Attention-Scores.
        
        Args:
            attention_weights: Dict mit Position → Score
        """
        for key, entry in self.cache.items():
            if entry.position in attention_weights:
                entry.attention_score = attention_weights[entry.position]
                self.scores[key] = entry.attention_score


class SlidingWindowEvictionPolicy:
    """
    Sliding Window + Important Tokens Eviction Policy.
    """
    
    def __init__(
        self,
        max_size: int,
        window_size: int = 512,
        important_ratio: float = 0.1,
    ):
        self.max_size = max_size
        self.window_size = window_size
        self.important_count = int(max_size * important_ratio)
        self.cache = {}
        self.important_keys = set()
        
    def get(self, key: int) -> Optional[CacheEntry]:
        """Hole Entry aus Cache."""
        return self.cache.get(key)
        
    def put(self, key: int, entry: CacheEntry):
        """Füge Entry in Cache ein."""
        if key in self.cache:
            self.cache[key] = entry
        else:
            if len(self.cache) >= self.max_size:
                self._evict_one()
            self.cache[key] = entry
            
    def _evict_one(self):
        """Entferne Entry außerhalb des Fensters und nicht important."""
        if not self.cache:
            return
            
        # Behalte wichtige Tokens
        if len(self.important_keys) >= self.important_count:
            return
            
        # Finde Entry zum Entfernen
        for key, entry in list(self.cache.items()):
            if key not in self.important_keys:
                if entry.position < len(self.cache) - self.window_size:
                    del self.cache[key]
                    return
                    
    def evict(self, n: int = 1) -> List[int]:
        """Entferne n Entries."""
        evicted = []
        for _ in range(min(n, len(self.cache))):
            if len(self.cache) <= self.window_size + len(self.important_keys):
                break
                
            for key, entry in list(self.cache.items()):
                if key not in self.important_keys:
                    if entry.position < len(self.cache) - self.window_size:
                        del self.cache[key]
                        evicted.append(key)
                        break
                        
        return evicted
        
    def mark_important(self, keys: List[int]):
        """Markiere Tokens als wichtig."""
        self.important_keys.update(keys)
        
    def size(self) -> int:
        return len(self.cache)


class CacheEvictionManager:
    """
    Verwaltet Cache Eviction über verschiedene Policies hinweg.
    """
    
    def __init__(
        self,
        max_size: int,
        policy: str = "attention",
        window_size: int = 256,
    ):
        self.max_size = max_size
        self.policy_name = policy
        
        if policy == "lru":
            self.policy = LRUEvictionPolicy(max_size)
        elif policy == "lfu":
            self.policy = LFUEvictionPolicy(max_size)
        elif policy == "attention":
            self.policy = AttentionScoreEvictionPolicy(max_size, window_size)
        elif policy == "sliding_window":
            self.policy = SlidingWindowEvictionPolicy(max_size, window_size)
        else:
            self.policy = AttentionScoreEvictionPolicy(max_size, window_size)
            
    def get(self, key: int) -> Optional[CacheEntry]:
        """Hole Entry aus Cache."""
        return self.policy.get(key)
        
    def put(self, key: int, entry: CacheEntry):
        """Füge Entry in Cache ein."""
        self.policy.put(key, entry)
        
    def evict(self, n: int = 1) -> List[int]:
        """Entferne n Entries."""
        return self.policy.evict(n)
        
    def size(self) -> int:
        return self.policy.size()
        
    def get_stats(self) -> dict:
        """Hole Cache-Statistiken."""
        return {
            "policy": self.policy_name,
            "size": self.size(),
            "max_size": self.max_size,
            "utilization": self.size() / max(self.max_size, 1),
        }
