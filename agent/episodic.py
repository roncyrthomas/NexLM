"""Tier E — Episodic memory.

Embedding-keyed cache of (query, response, reward) triples with
nearest-neighbor recall. Provides the agent with fast intuition:
"have I seen something like this before, and what worked then?"

Implementation choice: hashing-based embedding + scipy KDTree for KNN.
For production swap to sentence-transformers + FAISS; we keep it
dependency-light here so unit tests don't need a download.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


@dataclass
class Episode:
    query: str
    response: str
    reward: float
    timestamp: float
    embedding: np.ndarray
    hits: int = 0  # number of times this episode was recalled


def _hash_embed(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic hashing-based embedding.

    Cheap stand-in for sentence-transformers — gives stable vectors for
    similar inputs without any model load. Production: replace with real
    sentence embeddings.
    """
    vec = np.zeros(dim, dtype=np.float32)
    # bag of trigrams hashed into the vector
    tokens = text.lower().split()
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf8")).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
        for i in range(len(tok) - 2):
            tri = tok[i : i + 3]
            h = int(hashlib.md5(tri.encode("utf8")).hexdigest()[:8], 16)
            vec[h % dim] += 0.5
    norm = float(np.linalg.norm(vec))
    return vec / (norm + 1e-8)


@dataclass
class EpisodicMemory:
    max_size: int = 10_000
    embed_dim: int = 256
    similarity_threshold: float = 0.85
    embed_fn = None  # plug in a real embed function; defaults to hashing

    episodes: list[Episode] = field(default_factory=list)
    _tree: Optional["cKDTree"] = field(default=None, repr=False)
    _dirty: bool = True

    def _embed(self, text: str) -> np.ndarray:
        if self.embed_fn is not None:
            return self.embed_fn(text)
        return _hash_embed(text, dim=self.embed_dim)

    def _rebuild_tree(self) -> None:
        if not HAVE_SCIPY or not self.episodes:
            self._tree = None
        else:
            data = np.stack([e.embedding for e in self.episodes])
            self._tree = cKDTree(data)
        self._dirty = False

    def remember(self, query: str, response: str, reward: float = 0.0) -> None:
        """Add a new episode. Auto-prunes if buffer is full."""
        ep = Episode(
            query=query,
            response=response,
            reward=reward,
            timestamp=time.time(),
            embedding=self._embed(query),
        )
        self.episodes.append(ep)
        self._dirty = True
        if len(self.episodes) > self.max_size:
            self.prune(self.max_size)

    def recall(self, query: str, k: int = 5) -> list[Episode]:
        """Return the k most similar past episodes."""
        if not self.episodes:
            return []
        if self._dirty:
            self._rebuild_tree()
        q_emb = self._embed(query)
        if self._tree is None:
            # Fallback: brute force cosine
            sims = [(float(q_emb @ e.embedding), e) for e in self.episodes]
            sims.sort(key=lambda x: -x[0])
            results = [e for _, e in sims[:k]]
        else:
            # KDTree uses Euclidean; for unit vectors, dist ↔ 2(1 - cos)
            dists, idxs = self._tree.query(q_emb, k=min(k, len(self.episodes)))
            if np.isscalar(dists):
                idxs = [int(idxs)]
            else:
                idxs = [int(i) for i in idxs]
            results = [self.episodes[i] for i in idxs]
        for e in results:
            e.hits += 1
        return results

    def confidence(self, query: str) -> float:
        """Cosine similarity to the closest stored episode (0 if empty)."""
        if not self.episodes:
            return 0.0
        q_emb = self._embed(query)
        sims = [float(q_emb @ e.embedding) for e in self.episodes]
        return max(sims) if sims else 0.0

    def is_familiar(self, query: str) -> bool:
        """True iff we have a sufficiently similar past episode."""
        return self.confidence(query) >= self.similarity_threshold

    def prune(self, target_size: int) -> int:
        """Drop lowest-utility episodes. Utility = recency × success × distinctness."""
        if len(self.episodes) <= target_size:
            return 0
        now = time.time()
        scored = []
        for e in self.episodes:
            recency = math.exp(-(now - e.timestamp) / (7 * 86400))  # 1-week half-life
            success = max(0.0, e.reward)
            scored.append((recency * (1 + success) * (1 + 0.1 * e.hits), e))
        scored.sort(key=lambda x: -x[0])
        before = len(self.episodes)
        self.episodes = [e for _, e in scored[:target_size]]
        self._dirty = True
        return before - len(self.episodes)

    def stats(self) -> dict:
        if not self.episodes:
            return {"size": 0, "avg_reward": 0.0, "total_hits": 0}
        return {
            "size": len(self.episodes),
            "avg_reward": float(np.mean([e.reward for e in self.episodes])),
            "total_hits": int(sum(e.hits for e in self.episodes)),
        }


import math  # noqa: E402 (kept at bottom to avoid reorg)
