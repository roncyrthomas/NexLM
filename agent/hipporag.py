"""Tier 1 — HippoRAG: knowledge-graph retrieval via Personalized PageRank.

Pipeline:
  1. Ingest: chunk docs → extract (subj, pred, obj) triples → build NetworkX graph.
  2. Retrieve: seed graph with query entities → run PPR → pull chunks attached
     to top-scoring nodes.

The extraction LLM is plugged in via a callable: any model that takes text and
returns triples works (we'll use Phi-3-mini for production, or stub for tests).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

try:
    import networkx as nx
    HAVE_NETWORKX = True
except ImportError:
    HAVE_NETWORKX = False


# ─── Data types ───────────────────────────────────────────────────────────
@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    chunk_id: str
    confidence: float = 1.0


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


# ─── Extractor: callable that takes text → list of triples ─────────────────
ExtractorFn = Callable[[str], list[dict]]


def regex_stub_extractor(text: str) -> list[dict]:
    """Deterministic, dependency-free stub that finds "X is Y" / "X has Y" patterns.

    Useful for testing and demos without needing an LLM available. Production
    uses a real LLM (Phi-3-mini-based extractor below).
    """
    triples: list[dict] = []
    for m in re.finditer(r"([A-Z][\w ]{2,30})\s+is\s+(?:a |an |the )?([\w \-]{2,40})", text):
        triples.append({"subject": m.group(1).strip(), "predicate": "is_a", "object": m.group(2).strip(), "confidence": 0.7})
    for m in re.finditer(r"([A-Z][\w ]{2,30})\s+has\s+(?:a |an |the )?([\w \-]{2,40})", text):
        triples.append({"subject": m.group(1).strip(), "predicate": "has", "object": m.group(2).strip(), "confidence": 0.7})
    return triples


def llm_extractor_factory(generate_fn: Callable[[str], str]) -> ExtractorFn:
    """Build an extractor that prompts an LLM and parses JSON output.

    `generate_fn` should be `NexAgent.generate` or a similar string-in/string-out callable.
    """
    PROMPT = (
        "Extract factual triples from the following text as a JSON list, each item "
        '{{"subject": ..., "predicate": ..., "object": ..., "confidence": 0-1}}.\n'
        "Only output the JSON list, no preamble.\n\n"
        "Text: {text}\n\nJSON:"
    )

    def _extract(text: str) -> list[dict]:
        out = generate_fn(PROMPT.format(text=text[:2000]))
        m = re.search(r"\[.*\]", out, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
            return [d for d in data if isinstance(d, dict) and "subject" in d]
        except json.JSONDecodeError:
            return []

    return _extract


# ─── Main pipeline ─────────────────────────────────────────────────────────
@dataclass
class HippoRAG:
    confidence_threshold: float = 0.5
    ppr_alpha: float = 0.5    # PPR damping; lower = more weight on seed nodes

    graph: Optional["nx.Graph"] = field(default=None, repr=False)
    chunks: dict[str, Chunk] = field(default_factory=dict)
    node_chunks: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def __post_init__(self):
        if not HAVE_NETWORKX:
            raise ImportError("HippoRAG requires networkx. pip install networkx")
        if self.graph is None:
            self.graph = nx.Graph()

    # ─── Ingest ────────────────────────────────────────────────────────────
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Simple character-based chunking. Replace with token-based for production."""
        chunks: list[str] = []
        i = 0
        while i < len(text):
            chunks.append(text[i : i + chunk_size])
            i += chunk_size - overlap
        return chunks

    def ingest_documents(
        self,
        docs: list[tuple[str, str]],  # (doc_id, text) pairs
        extractor: ExtractorFn,
        chunk_size: int = 500,
        verbose: bool = False,
    ) -> None:
        """Add documents to the graph + chunk store."""
        for doc_id, text in docs:
            for chunk_idx, chunk_text in enumerate(self.chunk_text(text, chunk_size=chunk_size)):
                chunk_id = f"{doc_id}::c{chunk_idx}"
                self.chunks[chunk_id] = Chunk(chunk_id=chunk_id, doc_id=doc_id, text=chunk_text)

                for t in extractor(chunk_text):
                    if t.get("confidence", 1.0) < self.confidence_threshold:
                        continue
                    s = _canon(t["subject"])
                    o = _canon(t["object"])
                    p = t.get("predicate", "rel")
                    self.graph.add_node(s)
                    self.graph.add_node(o)
                    self.graph.add_edge(s, o, predicate=p)
                    self.node_chunks[s].add(chunk_id)
                    self.node_chunks[o].add(chunk_id)
            if verbose:
                print(f"[hipporag] ingested {doc_id}: {len(self.graph)} nodes, {self.graph.number_of_edges()} edges")

    # ─── Retrieve ──────────────────────────────────────────────────────────
    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        """Personalized-PageRank retrieval. Seed = entities in the query."""
        seeds = self._match_entities(query)
        if not seeds or len(self.graph) == 0:
            return []

        personalization = {n: 1.0 if n in seeds else 0.0 for n in self.graph.nodes()}
        try:
            scores = nx.pagerank(self.graph, personalization=personalization, alpha=self.ppr_alpha, max_iter=100)
        except nx.PowerIterationFailedConvergence:
            scores = {n: personalization[n] for n in self.graph.nodes()}

        # Aggregate node scores to chunk scores
        chunk_scores: dict[str, float] = defaultdict(float)
        for node, s in scores.items():
            for chunk_id in self.node_chunks.get(node, ()):
                chunk_scores[chunk_id] += s

        top_chunks = sorted(chunk_scores.items(), key=lambda x: -x[1])[:k]
        return [self.chunks[cid] for cid, _ in top_chunks if cid in self.chunks]

    def _match_entities(self, query: str) -> set[str]:
        """Naive substring match — good enough for v1. Replace with embedding match later."""
        q_lower = query.lower()
        seeds: set[str] = set()
        for node in self.graph.nodes():
            if node.lower() in q_lower or any(tok in q_lower for tok in node.lower().split() if len(tok) > 3):
                seeds.add(node)
        return seeds

    # ─── Stats + persistence ──────────────────────────────────────────────
    def stats(self) -> dict:
        return {
            "n_nodes": len(self.graph) if self.graph else 0,
            "n_edges": self.graph.number_of_edges() if self.graph else 0,
            "n_chunks": len(self.chunks),
            "avg_chunks_per_node": (
                sum(len(s) for s in self.node_chunks.values()) / max(1, len(self.node_chunks))
            ),
        }

    def save(self, path: str | Path) -> None:
        import pickle

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"graph": self.graph, "chunks": self.chunks, "node_chunks": dict(self.node_chunks)}, f)

    @classmethod
    def load(cls, path: str | Path) -> "HippoRAG":
        import pickle

        with open(path, "rb") as f:
            d = pickle.load(f)
        h = cls()
        h.graph = d["graph"]
        h.chunks = d["chunks"]
        h.node_chunks = defaultdict(set, {k: set(v) for k, v in d["node_chunks"].items()})
        return h


def _canon(entity: str) -> str:
    """Canonicalize an entity string for graph nodes."""
    return entity.strip().lower()
