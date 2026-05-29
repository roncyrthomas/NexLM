"""Tests for Tier 1 HippoRAG."""

import pytest

pytest.importorskip("networkx")

from agent.hipporag import HippoRAG, regex_stub_extractor


def test_ingest_builds_graph():
    h = HippoRAG()
    docs = [("doc1", "Alice is a engineer. Bob has a dog.")]
    h.ingest_documents(docs, extractor=regex_stub_extractor)
    assert len(h.graph) > 0
    assert h.graph.number_of_edges() > 0
    assert "alice" in h.graph
    assert "bob" in h.graph


def test_retrieve_returns_chunks():
    h = HippoRAG()
    h.ingest_documents(
        [("doc1", "Alice is a engineer. Alice has a cat.")],
        extractor=regex_stub_extractor,
    )
    chunks = h.retrieve("Tell me about Alice", k=3)
    assert len(chunks) >= 1
    assert any("alice" in c.text.lower() for c in chunks)


def test_no_seeds_returns_empty():
    h = HippoRAG()
    h.ingest_documents(
        [("doc1", "Alice is a engineer.")],
        extractor=regex_stub_extractor,
    )
    chunks = h.retrieve("xyzzy quux not in graph", k=3)
    assert chunks == []


def test_save_load_roundtrip(tmp_path):
    h = HippoRAG()
    h.ingest_documents(
        [("doc1", "Alice is a engineer. Bob has a dog.")],
        extractor=regex_stub_extractor,
    )
    path = tmp_path / "kg.pkl"
    h.save(path)
    h2 = HippoRAG.load(path)
    assert len(h2.graph) == len(h.graph)
    assert len(h2.chunks) == len(h.chunks)


def test_stats_dict():
    h = HippoRAG()
    h.ingest_documents([("d", "Alice is a engineer.")], extractor=regex_stub_extractor)
    s = h.stats()
    assert s["n_nodes"] > 0
    assert s["n_chunks"] > 0
