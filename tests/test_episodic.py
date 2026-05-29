"""Tests for Tier E — Episodic memory."""

from agent.episodic import EpisodicMemory


def test_empty_recall():
    em = EpisodicMemory()
    assert em.recall("anything") == []
    assert em.confidence("anything") == 0.0
    assert not em.is_familiar("anything")


def test_remember_and_recall_same_query():
    em = EpisodicMemory()
    em.remember("How do I sort a list?", "Use sorted(list)", reward=1.0)
    hits = em.recall("How do I sort a list?", k=1)
    assert len(hits) == 1
    assert hits[0].response.startswith("Use sorted")


def test_similar_query_recalls_close():
    em = EpisodicMemory(similarity_threshold=0.3)
    em.remember("How do I sort a list?", "Use sorted()", reward=1.0)
    em.remember("How do I make pancakes?", "Mix and fry", reward=1.0)
    hits = em.recall("How do I sort an array?", k=1)
    assert hits[0].response.startswith("Use sorted")


def test_pruning_keeps_high_utility():
    em = EpisodicMemory(max_size=3)
    em.remember("A", "response_a", reward=1.0)
    em.remember("B", "response_b", reward=0.0)
    em.remember("C", "response_c", reward=1.0)
    em.remember("D", "response_d", reward=1.0)
    assert len(em.episodes) <= 3
    rewards = {e.query for e in em.episodes if e.reward >= 0.5}
    assert "A" in rewards or "C" in rewards or "D" in rewards


def test_stats_report():
    em = EpisodicMemory()
    em.remember("q1", "r1", reward=0.5)
    em.remember("q2", "r2", reward=1.0)
    s = em.stats()
    assert s["size"] == 2
    assert s["avg_reward"] > 0
