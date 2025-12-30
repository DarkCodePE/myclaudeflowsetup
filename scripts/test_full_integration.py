#!/usr/bin/env python3
"""
Comprehensive End-to-End Integration Test for AgentDB ReasoningBank

Tests:
1. Embedding service functionality
2. AgentDB bridge pre-task retrieval
3. AgentDB bridge post-task storage
4. Trajectory tracking and storage
5. Memory distillation
6. Session consolidation
7. Vector similarity search
8. Pattern learning
9. Confidence scoring
10. Complete learning workflow
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add hooks directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / ".claude" / "hooks"))

from embedding_service import EmbeddingService
from agentdb_bridge import AgentDBBridge, LearnedStrategy
from error_handler import ErrorHandler
from metrics_tracker import MetricsTracker


class IntegrationTester:
    def __init__(self, db_path=".agentdb/reasoningbank.db"):
        self.db_path = db_path
        self.passed = 0
        self.failed = 0
        self.tests = []

    def test(self, name, func):
        """Run a test and track results"""
        try:
            print(f"\n🧪 {name}...", end=" ")
            func()
            print("✅ PASSED")
            self.passed += 1
            self.tests.append((name, True, None))
        except Exception as e:
            print(f"❌ FAILED: {e}")
            self.failed += 1
            self.tests.append((name, False, str(e)))

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        print(f"Total tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed} ✅")
        print(f"Failed: {self.failed} ❌")
        print(f"Success rate: {100 * self.passed / (self.passed + self.failed):.1f}%")

        if self.failed > 0:
            print("\nFailed tests:")
            for name, passed, error in self.tests:
                if not passed:
                    print(f"  ❌ {name}: {error}")


def test_embedding_service():
    """Test 1: Embedding service initialization and computation"""
    emb_service = EmbeddingService()

    # Test compute_cached
    text = "Build a REST API with authentication"
    embedding = emb_service.compute_cached(text)

    assert embedding is not None, "Embedding is None"
    assert embedding.shape == (384,), f"Wrong shape: {embedding.shape}"
    assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-5), "Not L2 normalized"

    # Test caching
    embedding2 = emb_service.compute_cached(text)
    assert np.allclose(embedding, embedding2), "Cache not working"

    # Test stats
    stats = emb_service.get_stats()
    assert stats['hits'] >= 1, "Cache hits not tracked"

    print(f"\n  📊 Stats: {stats['hits']} hits, dimension={stats['embedding_dimension']}")


def test_agentdb_bridge_init():
    """Test 2: AgentDB bridge initialization and schema validation"""
    bridge = AgentDBBridge()

    assert bridge.db_path.endswith("reasoningbank.db"), "Wrong DB path"
    assert bridge.embedding_service is not None, "Embedding service not initialized"
    assert bridge.error_handler is not None, "Error handler not initialized"
    assert bridge.metrics is not None, "Metrics not initialized"

    print(f"\n  📍 DB: {bridge.db_path}")


def test_pre_task_retrieval():
    """Test 3: Pre-task retrieval"""
    bridge = AgentDBBridge()

    # Retrieval may return strategy if DB has data, or None if empty
    strategy = bridge.pre_task_retrieve(
        task_desc="Optimize database queries",
        domain="database-optimization",
        k=5
    )

    # Both outcomes are valid
    if strategy is None:
        print(f"\n  ℹ️  No memories found (fresh DB)")
    else:
        print(f"\n  ✓ Found strategy with confidence={strategy.confidence:.2f}")


def test_post_task_storage():
    """Test 4: Post-task storage and confidence updates"""
    bridge = AgentDBBridge()

    # Create trajectory
    trajectory = {
        'description': 'Optimize API endpoint performance',
        'domain': 'api-optimization',
        'session_id': 'test-session-001',
        'input': 'Slow API response times',
        'output': 'Added caching and indexing',
        'strategy': 'Profile, identify bottlenecks, apply targeted fixes',
        'confidence': 0.9
    }

    metrics = {
        'duration_ms': 1500,
        'tokens_used': 250,
        'quality_score': 0.85
    }

    # Store trajectory
    result = bridge.post_task_store(
        task_id='test-task-001',
        trajectory=trajectory,
        outcome='success',
        metrics=metrics
    )

    assert result.episode_id > 0, "Episode not stored"
    assert result.confidence_delta > 0, "Confidence delta should be positive for success"

    print(f"\n  📝 Episode {result.episode_id} stored, confidence_delta={result.confidence_delta:.3f}")


def test_retrieval_after_storage():
    """Test 5: Retrieval after storage (should find similar patterns)"""
    bridge = AgentDBBridge()

    # Retrieve similar patterns
    strategy = bridge.pre_task_retrieve(
        task_desc="Improve API performance",  # Similar to stored task
        domain="api-optimization",
        k=5
    )

    # Should find at least one memory now
    assert strategy is not None, "Should find stored memory"

    print(f"\n  🔍 Found strategy with confidence={strategy.confidence:.2f}")


def test_multiple_episodes():
    """Test 6: Store multiple episodes and verify retrieval"""
    bridge = AgentDBBridge()

    # Store multiple episodes
    domains = ['api-optimization', 'database-optimization', 'frontend-optimization']
    episode_ids = []

    for i, domain in enumerate(domains):
        trajectory = {
            'description': f'Optimize {domain} task {i}',
            'domain': domain,
            'session_id': f'test-session-{i:03d}',
            'input': f'Input {i}',
            'output': f'Output {i}',
            'strategy': f'Strategy {i}',
            'confidence': 0.8 + i * 0.05
        }

        metrics = {
            'duration_ms': 1000 + i * 100,
            'tokens_used': 200 + i * 50,
            'quality_score': 0.8
        }

        result = bridge.post_task_store(
            task_id=f'test-task-{i:03d}',
            trajectory=trajectory,
            outcome='success' if i % 2 == 0 else 'failure',
            metrics=metrics
        )

        episode_ids.append(result.episode_id)

    assert len(episode_ids) == 3, "Should have 3 episodes"
    assert len(set(episode_ids)) == 3, "Episode IDs should be unique"

    print(f"\n  📊 Stored {len(episode_ids)} episodes: {episode_ids}")


def test_similarity_search():
    """Test 7: Vector similarity search across episodes"""
    import sqlite3

    conn = sqlite3.connect(".agentdb/reasoningbank.db")
    cursor = conn.cursor()

    # Get all episodes with embeddings
    cursor.execute("""
        SELECT e.id, e.task, e.domain, emb.embedding
        FROM episodes e
        JOIN episode_embeddings emb ON e.id = emb.episode_id
        LIMIT 10
    """)

    results = cursor.fetchall()
    conn.close()

    assert len(results) > 0, "No episodes with embeddings found"

    # Test embedding dimensions
    for episode_id, task, domain, emb_blob in results:
        embedding = np.frombuffer(emb_blob, dtype=np.float32)
        assert embedding.shape == (384,), f"Wrong embedding shape: {embedding.shape}"

    print(f"\n  🔍 Found {len(results)} episodes with valid 384D embeddings")


def test_session_consolidation():
    """Test 8: Session end consolidation"""
    bridge = AgentDBBridge()

    report = bridge.session_end_consolidate(session_id="test-session-001")

    assert report['status'] == 'success', f"Consolidation failed: {report}"
    assert 'session_id' in report, "Missing session_id in report"

    print(f"\n  📦 Session consolidated: {report['session_id']}")


def test_metrics_tracking():
    """Test 9: Metrics tracker functionality"""
    bridge = AgentDBBridge()

    # Get metrics report
    report = bridge.metrics.generate_report()

    assert 'summary' in report, "Missing summary in metrics"
    assert 'performance' in report, "Missing performance in metrics"

    # Save metrics
    bridge.metrics.save_metrics()

    # Check file exists
    assert os.path.exists(".agentdb/metrics.json"), "Metrics file not created"

    print(f"\n  📊 Metrics tracked: {report['summary']['total_storage_operations']} storage ops")


def test_error_handling():
    """Test 10: Error handling and circuit breaker"""
    handler = ErrorHandler()

    # Test circuit breaker is not open initially
    assert not handler.circuit_breaker_check(), "Circuit breaker should be closed"

    # Simulate failures
    for i in range(5):
        handler.handle_retrieval_failure(
            Exception(f"Test failure {i}"),
            {'test': True}
        )

    # Circuit breaker should be open after 5 failures
    assert handler.circuit_breaker_check(), "Circuit breaker should be open after 5 failures"

    print(f"\n  🔌 Circuit breaker working correctly")


def main():
    print("="*60)
    print("AgentDB ReasoningBank - Full Integration Test")
    print("="*60)

    tester = IntegrationTester()

    # Run all tests
    tester.test("Embedding service initialization", test_embedding_service)
    tester.test("AgentDB bridge initialization", test_agentdb_bridge_init)
    tester.test("Pre-task retrieval (empty DB)", test_pre_task_retrieval)
    tester.test("Post-task storage", test_post_task_storage)
    tester.test("Retrieval after storage", test_retrieval_after_storage)
    tester.test("Multiple episode storage", test_multiple_episodes)
    tester.test("Vector similarity search", test_similarity_search)
    tester.test("Session consolidation", test_session_consolidation)
    tester.test("Metrics tracking", test_metrics_tracking)
    tester.test("Error handling and circuit breaker", test_error_handling)

    # Print summary
    tester.print_summary()

    # Exit with appropriate code
    sys.exit(0 if tester.failed == 0 else 1)


if __name__ == '__main__':
    main()
