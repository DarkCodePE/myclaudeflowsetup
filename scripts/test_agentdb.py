#!/usr/bin/env python3
"""
AgentDB ReasoningBank Test Suite

Comprehensive tests for database initialization, embedding storage,
similarity search, and schema validation.

Usage:
    python scripts/test_agentdb.py [--db PATH] [--verbose]
"""

import sqlite3
import os
import sys
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path


class AgentDBTester:
    """Test suite for AgentDB ReasoningBank database."""

    def __init__(self, db_path: str = ".agentdb/reasoningbank.db", verbose: bool = False):
        self.db_path = db_path
        self.verbose = verbose
        self.test_results = []

    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")

    def run_test(self, test_name: str, test_func):
        """Run a single test and record result."""
        print(f"\n🧪 {test_name}...", end=" ")
        try:
            test_func()
            print("✅ PASSED")
            self.test_results.append((test_name, True, None))
            return True
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            self.test_results.append((test_name, False, str(e)))
            return False
        except Exception as e:
            print(f"❌ ERROR: {e}")
            self.test_results.append((test_name, False, f"Exception: {e}"))
            return False

    def test_database_exists(self):
        """Test that database file exists."""
        self.log(f"Checking database exists at: {self.db_path}")
        assert os.path.exists(self.db_path), f"Database not found at {self.db_path}"
        assert os.path.getsize(self.db_path) > 0, "Database file is empty"

    def test_all_tables_exist(self):
        """Test that all 25 required tables exist."""
        required_tables = [
            'episodes', 'episode_embeddings', 'skills', 'skill_embeddings', 'skill_links',
            'facts', 'notes', 'note_embeddings', 'causal_edges', 'causal_experiments',
            'causal_observations', 'exp_nodes', 'exp_edges', 'exp_node_embeddings',
            'learning_experiences', 'learning_sessions', 'consolidated_memories',
            'consolidation_runs', 'memory_scores', 'memory_access_log',
            'provenance_sources', 'justification_paths', 'recall_certificates',
            'events', 'learning_algorithms'
        ]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = {row[0] for row in cursor.fetchall()}

        self.log(f"Found {len(existing_tables)} tables: {existing_tables}")

        missing = set(required_tables) - existing_tables
        conn.close()

        assert len(missing) == 0, f"Missing tables: {missing}"
        assert len(existing_tables) >= 25, f"Expected at least 25 tables, found {len(existing_tables)}"

    def test_episodes_table_structure(self):
        """Test episodes table has correct columns."""
        required_columns = ['id', 'ts', 'task', 'input', 'output', 'critique',
                          'reward', 'success', 'created_at']

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(episodes);")
        columns = {row[1]: row[2] for row in cursor.fetchall()}

        self.log(f"Episodes columns: {columns}")

        conn.close()

        for col in required_columns:
            assert col in columns, f"Missing column '{col}' in episodes table"

    def test_embedding_tables_structure(self):
        """Test embedding tables have BLOB columns."""
        embedding_tables = ['episode_embeddings', 'skill_embeddings',
                          'note_embeddings', 'exp_node_embeddings']

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for table in embedding_tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

            self.log(f"{table} columns: {columns}")

            assert 'embedding' in columns, f"Missing 'embedding' column in {table}"
            assert 'embedding_model' in columns, f"Missing 'embedding_model' column in {table}"

        conn.close()

    def test_insert_episode(self):
        """Test inserting an episode."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        episode_data = {
            'ts': datetime.now().timestamp(),
            'session_id': 'test_session_001',
            'task': 'Test task execution',
            'input': 'Test input data',
            'output': 'Test output result',
            'critique': 'Good performance',
            'reward': 0.85,
            'success': 1,
            'latency_ms': 150,
            'tokens_used': 42,
            'tags': 'test,example',
            'metadata': '{"test": true}'
        }

        cursor.execute("""
            INSERT INTO episodes (ts, session_id, task, input, output, critique,
                                reward, success, latency_ms, tokens_used, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, tuple(episode_data.values()))

        episode_id = cursor.lastrowid
        self.log(f"Inserted episode with ID: {episode_id}")

        # Verify insertion
        cursor.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
        result = cursor.fetchone()

        conn.commit()
        conn.close()

        assert result is not None, "Episode not found after insertion"
        assert result[3] == 'Test task execution', "Task data mismatch"

    def test_insert_embedding(self):
        """Test storing embedding vector."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create test episode first
        cursor.execute("""
            INSERT INTO episodes (ts, task, input, output)
            VALUES (?, ?, ?, ?)
        """, (datetime.now().timestamp(), 'Embedding test', 'input', 'output'))

        episode_id = cursor.lastrowid

        # Create random embedding (384 dimensions, typical for all-MiniLM-L6-v2)
        embedding_vector = np.random.randn(384).astype(np.float32)
        embedding_blob = embedding_vector.tobytes()

        self.log(f"Embedding vector shape: {embedding_vector.shape}, dtype: {embedding_vector.dtype}")

        cursor.execute("""
            INSERT INTO episode_embeddings (episode_id, embedding, embedding_model)
            VALUES (?, ?, ?)
        """, (episode_id, embedding_blob, 'all-MiniLM-L6-v2'))

        # Retrieve and verify
        cursor.execute("SELECT embedding FROM episode_embeddings WHERE episode_id = ?", (episode_id,))
        result = cursor.fetchone()

        conn.commit()
        conn.close()

        assert result is not None, "Embedding not found"

        retrieved_vector = np.frombuffer(result[0], dtype=np.float32)
        assert retrieved_vector.shape == embedding_vector.shape, "Shape mismatch"
        assert np.allclose(retrieved_vector, embedding_vector), "Vector data mismatch"

    def test_similarity_search(self):
        """Test basic similarity search capability."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert multiple episodes with embeddings
        embeddings = []
        episode_ids = []

        for i in range(5):
            cursor.execute("""
                INSERT INTO episodes (ts, task, input, output)
                VALUES (?, ?, ?, ?)
            """, (datetime.now().timestamp(), f'Task {i}', f'input {i}', f'output {i}'))

            episode_id = cursor.lastrowid
            episode_ids.append(episode_id)

            # Create distinct embeddings
            embedding = np.random.randn(384).astype(np.float32)
            if i == 0:
                query_embedding = embedding.copy()  # Save first as query

            embeddings.append(embedding)

            cursor.execute("""
                INSERT INTO episode_embeddings (episode_id, embedding)
                VALUES (?, ?)
            """, (episode_id, embedding.tobytes()))

        conn.commit()

        # Perform similarity search (cosine similarity)
        cursor.execute("SELECT episode_id, embedding FROM episode_embeddings")
        results = cursor.fetchall()

        similarities = []
        for ep_id, emb_blob in results:
            stored_emb = np.frombuffer(emb_blob, dtype=np.float32)

            # Cosine similarity
            similarity = np.dot(query_embedding, stored_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_emb)
            )
            similarities.append((ep_id, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        self.log(f"Similarity scores: {similarities}")

        conn.close()

        # First result should be the query itself (highest similarity)
        assert similarities[0][0] == episode_ids[0], "Most similar should be the query itself"
        assert similarities[0][1] > 0.99, "Self-similarity should be ~1.0"

    def test_foreign_key_constraints(self):
        """Test foreign key relationships work correctly."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        cursor = conn.cursor()

        # Create parent episode
        cursor.execute("""
            INSERT INTO episodes (ts, task, input, output)
            VALUES (?, ?, ?, ?)
        """, (datetime.now().timestamp(), 'FK test', 'input', 'output'))

        episode_id = cursor.lastrowid

        # Create child records
        embedding = np.random.randn(384).astype(np.float32).tobytes()
        cursor.execute("""
            INSERT INTO episode_embeddings (episode_id, embedding)
            VALUES (?, ?)
        """, (episode_id, embedding))

        cursor.execute("""
            INSERT INTO memory_scores (episode_id, composite_score)
            VALUES (?, ?)
        """, (episode_id, 0.75))

        # Verify children exist
        cursor.execute("SELECT COUNT(*) FROM episode_embeddings WHERE episode_id = ?", (episode_id,))
        assert cursor.fetchone()[0] == 1, "Embedding not created"

        cursor.execute("SELECT COUNT(*) FROM memory_scores WHERE episode_id = ?", (episode_id,))
        assert cursor.fetchone()[0] == 1, "Memory score not created"

        # Delete parent (should cascade)
        cursor.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))

        # Verify children are deleted
        cursor.execute("SELECT COUNT(*) FROM episode_embeddings WHERE episode_id = ?", (episode_id,))
        assert cursor.fetchone()[0] == 0, "Embedding not cascaded"

        cursor.execute("SELECT COUNT(*) FROM memory_scores WHERE episode_id = ?", (episode_id,))
        assert cursor.fetchone()[0] == 0, "Memory score not cascaded"

        conn.commit()
        conn.close()

    def test_indexes_exist(self):
        """Test that performance indexes are created."""
        required_indexes = [
            'idx_episodes_ts',
            'idx_episodes_session',
            'idx_episode_emb_episode',
            'idx_memory_scores_composite'
        ]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
        existing_indexes = {row[0] for row in cursor.fetchall()}

        self.log(f"Found indexes: {existing_indexes}")

        conn.close()

        for idx in required_indexes:
            assert idx in existing_indexes, f"Missing index: {idx}"

    def test_skill_learning_workflow(self):
        """Test complete skill learning workflow."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 1. Create skill
        cursor.execute("""
            INSERT INTO skills (name, description, code, success_rate)
            VALUES (?, ?, ?, ?)
        """, ('test_skill', 'A test skill', 'def test(): pass', 0.0))

        skill_id = cursor.lastrowid

        # 2. Create episode
        cursor.execute("""
            INSERT INTO episodes (ts, task, input, output, reward, success)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.now().timestamp(), 'skill test', 'input', 'output', 0.9, 1))

        episode_id = cursor.lastrowid

        # 3. Link skill to episode
        cursor.execute("""
            INSERT INTO skill_links (episode_id, skill_id, relevance)
            VALUES (?, ?, ?)
        """, (episode_id, skill_id, 0.95))

        # 4. Update skill stats
        cursor.execute("""
            UPDATE skills
            SET success_rate = 0.9, usage_count = usage_count + 1
            WHERE id = ?
        """, (skill_id,))

        # Verify workflow
        cursor.execute("""
            SELECT s.name, s.success_rate, e.reward
            FROM skills s
            JOIN skill_links sl ON s.id = sl.skill_id
            JOIN episodes e ON sl.episode_id = e.id
            WHERE s.id = ?
        """, (skill_id,))

        result = cursor.fetchone()

        conn.commit()
        conn.close()

        assert result is not None, "Skill learning workflow incomplete"
        assert result[0] == 'test_skill', "Skill name mismatch"
        assert result[1] == 0.9, "Success rate not updated"
        assert result[2] == 0.9, "Reward mismatch"

    def print_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)

        print(f"Total tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {total - passed} ❌")
        print(f"Success rate: {(passed/total*100):.1f}%")

        if any(not success for _, success, _ in self.test_results):
            print("\nFailed tests:")
            for name, success, error in self.test_results:
                if not success:
                    print(f"  ❌ {name}: {error}")

        return passed == total

    def run_all_tests(self):
        """Run complete test suite."""
        print("=" * 60)
        print("AgentDB ReasoningBank Test Suite")
        print("=" * 60)
        print(f"Database: {self.db_path}")
        print(f"Verbose: {self.verbose}")

        # Run all tests
        self.run_test("Database file exists", self.test_database_exists)
        self.run_test("All 25 tables exist", self.test_all_tables_exist)
        self.run_test("Episodes table structure", self.test_episodes_table_structure)
        self.run_test("Embedding tables structure", self.test_embedding_tables_structure)
        self.run_test("Insert episode", self.test_insert_episode)
        self.run_test("Insert and retrieve embedding", self.test_insert_embedding)
        self.run_test("Similarity search", self.test_similarity_search)
        self.run_test("Foreign key constraints", self.test_foreign_key_constraints)
        self.run_test("Performance indexes", self.test_indexes_exist)
        self.run_test("Skill learning workflow", self.test_skill_learning_workflow)

        return self.print_summary()


def main():
    """Command-line interface for test suite."""
    parser = argparse.ArgumentParser(
        description="Test AgentDB ReasoningBank database",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--db',
        default='.agentdb/reasoningbank.db',
        help='Database path (default: .agentdb/reasoningbank.db)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    tester = AgentDBTester(db_path=args.db, verbose=args.verbose)

    success = tester.run_all_tests()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
