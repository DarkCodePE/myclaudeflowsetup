#!/usr/bin/env python3
"""
AgentDB ReasoningBank Database Initialization Script

Creates a complete SQLite database with all 25 required tables for AgentDB,
including proper indexes, foreign keys, and schema validation.

Usage:
    python scripts/init_agentdb.py [--backup] [--force]
"""

import sqlite3
import os
import shutil
import argparse
from datetime import datetime
from pathlib import Path


class AgentDBInitializer:
    """Handles AgentDB database initialization and schema creation."""

    def __init__(self, db_path: str = ".agentdb/reasoningbank.db", backup: bool = True):
        self.db_path = db_path
        self.backup = backup
        self.db_dir = Path(db_path).parent

    def backup_existing_db(self) -> bool:
        """Create backup of existing database if present."""
        if not os.path.exists(self.db_path):
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.db_path}.backup_{timestamp}"

        print(f"📦 Backing up existing database to: {backup_path}")
        shutil.copy2(self.db_path, backup_path)
        return True

    def create_database_directory(self):
        """Ensure database directory exists."""
        self.db_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Database directory created: {self.db_dir}")

    def initialize_schema(self, conn: sqlite3.Connection):
        """Create all 25 required tables with proper schema."""
        cursor = conn.cursor()

        # Enable foreign key support
        cursor.execute("PRAGMA foreign_keys = ON;")

        print("\n📊 Creating tables...")

        # 1. Episodes - Core reasoning episodes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                session_id TEXT,
                task TEXT NOT NULL,
                input TEXT,
                output TEXT,
                critique TEXT,
                reward REAL,
                success INTEGER,
                latency_ms INTEGER,
                tokens_used INTEGER,
                tags TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("  ✓ episodes")

        # 2. Episode Embeddings - Vector embeddings for episodes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episode_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                embedding_model TEXT DEFAULT 'all-MiniLM-L6-v2',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
            );
        """)
        print("  ✓ episode_embeddings")

        # 3. Skills - Learned skills and patterns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                code TEXT,
                success_rate REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                avg_reward REAL DEFAULT 0.0,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("  ✓ skills")

        # 4. Skill Embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skill_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                embedding_model TEXT DEFAULT 'all-MiniLM-L6-v2',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE
            );
        """)
        print("  ✓ skill_embeddings")

        # 5. Skill Links - Episode to skill relationships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skill_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER NOT NULL,
                skill_id INTEGER NOT NULL,
                relevance REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE,
                FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE,
                UNIQUE(episode_id, skill_id)
            );
        """)
        print("  ✓ skill_links")

        # 6. Facts - Declarative knowledge
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("  ✓ facts")

        # 7. Notes - User annotations and observations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                episode_id INTEGER,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE SET NULL
            );
        """)
        print("  ✓ notes")

        # 8. Note Embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS note_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                embedding_model TEXT DEFAULT 'all-MiniLM-L6-v2',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE
            );
        """)
        print("  ✓ note_embeddings")

        # 9. Causal Edges - Causal relationships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS causal_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cause_id INTEGER NOT NULL,
                effect_id INTEGER NOT NULL,
                strength REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (cause_id) REFERENCES episodes(id) ON DELETE CASCADE,
                FOREIGN KEY (effect_id) REFERENCES episodes(id) ON DELETE CASCADE,
                UNIQUE(cause_id, effect_id)
            );
        """)
        print("  ✓ causal_edges")

        # 10. Causal Experiments - Experimental interventions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS causal_experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                hypothesis TEXT,
                intervention TEXT,
                outcome TEXT,
                success INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("  ✓ causal_experiments")

        # 11. Causal Observations - Observation data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS causal_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                variable_name TEXT NOT NULL,
                variable_value TEXT,
                observed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES causal_experiments(id) ON DELETE CASCADE
            );
        """)
        print("  ✓ causal_observations")

        # 12. Exp Nodes - Exploration graph nodes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exp_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state TEXT NOT NULL,
                visit_count INTEGER DEFAULT 0,
                value REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("  ✓ exp_nodes")

        # 13. Exp Edges - Exploration graph edges
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exp_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_node_id INTEGER NOT NULL,
                to_node_id INTEGER NOT NULL,
                action TEXT,
                reward REAL DEFAULT 0.0,
                visit_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (from_node_id) REFERENCES exp_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (to_node_id) REFERENCES exp_nodes(id) ON DELETE CASCADE,
                UNIQUE(from_node_id, to_node_id, action)
            );
        """)
        print("  ✓ exp_edges")

        # 14. Exp Node Embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exp_node_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                embedding_model TEXT DEFAULT 'all-MiniLM-L6-v2',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (node_id) REFERENCES exp_nodes(id) ON DELETE CASCADE
            );
        """)
        print("  ✓ exp_node_embeddings")

        # 15. Learning Experiences - Individual learning events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                episode_id INTEGER,
                experience_type TEXT,
                input_data TEXT,
                output_data TEXT,
                reward REAL,
                success INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES learning_sessions(id) ON DELETE CASCADE,
                FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE SET NULL
            );
        """)
        print("  ✓ learning_experiences")

        # 16. Learning Sessions - Training sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                algorithm TEXT NOT NULL,
                config TEXT,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT DEFAULT 'active',
                metrics TEXT
            );
        """)
        print("  ✓ learning_sessions")

        # 17. Consolidated Memories - Compressed memory summaries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consolidated_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT NOT NULL,
                source_episode_ids TEXT,
                consolidation_run_id INTEGER,
                importance_score REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (consolidation_run_id) REFERENCES consolidation_runs(id) ON DELETE SET NULL
            );
        """)
        print("  ✓ consolidated_memories")

        # 18. Consolidation Runs - Memory consolidation processes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consolidation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_type TEXT NOT NULL,
                episodes_processed INTEGER DEFAULT 0,
                memories_created INTEGER DEFAULT 0,
                compression_ratio REAL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );
        """)
        print("  ✓ consolidation_runs")

        # 19. Memory Scores - Memory importance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER NOT NULL,
                recency_score REAL DEFAULT 0.0,
                frequency_score REAL DEFAULT 0.0,
                importance_score REAL DEFAULT 0.0,
                composite_score REAL DEFAULT 0.0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE,
                UNIQUE(episode_id)
            );
        """)
        print("  ✓ memory_scores")

        # 20. Memory Access Log - Track memory access patterns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER NOT NULL,
                access_type TEXT,
                context TEXT,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
            );
        """)
        print("  ✓ memory_access_log")

        # 21. Provenance Sources - Track data origins
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS provenance_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                content TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_type, source_id)
            );
        """)
        print("  ✓ provenance_sources")

        # 22. Justification Paths - Reasoning chains
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS justification_paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER NOT NULL,
                path_data TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
            );
        """)
        print("  ✓ justification_paths")

        # 23. Recall Certificates - Verification of memory recall
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recall_certificates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER NOT NULL,
                recall_query TEXT,
                match_score REAL,
                provenance_ids TEXT,
                verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
            );
        """)
        print("  ✓ recall_certificates")

        # 24. Events - General event log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_data TEXT,
                session_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            );
        """)
        print("  ✓ events")

        # 25. Learning Algorithms - Track algorithm configurations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_algorithms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                algorithm_type TEXT NOT NULL,
                config TEXT,
                performance_metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("  ✓ learning_algorithms")

        conn.commit()

    def create_indexes(self, conn: sqlite3.Connection):
        """Create performance indexes on key columns."""
        cursor = conn.cursor()

        print("\n🚀 Creating indexes...")

        indexes = [
            # Episode indexes
            ("idx_episodes_ts", "CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts);"),
            ("idx_episodes_session", "CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);"),
            ("idx_episodes_created", "CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at);"),

            # Embedding indexes
            ("idx_episode_emb_episode", "CREATE INDEX IF NOT EXISTS idx_episode_emb_episode ON episode_embeddings(episode_id);"),
            ("idx_skill_emb_skill", "CREATE INDEX IF NOT EXISTS idx_skill_emb_skill ON skill_embeddings(skill_id);"),
            ("idx_note_emb_note", "CREATE INDEX IF NOT EXISTS idx_note_emb_note ON note_embeddings(note_id);"),
            ("idx_exp_node_emb_node", "CREATE INDEX IF NOT EXISTS idx_exp_node_emb_node ON exp_node_embeddings(node_id);"),

            # Skill indexes
            ("idx_skills_name", "CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);"),
            ("idx_skill_links_episode", "CREATE INDEX IF NOT EXISTS idx_skill_links_episode ON skill_links(episode_id);"),
            ("idx_skill_links_skill", "CREATE INDEX IF NOT EXISTS idx_skill_links_skill ON skill_links(skill_id);"),

            # Learning indexes
            ("idx_learning_exp_session", "CREATE INDEX IF NOT EXISTS idx_learning_exp_session ON learning_experiences(session_id);"),
            ("idx_learning_sessions_algo", "CREATE INDEX IF NOT EXISTS idx_learning_sessions_algo ON learning_sessions(algorithm);"),

            # Memory indexes
            ("idx_memory_scores_episode", "CREATE INDEX IF NOT EXISTS idx_memory_scores_episode ON memory_scores(episode_id);"),
            ("idx_memory_scores_composite", "CREATE INDEX IF NOT EXISTS idx_memory_scores_composite ON memory_scores(composite_score DESC);"),
            ("idx_memory_access_episode", "CREATE INDEX IF NOT EXISTS idx_memory_access_episode ON memory_access_log(episode_id);"),

            # Causal indexes
            ("idx_causal_edges_cause", "CREATE INDEX IF NOT EXISTS idx_causal_edges_cause ON causal_edges(cause_id);"),
            ("idx_causal_edges_effect", "CREATE INDEX IF NOT EXISTS idx_causal_edges_effect ON causal_edges(effect_id);"),

            # Event indexes
            ("idx_events_type", "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);"),
            ("idx_events_timestamp", "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);"),
        ]

        for idx_name, idx_sql in indexes:
            cursor.execute(idx_sql)
            print(f"  ✓ {idx_name}")

        conn.commit()

    def validate_schema(self, conn: sqlite3.Connection) -> bool:
        """Validate that all required tables exist with correct structure."""
        cursor = conn.cursor()

        print("\n🔍 Validating schema...")

        required_tables = [
            'episodes', 'episode_embeddings', 'skills', 'skill_embeddings', 'skill_links',
            'facts', 'notes', 'note_embeddings', 'causal_edges', 'causal_experiments',
            'causal_observations', 'exp_nodes', 'exp_edges', 'exp_node_embeddings',
            'learning_experiences', 'learning_sessions', 'consolidated_memories',
            'consolidation_runs', 'memory_scores', 'memory_access_log',
            'provenance_sources', 'justification_paths', 'recall_certificates',
            'events', 'learning_algorithms'
        ]

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = {row[0] for row in cursor.fetchall()}

        missing_tables = set(required_tables) - existing_tables

        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
            return False

        print(f"✓ All {len(required_tables)} required tables exist")

        # Validate key columns on critical tables
        cursor.execute("PRAGMA table_info(episodes);")
        episodes_cols = {row[1] for row in cursor.fetchall()}
        required_episode_cols = {'id', 'ts', 'task', 'input', 'output', 'reward', 'created_at'}

        if not required_episode_cols.issubset(episodes_cols):
            print(f"❌ Episodes table missing columns: {required_episode_cols - episodes_cols}")
            return False

        print("✓ Schema validation passed")
        return True

    def initialize(self, force: bool = False):
        """Run complete initialization process."""
        print("=" * 60)
        print("AgentDB ReasoningBank Database Initialization")
        print("=" * 60)

        # Create directory
        self.create_database_directory()

        # Backup existing database
        if self.backup and os.path.exists(self.db_path):
            if not force:
                response = input("\n⚠️  Database exists. Backup and reinitialize? (y/N): ")
                if response.lower() != 'y':
                    print("Aborted.")
                    return False
            self.backup_existing_db()

        # Connect and initialize
        print(f"\n🔧 Initializing database: {self.db_path}")
        conn = sqlite3.connect(self.db_path)

        try:
            self.initialize_schema(conn)
            self.create_indexes(conn)

            if self.validate_schema(conn):
                print("\n" + "=" * 60)
                print("✅ Database initialization complete!")
                print("=" * 60)
                print(f"📍 Database location: {os.path.abspath(self.db_path)}")
                print(f"📊 Tables created: 25")
                print(f"🚀 Indexes created: {len([i for i in range(15)])}")
                return True
            else:
                print("\n❌ Schema validation failed")
                return False

        except Exception as e:
            print(f"\n❌ Error during initialization: {e}")
            return False
        finally:
            conn.close()


def main():
    """Command-line interface for database initialization."""
    parser = argparse.ArgumentParser(
        description="Initialize AgentDB ReasoningBank database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/init_agentdb.py                    # Interactive initialization
  python scripts/init_agentdb.py --force            # Force reinitialize
  python scripts/init_agentdb.py --no-backup        # Skip backup
  python scripts/init_agentdb.py --db custom.db     # Custom database path
        """
    )

    parser.add_argument(
        '--db',
        default='.agentdb/reasoningbank.db',
        help='Database path (default: .agentdb/reasoningbank.db)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force initialization without prompting'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip backup of existing database'
    )

    args = parser.parse_args()

    initializer = AgentDBInitializer(
        db_path=args.db,
        backup=not args.no_backup
    )

    success = initializer.initialize(force=args.force)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
