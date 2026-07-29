import os
import yaml
from typing import Dict, Any, Optional

from ai.retrieval.langgraph_agent import VideoIntelligenceGraph
from ai.retrieval.video_clip_generator import VideoClipGenerator

# Import your existing Phase 5 & 6 database managers
try:
    from ai.embeddings.postgres_manager import PostgresManager
except ImportError:
    PostgresManager = None

try:
    from ai.embeddings.chroma_manager import ChromaManager
except ImportError:
    ChromaManager = None


class Phase7Pipeline:
    def __init__(
        self, 
        config_path: str = "ai/configs/phase7.yaml", 
        postgres_mgr=None, 
        chroma_mgr=None
    ):
        self.config = self._load_config(config_path)
        
        # 1. Initialize Postgres Manager if not passed explicitly
        if postgres_mgr is not None:
            self.postgres_mgr = postgres_mgr
        elif PostgresManager is not None:
            try:
                pg_cfg = self.config.get("database", {}).get("postgres", {})
                self.postgres_mgr = PostgresManager(config=pg_cfg)
                print("[Phase 7 Pipeline] PostgreSQL Manager connected successfully.")
            except Exception as e:
                print(f"[Phase 7 Pipeline Warning] Could not connect to PostgreSQL: {e}")
                self.postgres_mgr = None
        else:
            self.postgres_mgr = None

        # 2. Initialize ChromaDB Manager if not passed explicitly
        if chroma_mgr is not None:
            self.chroma_mgr = chroma_mgr
        elif ChromaManager is not None:
            try:
                chroma_cfg = self.config.get("database", {}).get("chromadb", {})
                self.chroma_mgr = ChromaManager(config=chroma_cfg)
                print("[Phase 7 Pipeline] ChromaDB Manager connected successfully.")
            except Exception as e:
                print(f"[Phase 7 Pipeline Warning] Could not connect to ChromaDB: {e}")
                self.chroma_mgr = None
        else:
            self.chroma_mgr = None

        # 3. Setup Clip Generator
        clip_cfg = self.config.get("clip_generation", {})
        self.clip_generator = VideoClipGenerator(
            output_dir=clip_cfg.get("output_dir", "uploads/clips"),
            padding=clip_cfg.get("padding_seconds", 1.5)
        )

        # 4. Initialize LangGraph Agent
        self.agent = VideoIntelligenceGraph(
            postgres_mgr=self.postgres_mgr,
            chroma_mgr=self.chroma_mgr,
            clip_generator=self.clip_generator,
            config=self.config.get("retrieval", {})
        )

    def _load_config(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def query(self, natural_language_query: str, video_path: Optional[str] = None) -> Dict[str, Any]:
        print(f"\n[Phase 7 Pipeline] Processing Query: '{natural_language_query}'")
        results = self.agent.run(user_query=natural_language_query, video_path=video_path)
        return results