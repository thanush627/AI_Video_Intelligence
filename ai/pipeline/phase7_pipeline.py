import os
import yaml
from typing import Dict, Any, Optional

from ai.retrieval.langgraph_agent import VideoIntelligenceGraph
from ai.retrieval.video_clip_generator import VideoClipGenerator

class Phase7Pipeline:
    def __init__(self, config_path: str = "ai/configs/phase7.yaml", postgres_mgr=None, chroma_mgr=None):
        self.config = self._load_config(config_path)
        self.postgres_mgr = postgres_mgr
        self.chroma_mgr = chroma_mgr
        
        clip_cfg = self.config.get("clip_generation", {})
        self.clip_generator = VideoClipGenerator(
            output_dir=clip_cfg.get("output_dir", "uploads/clips"),
            padding=clip_cfg.get("padding_seconds", 1.5)
        )

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