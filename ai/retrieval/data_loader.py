"""
Phase 7

Central Data Loader

Loads metadata once and shares it
across the retrieval engine.
"""

import json
from pathlib import Path


class RetrievalDataLoader:

    def __init__(self, project_root=None):

        if project_root is None:
            project_root = Path.cwd()

        self.project_root = Path(project_root)

        self.semantic_metadata = None
        self.embedding_metadata = None
        self.phase6_results = None

    # ---------------------------------------------------
    # Generic JSON Loader
    # ---------------------------------------------------

    def load_json(self, path):

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---------------------------------------------------
    # Phase 4
    # ---------------------------------------------------

    def load_phase4(self):

        path = (
            self.project_root
            / "outputs"
            / "phase4"
            / "semantic_metadata.json"
        )

        self.semantic_metadata = self.load_json(path)

        print(f"[INFO] Loaded {len(self.semantic_metadata.get('tracks', []))} semantic tracks")

    # ---------------------------------------------------
    # Phase 5
    # ---------------------------------------------------

    def load_phase5(self):

        path = (
            self.project_root
            / "outputs"
            / "phase5"
            / "embedding_metadata.json"
        )

        self.embedding_metadata = self.load_json(path)

        print(f"[INFO] Loaded {len(self.embedding_metadata)} embeddings")

    # ---------------------------------------------------
    # Phase 6
    # ---------------------------------------------------

    def load_phase6(self):

        path = (
            self.project_root
            / "outputs"
            / "phase6_result.json"
        )

        if path.exists():

            self.phase6_results = self.load_json(path)

            print("[INFO] Phase 6 results loaded")

        else:

            print("[WARNING] phase6_result.json not found")

            self.phase6_results = None

    # ---------------------------------------------------
    # Load Everything
    # ---------------------------------------------------

    def load_all(self):

        self.load_phase4()

        self.load_phase5()

        self.load_phase6()

    # ---------------------------------------------------
    # Statistics
    # ---------------------------------------------------

    def statistics(self):

        return {

            "semantic_tracks":
                len(self.semantic_metadata)
                if self.semantic_metadata
                else 0,

            "embeddings":
                len(self.embedding_metadata)
                if self.embedding_metadata
                else 0,

            "phase6_loaded":
                self.phase6_results is not None

        }


# -------------------------------------------------------
# Testing
# -------------------------------------------------------

if __name__ == "__main__":

    loader = RetrievalDataLoader()

    loader.load_all()

    print()

    print(loader.statistics())