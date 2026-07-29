import os
import json
import subprocess
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

# --- Graph State Definition ---
class RetrievalState(TypedDict):
    user_query: str
    video_path: Optional[str]
    sql_filters: Dict[str, Any]
    vector_query_text: str
    sql_results: List[Dict[str, Any]]
    vector_results: List[Dict[str, Any]]
    merged_results: List[Dict[str, Any]]
    clip_paths: List[str]
    final_response: str


class VideoIntelligenceGraph:
    def __init__(self, postgres_mgr=None, chroma_mgr=None, clip_generator=None, config: Optional[Dict[str, Any]] = None):
        self.postgres_mgr = postgres_mgr
        self.chroma_mgr = chroma_mgr
        self.clip_generator = clip_generator
        self.config = config or {}
        self.graph = self._build_graph()

    def _parse_query_node(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Parses raw user query into structured SQL filter dict and semantic text.
        """
        query = state["user_query"].lower()
        sql_filters = {}
        
        # Color extraction heuristic / rules
        colors = ["red", "blue", "green", "black", "white", "yellow", "silver", "grey", "gray"]
        for color in colors:
            if color in query:
                sql_filters["color"] = color
                break

        # Object category extraction
        objects = ["car", "bus", "truck", "person", "bicycle", "motorcycle"]
        for obj in objects:
            if obj in query:
                sql_filters["object_type"] = obj
                break

        # Action / Motion heuristic
        actions = ["walking", "running", "stopped", "speeding", "turning", "parking"]
        for action in actions:
            if action in query:
                sql_filters["action"] = action
                break

        return {
            "sql_filters": sql_filters,
            "vector_query_text": state["user_query"]
        }

    def _sql_search_node(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Queries PostgreSQL metadata database using extracted attributes.
        """
        filters = state.get("sql_filters", {})
        results = []
        if self.postgres_mgr and hasattr(self.postgres_mgr, "search_events"):
            try:
                results = self.postgres_mgr.search_events(filters, limit=self.config.get("top_k_sql", 20))
            except Exception as e:
                print(f"[SQL Search Warning] {e}")
        return {"sql_results": results}

    def _vector_search_node(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Queries ChromaDB vector collection using text query embeddings.
        """
        query_text = state.get("vector_query_text", "")
        results = []
        if self.chroma_mgr and hasattr(self.chroma_mgr, "search_by_text"):
            try:
                results = self.chroma_mgr.search_by_text(query_text, top_k=self.config.get("top_k_vector", 20))
            except Exception as e:
                print(f"[Vector Search Warning] {e}")
        return {"vector_results": results}

    def _hybrid_rerank_node(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Merges SQL and Vector search results using Reciprocal Rank Fusion (RRF).
        """
        sql_res = state.get("sql_results", [])
        vec_res = state.get("vector_results", [])
        rrf_k = self.config.get("rrf_k", 60)

        scores: Dict[str, float] = {}
        item_map: Dict[str, Dict[str, Any]] = {}

        # Process SQL RRF
        for rank, item in enumerate(sql_res):
            track_id = str(item.get("track_id", f"sql_{rank}"))
            scores[track_id] = scores.get(track_id, 0.0) + (0.5 / (rrf_k + rank + 1))
            item_map[track_id] = item

        # Process Vector RRF
        for rank, item in enumerate(vec_res):
            track_id = str(item.get("track_id", f"vec_{rank}"))
            scores[track_id] = scores.get(track_id, 0.0) + (0.5 / (rrf_k + rank + 1))
            if track_id not in item_map:
                item_map[track_id] = item

        # Sort combined items by fused score
        sorted_track_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        top_k = self.config.get("final_top_k", 5)
        
        merged = []
        for tid in sorted_track_ids[:top_k]:
            record = item_map[tid]
            record["rrf_score"] = round(scores[tid], 4)
            merged.append(record)

        return {"merged_results": merged}

    def _generate_clips_node(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Calls FFmpeg video generator for the top ranked event results.
        """
        video_path = state.get("video_path")
        merged = state.get("merged_results", [])
        clip_paths = []

        if video_path and os.path.exists(video_path) and self.clip_generator:
            for idx, event in enumerate(merged):
                start_t = event.get("start_time", 0.0)
                end_t = event.get("end_time", start_t + 5.0)
                out_path = f"uploads/clips/event_{event.get('track_id', idx)}.mp4"
                
                generated = self.clip_generator.extract_clip(
                    video_path=video_path,
                    start_time=start_t,
                    end_time=end_t,
                    output_path=out_path
                )
                if generated:
                    clip_paths.append(generated)

        return {"clip_paths": clip_paths}

    def _generate_response_node(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Formats final structured response for backend/UI consumption.
        """
        query = state["user_query"]
        results = state.get("merged_results", [])
        clips = state.get("clip_paths", [])

        if not results:
            response_str = f"No events found matching query: '{query}'."
        else:
            response_str = f"Found {len(results)} relevant event(s) for query: '{query}'.\n\n"
            for i, res in enumerate(results, 1):
                track_id = res.get("track_id", "N/A")
                label = res.get("object_type", res.get("label", "object"))
                start = res.get("start_time", 0.0)
                end = res.get("end_time", 0.0)
                score = res.get("rrf_score", 0.0)
                response_str += f"{i}. [{label.upper()}] Track #{track_id} | Time: {start:.1f}s - {end:.1f}s | Confidence: {score}\n"

        return {"final_response": response_str}

    def _build_graph(self) -> Any:
        workflow = StateGraph(RetrievalState)

        # Add Nodes
        workflow.add_node("parse_query", self._parse_query_node)
        workflow.add_node("sql_search", self._sql_search_node)
        workflow.add_node("vector_search", self._vector_search_node)
        workflow.add_node("hybrid_rerank", self._hybrid_rerank_node)
        workflow.add_node("generate_clips", self._generate_clips_node)
        workflow.add_node("generate_response", self._generate_response_node)

        # Build Edges
        workflow.set_entry_point("parse_query")
        
        # Branch to parallel searches
        workflow.add_edge("parse_query", "sql_search")
        workflow.add_edge("parse_query", "vector_search")
        
        # Converge into hybrid reranker
        workflow.add_edge(["sql_search", "vector_search"], "hybrid_rerank")
        workflow.add_edge("hybrid_rerank", "generate_clips")
        workflow.add_edge("generate_clips", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    def run(self, user_query: str, video_path: Optional[str] = None) -> Dict[str, Any]:
        initial_state: RetrievalState = {
            "user_query": user_query,
            "video_path": video_path,
            "sql_filters": {},
            "vector_query_text": "",
            "sql_results": [],
            "vector_results": [],
            "merged_results": [],
            "clip_paths": [],
            "final_response": ""
        }
        return self.graph.invoke(initial_state)