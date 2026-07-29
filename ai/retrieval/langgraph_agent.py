import os
import json
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

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
        query = state["user_query"].lower()
        sql_filters = {}
        
        colors = ["red", "blue", "green", "black", "white", "yellow", "silver", "grey", "gray"]
        for color in colors:
            if color in query:
                sql_filters["color"] = color
                break

        objects = ["car", "bus", "truck", "person", "bicycle", "motorcycle"]
        for obj in objects:
            if obj in query:
                sql_filters["object_type"] = obj
                break

        actions = ["walking", "running", "stopped", "speeding", "turning", "parking", "moving"]
        for action in actions:
            if action in query:
                sql_filters["action"] = action
                break

        return {
            "sql_filters": sql_filters,
            "vector_query_text": state["user_query"]
        }

    def _sql_search_node(self, state: RetrievalState) -> Dict[str, Any]:
        filters = state.get("sql_filters", {})
        results = []
        
        if self.postgres_mgr:
            try:
                if hasattr(self.postgres_mgr, "search_events"):
                    results = self.postgres_mgr.search_events(filters, limit=self.config.get("top_k_sql", 20))
                elif hasattr(self.postgres_mgr, "query"):
                    results = self.postgres_mgr.query(filters)
            except Exception as e:
                print(f"[SQL Search Exception] {e}")

        # Fallback to local tracks.json if database isn't populated on current Kaggle run
        if not results and os.path.exists("tracks.json"):
            try:
                with open("tracks.json", "r") as f:
                    tracks = json.load(f)
                    for track_id, data in tracks.items():
                        obj_type = data.get("object_type", "").lower()
                        color = data.get("color", "").lower()
                        
                        target_obj = filters.get("object_type", "")
                        target_col = filters.get("color", "")

                        if (not target_obj or target_obj in obj_type) and (not target_col or target_col in color):
                            results.append({
                                "track_id": track_id,
                                "object_type": data.get("object_type", "object"),
                                "color": data.get("color", "unknown"),
                                "start_time": data.get("start_time", 0.0),
                                "end_time": data.get("end_time", 5.0)
                            })
            except Exception as e:
                print(f"[Tracks JSON Fallback Warning] {e}")

        return {"sql_results": results}

    def _vector_search_node(self, state: RetrievalState) -> Dict[str, Any]:
        query_text = state.get("vector_query_text", "")
        results = []

        if self.chroma_mgr:
            try:
                if hasattr(self.chroma_mgr, "semantic_search"):
                    results = self.chroma_mgr.semantic_search(
                        query=query_text,
                        top_k=self.config.get("top_k_vector", 20),
                    )

                elif hasattr(self.chroma_mgr, "search"):
                    results = self.chroma_mgr.search(
                        query=query_text,
                        top_k=self.config.get("top_k_vector", 20),
                    )

                elif hasattr(self.chroma_mgr, "search_by_text"):
                    results = self.chroma_mgr.search_by_text(
                        query_text,
                        top_k=self.config.get("top_k_vector", 20),
                    )

                elif hasattr(self.chroma_mgr, "query"):
                    results = self.chroma_mgr.query(query_text)

                print(f"[Vector Search] Retrieved {len(results)} results")

            except Exception as e:
                print(f"[Vector Search Exception] {e}")

        return {"vector_results": results}

    def _hybrid_rerank_node(self, state: RetrievalState) -> Dict[str, Any]:
        sql_res = state.get("sql_results", [])
        vec_res = state.get("vector_results", [])
        rrf_k = self.config.get("rrf_k", 60)

        scores: Dict[str, float] = {}
        item_map: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(sql_res):
            track_id = str(item.get("track_id", f"sql_{rank}"))
            scores[track_id] = scores.get(track_id, 0.0) + (0.5 / (rrf_k + rank + 1))
            item_map[track_id] = item

        for rank, item in enumerate(vec_res):
            track_id = str(item.get("track_id", f"vec_{rank}"))
            scores[track_id] = scores.get(track_id, 0.0) + (0.5 / (rrf_k + rank + 1))
            if track_id not in item_map:
                item_map[track_id] = item

        sorted_track_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        top_k = self.config.get("final_top_k", 5)
        
        merged = []
        for tid in sorted_track_ids[:top_k]:
            record = item_map[tid]
            record["rrf_score"] = round(scores[tid], 4)
            merged.append(record)

        return {"merged_results": merged}

    def _generate_clips_node(self, state: RetrievalState) -> Dict[str, Any]:
        video_path = state.get("video_path")
        merged = state.get("merged_results", [])
        clip_paths = []

        if video_path and os.path.exists(video_path) and self.clip_generator:
            for idx, event in enumerate(merged):
                start_t = float(event.get("start_time", 0.0))
                end_t = float(event.get("end_time", start_t + 5.0))
                track_id = event.get("track_id", idx)
                out_path = f"uploads/clips/event_{track_id}.mp4"
                
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
        query = state["user_query"]
        results = state.get("merged_results", [])

        if not results:
            response_str = f"No events found matching query: '{query}'."
        else:
            response_str = f"Found {len(results)} relevant event(s) for query: '{query}'.\n\n"
            for i, res in enumerate(results, 1):
                track_id = res.get("track_id", "N/A")
                label = res.get("object_type", res.get("label", "object"))
                color = res.get("color", "")
                start = float(res.get("start_time", 0.0))
                end = float(res.get("end_time", 0.0))
                score = res.get("rrf_score", 0.0)
                
                color_str = f" ({color})" if color else ""
                response_str += f"{i}. [{label.upper()}{color_str}] Track #{track_id} | Time: {start:.1f}s - {end:.1f}s | RRF Score: {score}\n"

        return {"final_response": response_str}

    def _build_graph(self) -> Any:
        workflow = StateGraph(RetrievalState)

        workflow.add_node("parse_query", self._parse_query_node)
        workflow.add_node("sql_search", self._sql_search_node)
        workflow.add_node("vector_search", self._vector_search_node)
        workflow.add_node("hybrid_rerank", self._hybrid_rerank_node)
        workflow.add_node("generate_clips", self._generate_clips_node)
        workflow.add_node("generate_response", self._generate_response_node)

        workflow.set_entry_point("parse_query")
        
        workflow.add_edge("parse_query", "sql_search")
        workflow.add_edge("parse_query", "vector_search")
        
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