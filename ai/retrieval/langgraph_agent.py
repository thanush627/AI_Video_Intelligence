import time
from typing import Dict, List

from langgraph.graph import END, StateGraph

from ai.retrieval.query_parser import QueryParser
from ai.retrieval.planner import QueryPlanner
from ai.retrieval.sql_retriever import SQLRetriever
from ai.retrieval.chroma_retriever import ChromaRetriever
from ai.retrieval.hybrid_retriever import HybridRetriever
from ai.retrieval.reranker import ReRanker
from ai.retrieval.clip_selector import ClipSelector
from ai.retrieval.response_generator import ResponseGenerator


from typing import TypedDict, Optional, List, Dict, Any


class RetrievalState(TypedDict, total=False):
    query: str
    parsed_query: Dict[str, Any]
    plan: Dict[str, Any]
    sql_results: List[Dict[str, Any]]
    vector_results: List[Dict[str, Any]]
    merged_results: List[Dict[str, Any]]
    ranked_results: List[Dict[str, Any]]
    clips: List[Dict[str, Any]]
    response: Dict[str, Any]
    sql_time: float
    vector_time: float
    total_time: float


class LangGraphRetrievalAgent:

    def __init__(
        self,
        event_database_path: str,
        chroma_path: str,
        collection_name: str = "video_embeddings",
        top_k: int = 10,
    ):

        self.parser = QueryParser()

        self.planner = QueryPlanner()

        self.sql = SQLRetriever(
            event_database_path=event_database_path
        )

        self.chroma = ChromaRetriever(
            chroma_path=chroma_path,
            collection_name=collection_name,
        )

        self.hybrid = HybridRetriever()

        self.reranker = ReRanker()

        self.clip_selector = ClipSelector(top_k=top_k)

        self.response_generator = ResponseGenerator()

        self.graph = self._build_graph()

    ####################################################################
    # Nodes
    ####################################################################

    def parse_query(self, state: RetrievalState):

        parsed = self.parser.parse(state["query"])

        state["parsed_query"] = parsed

        return state

    def create_plan(self, state: RetrievalState):

        plan = self.planner.plan(state["parsed_query"])

        state["plan"] = plan

        return state

    def retrieve_sql(self, state: RetrievalState):

        if not state["plan"]["use_sql"]:

            state["sql_results"] = []

            return state

        start = time.perf_counter()

        state["sql_results"] = self.sql.retrieve(
            state["parsed_query"]
        )

        state["sql_time"] = (
            time.perf_counter() - start
        ) * 1000

        return state

    def retrieve_chromadb(self, state: RetrievalState):

        if not state["plan"]["use_chromadb"]:

            state["vector_results"] = []

            return state

        start = time.perf_counter()

        state["vector_results"] = self.chroma.retrieve(
            query=state["query"]
        )

        state["vector_time"] = (
            time.perf_counter() - start
        ) * 1000

        return state

    def hybrid_merge(self, state: RetrievalState):

        state["merged_results"] = self.hybrid.retrieve(
            sql_results=state.get("sql_results", []),
            chroma_results=state.get("vector_results", [])
        )

        return state

    def rerank(self, state: RetrievalState):

        state["ranked_results"] = self.reranker.rerank(
            state["merged_results"]
        )

        return state

    def select_clips(self, state: RetrievalState):

        state["clips"] = self.clip_selector.select(
            state["ranked_results"]
        )

        return state

    def generate_response(self, state: RetrievalState):

        statistics = {
            "query_time_ms": 0.0,
            "sql_time_ms": round(state.get("sql_time", 0.0), 3),
            "vector_time_ms": round(state.get("vector_time", 0.0), 3),
            "total_time_ms": 0.0,
            "results_returned": len(state.get("clips", []))
        }

        state["response"] = self.response_generator.generate(
            query=state["query"],
            parsed_query=state["parsed_query"],
            retrieval_plan=state["plan"],
            clips=state["clips"],
            statistics=statistics,
        )

        return state

    ####################################################################
    # Graph
    ####################################################################

    def _build_graph(self):

        graph = StateGraph(RetrievalState)

        graph.add_node("parse", self.parse_query)

        graph.add_node("plan", self.create_plan)

        graph.add_node("sql", self.retrieve_sql)

        graph.add_node("vector", self.retrieve_chromadb)

        graph.add_node("merge", self.hybrid_merge)

        graph.add_node("rerank", self.rerank)

        graph.add_node("clips", self.select_clips)

        graph.add_node("response", self.generate_response)

        graph.set_entry_point("parse")

        graph.add_edge("parse", "plan")

        graph.add_edge("plan", "sql")

        graph.add_edge("sql", "vector")

        graph.add_edge("vector", "merge")

        graph.add_edge("merge", "rerank")

        graph.add_edge("rerank", "clips")

        graph.add_edge("clips", "response")

        graph.add_edge("response", END)

        return graph.compile()

    ####################################################################
    # Run
    ####################################################################

    def run(self, query: str):

        start = time.perf_counter()

        state = {
            "query": query
        }

        result = self.graph.invoke(state)

        total_time = (time.perf_counter() - start) * 1000

        result["total_time"] = total_time

        if "response" in result:
            result["response"]["statistics"]["query_time_ms"] = round(total_time, 3)
            result["response"]["statistics"]["total_time_ms"] = round(total_time,3)

        return result["response"]


if __name__ == "__main__":

    agent = LangGraphRetrievalAgent(
        event_database_path="../../outputs/phase6/event_database.json",
        chroma_path="../../database/chromadb",
    )

    response = agent.run(
        "Show me person wearing blue backpack"
    )

    print(response)