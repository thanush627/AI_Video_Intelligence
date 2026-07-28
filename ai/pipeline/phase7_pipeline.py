import json
import os
import time

from ai.retrieval.langgraph_agent import LangGraphRetrievalAgent
from ai.retrieval.result_formatter import ResultFormatter


class Phase7Pipeline:

    def __init__(
        self,
        event_database_path,
        chroma_path,
        output_dir,
        collection_name="video_embeddings",
        top_k=10,
    ):

        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.agent = LangGraphRetrievalAgent(
            event_database_path=event_database_path,
            chroma_path=chroma_path,
            collection_name=collection_name,
            top_k=top_k,
        )

        self.formatter = ResultFormatter()

    def run(self, query):

        total_start = time.perf_counter()

        response = self.agent.run(query)

        retrieval_results_file = os.path.join(
            self.output_dir,
            "retrieval_results.json"
        )

        ranked_results_file = os.path.join(
            self.output_dir,
            "ranked_results.json"
        )

        statistics_file = os.path.join(
            self.output_dir,
            "retrieval_statistics.json"
        )

        with open(
            retrieval_results_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                response,
                f,
                indent=4,
                ensure_ascii=False
            )

        ranked_results = response.get("results", [])

        ranked_results.sort(
            key=lambda x: x.get("ranking_score", 0),
            reverse=True
        )

        with open(
            ranked_results_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                ranked_results,
                f,
                indent=4,
                ensure_ascii=False
            )

        total_time = (
            time.perf_counter() - total_start
        ) * 1000

        statistics = response.get("statistics", {})

        statistics["pipeline_time_ms"] = round(
            total_time,
            3
        )

        statistics["results_returned"] = len(
            ranked_results
        )

        with open(
            statistics_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                statistics,
                f,
                indent=4
            )

        print(

            self.formatter.format_console(

                query=query,

                results=ranked_results,

                statistics=statistics,
            )

        )

        return response


if __name__ == "__main__":

    pipeline = Phase7Pipeline(

        event_database_path="outputs/phase6/event_database.json",

        chroma_path="database/chromadb",

        output_dir="outputs/phase7",

        collection_name="video_embeddings",

        top_k=10,
    )

    while True:

        print("\n" + "=" * 80)

        query = input("Enter Query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        pipeline.run(query)