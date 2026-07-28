from typing import Dict, List


class HybridRetriever:
    """
    Merges SQL and ChromaDB retrieval results into
    a unified candidate list.
    """

    def __init__(self):
        pass

    def retrieve(
        self,
        sql_results: List[Dict],
        chroma_results: List[Dict]
    ) -> List[Dict]:

        merged = {}

        # -----------------------------
        # Add SQL Results
        # -----------------------------
        for result in sql_results:

            event_id = (
                result.get("event_id")
                or result.get("track_id")
                or result.get("id")
            )

            if event_id is None:
                continue

            merged[event_id] = {
                "event_id": event_id,
                "sql_score": 1.0,
                "vector_score": 0.0,
                "final_score": 1.0,
                "metadata": result,
                "sources": ["sql"]
            }

        # -----------------------------
        # Merge Chroma Results
        # -----------------------------
        for result in chroma_results:

            metadata = result.get("metadata", {})

            event_id = (
                metadata.get("event_id")
                or metadata.get("track_id")
                or result.get("id")
            )

            similarity = float(result.get("similarity", 0.0))

            if event_id in merged:

                merged[event_id]["vector_score"] = similarity

                merged[event_id]["final_score"] = (
                    0.6 * merged[event_id]["sql_score"] +
                    0.4 * similarity
                )

                if "chromadb" not in merged[event_id]["sources"]:
                    merged[event_id]["sources"].append("chromadb")

            else:

                merged[event_id] = {
                    "event_id": event_id,
                    "sql_score": 0.0,
                    "vector_score": similarity,
                    "final_score": similarity,
                    "metadata": metadata,
                    "sources": ["chromadb"]
                }

        # -----------------------------
        # Sort
        # -----------------------------
        results = list(merged.values())

        results.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        return results


if __name__ == "__main__":

    sql_results = [
        {
            "event_id": 10,
            "objects": ["person"],
            "start_time": "00:00:15",
            "end_time": "00:00:22"
        },
        {
            "event_id": 11,
            "objects": ["car"]
        }
    ]

    chroma_results = [
        {
            "id": 10,
            "similarity": 0.93,
            "metadata": {
                "event_id": 10
            }
        },
        {
            "id": 15,
            "similarity": 0.88,
            "metadata": {
                "event_id": 15
            }
        }
    ]

    retriever = HybridRetriever()

    results = retriever.retrieve(
        sql_results,
        chroma_results
    )

    print("=" * 80)

    for result in results:
        print(result)