from typing import Dict


class QueryPlanner:
    """
    Decides which retrieval engines should be used
    based on the parsed query.
    """

    def __init__(self):
        pass

    def plan(self, parsed_query: Dict) -> Dict:
        """
        Returns:
        {
            "use_sql": bool,
            "use_chromadb": bool,
            "strategy": str
        }
        """

        use_sql = False
        use_chromadb = False

        has_object = parsed_query.get("object") is not None
        has_color = parsed_query.get("color") is not None
        has_action = parsed_query.get("action") is not None
        has_attribute = parsed_query.get("attribute") is not None
        has_time = parsed_query.get("time_range") is not None

        # -------------------------
        # SQL Retrieval
        # -------------------------

        if has_object:
            use_sql = True

        if has_action:
            use_sql = True

        if has_attribute:
            use_sql = True

        if has_time:
            use_sql = True

        # -------------------------
        # ChromaDB Retrieval
        # -------------------------

        if has_color:
            use_chromadb = True

        if has_object and has_color:
            use_chromadb = True

        if not use_sql:
            use_chromadb = True

        # -------------------------
        # Retrieval Strategy
        # -------------------------

        if use_sql and use_chromadb:
            strategy = "hybrid"

        elif use_sql:
            strategy = "sql"

        else:
            strategy = "chromadb"

        return {
            "strategy": strategy,
            "use_sql": use_sql,
            "use_chromadb": use_chromadb
        }


if __name__ == "__main__":

    from query_parser import QueryParser

    parser = QueryParser()
    planner = QueryPlanner()

    queries = [
        "Find red car",
        "Show person",
        "Blue backpack",
        "Walking person",
        "Find truck between 09:00 and 10:00",
        "Person carrying backpack"
    ]

    for q in queries:

        parsed = parser.parse(q)

        print("=" * 70)
        print("Query :", q)
        print("Parsed :", parsed)
        print("Plan :", planner.plan(parsed))