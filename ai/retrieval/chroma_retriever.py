import os
from typing import Dict, List

import chromadb
import torch
from transformers import CLIPModel, CLIPProcessor


class ChromaRetriever:
    """
    Retrieves semantically similar objects from ChromaDB
    using CLIP text embeddings.
    """

    def __init__(
        self,
        chroma_path: str,
        collection_name: str = "video_embeddings",
        clip_model: str = "openai/clip-vit-base-patch32",
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.client = chromadb.PersistentClient(path=chroma_path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        self.processor = CLIPProcessor.from_pretrained(clip_model)

        self.model = CLIPModel.from_pretrained(
            clip_model
        ).to(self.device)

        self.model.eval()

    def _encode_text(self, text: str):

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            embedding = self.model.get_text_features(**inputs)

            embedding = embedding / embedding.norm(
                dim=-1,
                keepdim=True
            )

        return embedding.squeeze().cpu().tolist()

    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:

        embedding = self._encode_text(query)

        print(f"Query: {query}")
        print(f"Collection Count: {self.collection.count()}")

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )

        output = []

        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for idx, distance, metadata in zip(
            ids,
            distances,
            metadatas
        ):

            output.append(
                {
                    "id": idx,
                    "similarity": 1.0 - float(distance),
                    "metadata": metadata,
                }
            )

        return output


if __name__ == "__main__":

    retriever = ChromaRetriever(
        chroma_path="../../database/chromadb"
    )

    query = "person wearing blue shirt"

    results = retriever.retrieve(
        query=query,
        top_k=5
    )

    print("=" * 80)
    print("Query :", query)
    print("Matches :", len(results))

    for result in results:
        print(result)