from pathlib import Path
import json

import numpy as np
import torch
import open_clip


class EventEmbedder:

    def __init__(
        self,
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=None,
        batch_size=32,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.batch_size = batch_size

        print(f"Loading CLIP model: {model_name}")
        print(f"Device: {self.device}")

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=self.device,
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.eval()

    def load_retrieval_documents(self, input_file):
        input_file = Path(input_file)

        if not input_file.exists():
            raise FileNotFoundError(
                f"Retrieval documents file not found: {input_file}"
            )

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            documents = (
                data.get("documents")
                or data.get("retrieval_documents")
                or data.get("events")
                or []
            )
        elif isinstance(data, list):
            documents = data
        else:
            documents = []

        if not documents:
            raise ValueError(
                "No retrieval documents found in the input file."
            )

        return documents

    @staticmethod
    def extract_text(document):
        text = (
            document.get("text")
            or document.get("document")
            or document.get("description")
            or document.get("event_text")
        )

        if not text:
            raise ValueError(
                f"No text field found for document: {document}"
            )

        return str(text).strip()

    def encode_texts(self, texts):
        all_embeddings = []

        with torch.no_grad():

            for start in range(0, len(texts), self.batch_size):

                batch_texts = texts[
                    start:start + self.batch_size
                ]

                tokens = self.tokenizer(batch_texts).to(self.device)

                embeddings = self.model.encode_text(tokens)

                embeddings = embeddings / embeddings.norm(
                    dim=-1,
                    keepdim=True,
                )

                embeddings = embeddings.cpu().numpy().astype(
                    np.float32
                )

                all_embeddings.append(embeddings)

                processed = min(
                    start + self.batch_size,
                    len(texts),
                )

                print(
                    f"Embedded {processed}/{len(texts)} documents"
                )

        return np.vstack(all_embeddings)

    def generate_embeddings(
        self,
        input_file,
        output_directory,
    ):
        output_directory = Path(output_directory)
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        documents = self.load_retrieval_documents(
            input_file
        )

        texts = [
            self.extract_text(document)
            for document in documents
        ]

        print(
            f"\nGenerating embeddings for "
            f"{len(texts)} retrieval documents...\n"
        )

        embeddings = self.encode_texts(texts)

        embeddings_file = (
            output_directory / "event_embeddings.npy"
        )

        metadata_file = (
            output_directory
            / "event_embedding_metadata.json"
        )

        np.save(
            embeddings_file,
            embeddings,
        )

        metadata = {
            "model_name": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
            "device": self.device,
            "document_count": len(documents),
            "embedding_count": int(
                embeddings.shape[0]
            ),
            "embedding_dimension": int(
                embeddings.shape[1]
            ),
            "normalized": True,
            "documents": [],
        }

        for index, document in enumerate(documents):

            metadata["documents"].append(
                {
                    "embedding_index": index,
                    "document_id": (
                        document.get("document_id")
                        or document.get("event_id")
                        or f"document_{index:06d}"
                    ),
                    "event_id": document.get("event_id"),
                    "text": texts[index],
                    "metadata": document.get(
                        "metadata",
                        {},
                    ),
                }
            )

        with open(
            metadata_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return {
            "documents": documents,
            "texts": texts,
            "embeddings": embeddings,
            "embeddings_file": str(
                embeddings_file.resolve()
            ),
            "metadata_file": str(
                metadata_file.resolve()
            ),
        }