import chromadb
from pprint import pprint

client = chromadb.PersistentClient("database/chromadb")

collection = client.get_collection("image_embeddings")

data = collection.get(
    limit=5,
    include=["documents", "metadatas"]
)

print("Total:", collection.count())

for i in range(len(data["ids"])):
    print("=" * 80)
    print("ID:", data["ids"][i])
    print("\nDOCUMENT:")
    print(data["documents"][i])
    print("\nMETADATA:")
    pprint(data["metadatas"][i])