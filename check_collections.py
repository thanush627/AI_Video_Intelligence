import chromadb

client = chromadb.PersistentClient("database/chromadb")

collections = client.list_collections()

print("\nCollections:\n")

for collection in collections:
    print(collection.name)