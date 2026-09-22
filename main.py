from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
reader = PdfReader("Enterprise-Document-Search/sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()
chunk_size = 500

chunks = []

for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i+chunk_size])

print("Number of chunks:", len(chunks))

print(chunks[0])
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Vectors stored:", index.ntotal)
print("Number of embeddings:", len(embeddings))
print("Embedding dimension:", len(embeddings[0]))
query = input("Ask a question: ")

query_embedding = model.encode([query])

query_embedding = np.array(query_embedding).astype("float32")

distance, index_ids = index.search(query_embedding, 1)

context = chunks[index_ids[0][0]]

print("\nRetrieved Context:\n")
print(context)

print("\nAnswer:")
print(context)
