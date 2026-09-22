from fastapi import FastAPI
from pydantic import BaseModel

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = FastAPI()

# -------------------
# Load PDF ONCE
# -------------------

reader = PdfReader("Enterprise-Document-Search/sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()

chunk_size = 500

chunks = []

for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i+chunk_size])

# -------------------
# Embeddings ONCE
# -------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS Ready")
print("Chunks:", len(chunks))


# -------------------
# Request Model
# -------------------

class Question(BaseModel):
    question: str


# -------------------
# Routes
# -------------------

@app.get("/")
def home():
    return {"message": "Enterprise Document Search API Running"}


@app.post("/ask")
def ask_question(data: Question):

    query = data.question

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distance, index_ids = index.search(query_embedding, 1)

    context = chunks[index_ids[0][0]]

    return {
        "question": query,
        "answer": context
    }