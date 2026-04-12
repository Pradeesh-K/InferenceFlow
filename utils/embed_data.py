from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# Information as arrays 
from utils.privacy import privacy
from utils.refund_policy import refunds

# Storage location folder
save_to = "vector_store"

# Embedding usage
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", dimensions=768)
# #Embed single
# single_vector = embeddings.embed_query("hello")
# print(str(single_vector)[:100])  # Show the first 100 characters of the vector

# # embed bulk
# two_vectors = embeddings.embed_documents(["hello", "prad"])
# for vector in two_vectors:
#     print(str(vector)[:100])  # Show the first 100 characters of the vector

import os
import faiss
import numpy as np

os.makedirs(save_to, exist_ok=True)

# --- Chunking function ---
def chunk_text(text, chunk_size=100, overlap=10):
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    
    return chunks

# --- Process data ---
def process_data(data_array):
    all_chunks = []

    for i in range(0, len(data_array), 2):
        heading = data_array[i]
        content = data_array[i+1]

        chunks = chunk_text(content)

        for chunk in chunks:
            combined = f"{heading}: {chunk}"
            all_chunks.append(combined)

    return all_chunks

# Combine your datasets
all_data = privacy + refunds

# Generate chunks
texts = process_data(all_data)

# --- Embed ---
vectors = embeddings.embed_documents(texts)
vectors_np = np.array(vectors).astype("float32")

# --- FAISS index ---
dim = vectors_np.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(vectors_np)

# --- Save ---
faiss.write_index(index, os.path.join(save_to, "faiss_index.bin"))

# Save texts for retrieval mapping
with open(os.path.join(save_to, "chunks.txt"), "w") as f:
    for t in texts:
        f.write(t + "\n")

print(f"✅ Stored {len(texts)} chunks in FAISS + saved to '{save_to}'")