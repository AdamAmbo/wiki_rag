import os
import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from retrieve_generate import setup_generator, generate_answer

# === File Paths ===
csv_path = "wiki_chunks2.csv"
index_path = "checkpoints/index_checkpoint.faiss"
meta_path = "checkpoints/meta.pkl"

# === Load Existing Index and Metadata ===
print(" Loading finalized index and metadata...")
index = faiss.read_index(index_path)

with open(meta_path, "rb") as f:
    chunk_limit = pickle.load(f)  # Only stores the last completed chunk
print(f" Using data up to chunk {chunk_limit}")

# === Load only used chunks ===
df = pd.read_csv(csv_path, nrows=chunk_limit)
corpus = df["text"].astype(str).tolist()
titles = df["title"].astype(str).tolist()

# === Load Retriever Model ===
retriever_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
print(" Retriever model loaded")

# === Setup Generator ===
generator = setup_generator()

# === Q&A Loop ===
while True:
    query = input("Ask a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    answer = generate_answer(query, retriever_model, index, corpus, titles, generator)
    print("\n Query:", query)
    print(" Answer:", answer, "\n")
