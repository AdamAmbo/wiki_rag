"""
--------
This script runs the interactive Question-Answer (Q&A) loop for the Wikipedia
Retrieval-Augmented Generation (RAG) system.

Workflow:
1. Loads a preprocessed CSV of Wikipedia text chunks (wiki_chunks2.csv).
2. Loads a prebuilt FAISS index of dense embeddings (checkpointed for efficiency).
3. Uses a SentenceTransformer model (multi-qa-mpnet-base-dot-v1) to encode queries.
4. Retrieves the most relevant text chunks from the FAISS index.
5. Passes the retrieved context into a Flan-T5 generator model to produce an answer.
6. Runs an interactive loop where the user can ask questions until typing 'exit'.

This script ties together the retriever (FAISS + embeddings) and generator (Flan-T5)
to allow open-domain question answering over Wikipedia data.
"""

import os
import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from retrieve_generate import setup_generator, generate_answer

# === File Paths ===
csv_path = "wiki_chunks2.csv"  # CSV containing text chunks
index_path = "checkpoints/index_checkpoint.faiss"  # Prebuilt FAISS index
meta_path = "checkpoints/meta.pkl"  # Stores number of processed chunks

# === Load Existing Index and Metadata ===
print(" Loading finalized index and metadata...")
index = faiss.read_index(index_path)

# Load how many chunks were embedded (progress marker)
with open(meta_path, "rb") as f:
    chunk_limit = pickle.load(f)  # Only stores the last completed chunk
print(f" Using data up to chunk {chunk_limit}")

# === Load only the used chunks ===
df = pd.read_csv(csv_path, nrows=chunk_limit)
corpus = df["text"].astype(str).tolist()   # text chunks
titles = df["title"].astype(str).tolist()  # article titles

# === Load Retriever Model ===
retriever_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
print(" Retriever model loaded")

# === Setup Generator Model (Flan-T5) ===
generator = setup_generator()

# === Interactive Q&A Loop ===
while True:
    query = input("Ask a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    # Generate an answer using retrieval + generation
    answer = generate_answer(query, retriever_model, index, corpus, titles, generator)
    print("\n Query:", query)
    print(" Answer:", answer, "\n")
