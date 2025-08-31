import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

# Function to embed corpus into vectors and build a FAISS index with checkpointing
def embed_and_index(corpus, batch_size=32, checkpoint_every=10000, checkpoint_dir="checkpoints"):
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load pretrained embedding model
    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    dim = model.get_sentence_embedding_dimension()

    # Define paths for index and metadata (progress tracking)
    index_path = os.path.join(checkpoint_dir, "index_checkpoint.faiss")
    meta_path = os.path.join(checkpoint_dir, "meta.pkl")

    # Resume from checkpoint if available
    if os.path.exists(index_path) and os.path.exists(meta_path):
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            start = pickle.load(f)  # where to resume from
        print(f" Resuming from checkpoint at chunk {start}")
    else:
        index = faiss.IndexFlatL2(dim)  # Flat index with L2 distance
        start = 0

    # Loop through corpus in batches
    for i in tqdm(range(start, len(corpus), batch_size), desc="Embedding", unit="batch"):
        batch = corpus[i:i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        index.add(embeddings)  # Add embeddings to FAISS index

        # Save progress periodically
        if (i // batch_size) % (checkpoint_every // batch_size) == 0:
            faiss.write_index(index, index_path)
            with open(meta_path, "wb") as f:
                pickle.dump(i + batch_size, f)

    print(" Embedding complete.")
    return model, index, None
