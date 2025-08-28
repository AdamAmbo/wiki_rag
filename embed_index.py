import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

def embed_and_index(corpus, batch_size=32, checkpoint_every=10000, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    dim = model.get_sentence_embedding_dimension()
    index_path = os.path.join(checkpoint_dir, "index_checkpoint.faiss")
    meta_path = os.path.join(checkpoint_dir, "meta.pkl")

    # Create or resume index
    if os.path.exists(index_path) and os.path.exists(meta_path):
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            start = pickle.load(f)
        print(f"🧩 Resuming from checkpoint at chunk {start}")
    else:
        index = faiss.IndexFlatL2(dim)
        start = 0

    # Start embedding
    for i in tqdm(range(start, len(corpus), batch_size), desc="Embedding", unit="batch"):
        batch = corpus[i:i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        index.add(embeddings)

        # Save every checkpoint
        if (i // batch_size) % (checkpoint_every // batch_size) == 0:
            faiss.write_index(index, index_path)
            with open(meta_path, "wb") as f:
                pickle.dump(i + batch_size, f)

    print("✅ Embedding complete.")
    return model, index, None
