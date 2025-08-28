
# Wikipedia Q&A System (RAG)

This project answers questions using a **Retrieval-Augmented Generation (RAG)** approach with Wikipedia data. It uses **SentenceTransformers** for embeddings, **FAISS** for efficient similarity search, and **Flan-T5** for natural-language answer generation.

---

## How It Works
- **buildchunks.py** → processes Wikipedia `.jsonl` files into text chunks and saves them as `wiki_chunks2.csv`.  
- **embed_index.py** → embeds the chunks, builds a FAISS index, and saves progress with checkpointing.  
- **retrieve_generate.py** → retrieves the most relevant chunks and generates answers using Flan-T5.  
- **main.py** → runs the interactive Q&A loop in the terminal.  

---

## Dataset
Download the dataset from Kaggle:  
 https://www.kaggle.com/datasets/wikimedia-foundation/wikipedia-structured-contents/data  

After downloading, extract the `.jsonl` files into a folder (e.g., `data/jsonl`) and point `buildchunks.py` to that folder.

---

## Setup
Install the required Python packages:

```bash
pip install pandas faiss-cpu sentence-transformers transformers tqdm
````

Prepare the dataset:

```bash
python buildchunks.py data/jsonl
```

This will create `wiki_chunks2.csv`.

---

## Training / Indexing

Run the embedding and indexing script:

```bash
python embed_index.py
```

This will:

* Embed the text chunks.
* Build and save a FAISS index in `checkpoints/index_checkpoint.faiss`.
* Save checkpoint metadata in `checkpoints/meta.pkl`.

---

## Prediction

Run the interactive Q\&A system:

```bash
python main.py
```

Then type any question.

**Example:**

```
Ask a question (or type 'exit'): Who developed the theory of relativity?

Query: Who developed the theory of relativity?
Answer: Albert Einstein developed the theory of relativity, first publishing the special theory in 1905.
```

Type `exit` to quit.

---

## Project Structure

```

├── README.md
├── main.py              # interactive Q&A
├── buildchunks.py       # preprocess Wikipedia JSONL → CSV
├── embed_index.py       # embed + index with FAISS
├── retrieve_generate.py # retrieval + generation logic
├── wiki_chunks2.csv     # generated dataset (not in repo)
└── checkpoints/         # FAISS index + metadata (not in repo)
```

---

## Notes

* The Wikipedia `.jsonl` files and large generated datasets are **not included** in this repository.
* Checkpoints and trained indexes are stored in `/checkpoints`.


---

## License

This project is released under the **MIT License**.

