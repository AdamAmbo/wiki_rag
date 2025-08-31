from transformers import pipeline

# Setup a text generation pipeline using Flan-T5
def setup_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

# Retrieve top-k most relevant chunks from FAISS index
def retrieve(query, model, index, corpus, titles, k=10):
    q_embedding = model.encode([query])  # Encode query into embedding
    D, I = index.search(q_embedding, k)  # Search FAISS for nearest neighbors
    return [(titles[i], corpus[i]) for i in I[0]]

# Generate an answer using retrieved context + Flan-T5
def generate_answer(query, retriever_model, index, corpus, titles, generator, top_k=10):
    # Get top documents relevant to query
    docs = retrieve(query, retriever_model, index, corpus, titles, k=top_k)
    # Build context string with titles and chunks
    context = "\n".join([f"{title}: {chunk}" for title, chunk in docs])
    # Build prompt for the generator model
    prompt = f"Answer the question based on the context:\n\n{context}\n\nQuestion: {query}"
    # Generate answer without sampling (deterministic output)
    return generator(prompt, max_new_tokens=128, do_sample=False)[0]['generated_text']
