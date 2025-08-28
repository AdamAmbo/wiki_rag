from transformers import pipeline

def setup_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

def retrieve(query, model, index, corpus, titles, k=10):
    q_embedding = model.encode([query])
    D, I = index.search(q_embedding, k)
    return [(titles[i], corpus[i]) for i in I[0]]

def generate_answer(query, retriever_model, index, corpus, titles, generator, top_k=10):
    docs = retrieve(query, retriever_model, index, corpus, titles, k=top_k)
    context = "\n".join([f"{title}: {chunk}" for title, chunk in docs])
    prompt = f"Answer the question based on the context:\n\n{context}\n\nQuestion: {query}"
    return generator(prompt, max_new_tokens=128, do_sample=False)[0]['generated_text']
