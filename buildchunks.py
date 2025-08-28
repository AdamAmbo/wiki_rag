import json
import pandas as pd
import os
from tqdm import tqdm

def extract_paragraphs(sections):
    if not isinstance(sections, list):
        return []
    paragraphs = []
    for section in sections:
        parts = section.get("has_parts", [])
        for part in parts:
            if part.get("type") == "paragraph":
                value = part.get("value", "").strip()
                if value:
                    paragraphs.append(value)
    return paragraphs

def chunk_text(text, chunk_size=250):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def prepare_data(jsonl_path):
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                title = data.get("name", "")
                paragraphs = extract_paragraphs(data.get("sections", []))
                full_text = " ".join(paragraphs)
                for chunk in chunk_text(full_text):
                    if chunk.strip():
                        chunks.append({"title": title, "text": chunk})
            except json.JSONDecodeError:
                continue
    return chunks

# === Process All Files ===
jsonl_dir = r"C:\Users\Adam\Desktop\intern\wiki_ai\enwiki_namespace_0"
jsonl_files = [os.path.join(jsonl_dir, f) for f in os.listdir(jsonl_dir) if f.endswith(".jsonl")]

all_chunks = []
for file in jsonl_files:
    print(f"📄 Processing {file} ...")
    all_chunks.extend(prepare_data(file))

# Save as a new file
df = pd.DataFrame(all_chunks)
df.to_csv("wiki_chunks2.csv", index=False)
print(f"\n✅ Total chunks extracted: {len(df)}")
print("✅ Saved to wiki_chunks2.csv.")
