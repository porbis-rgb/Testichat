# summarizerHO.py
import re
from transformers import pipeline
from retriever import hybrid_retrieve
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm

# ---------------------------------------------------------
# 0. FAST SUMMARIZER + EMBEDDING MODEL
# ---------------------------------------------------------

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str) -> np.ndarray:
    return embedder.encode(text, convert_to_numpy=True)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if norm(a) == 0 or norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (norm(a) * norm(b)))


# ---------------------------------------------------------
# 1. PARAGRAPH + VERSE SPLITTING
# ---------------------------------------------------------

def split_into_chunks(text: str):
    """
    Splits text into paragraphs AND detects Bible verse-like lines.
    """
    chunks = []

    # Split by paragraphs
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue

        # Detect verse-like patterns (e.g., Prov. 11:13)
        if re.search(r"\b\d{1,3}:\d{1,3}\b", block):
            chunks.append(block)
        else:
            chunks.append(block)

    return chunks


# ---------------------------------------------------------
# 2. SEMANTIC RANKING OF CHUNKS
# ---------------------------------------------------------

def rank_chunks(question: str, chunks: list, top_k: int = 8):
    q_vec = embed(question)
    scored = []

    for ch in chunks:
        vec = embed(ch)
        score = cosine(q_vec, vec)
        scored.append((score, ch))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for s, c in scored[:top_k] if s > 0.15]


# ---------------------------------------------------------
# 3. SUMMARIZATION PIPELINE
# ---------------------------------------------------------

def summarize_chunks(chunks: list):
    combined = " ".join(chunks)
    if len(combined) < 50:
        return combined

    result = summarizer(
        combined,
        max_length=180,
        min_length=50,
        do_sample=False,
        truncation=True
    )
    return result[0]["summary_text"]
# ---------------------------------------------------------
# VERSE EXTRACTION MODE
# ---------------------------------------------------------

VERSE_PATTERN = re.compile(r"\b([A-Za-z]+\s*\d{1,3}:\d{1,3})\b")

def extract_exact_verses(chunks):
    """
    Returns exact verse lines (1–2 lines max) from chunks.
    """
    verses = []

    for ch in chunks:
        # Look for verse-like patterns
        if re.search(r"\b\d{1,3}:\d{1,3}\b", ch):
            # Keep only the first 1–2 lines to stay within copyright rules
            lines = ch.strip().split("\n")
            verses.append("\n".join(lines[:2]))

    return verses[:3]  # return up to 3 verse snippets


# ---------------------------------------------------------
# 4. MAIN QA PIPELINE
# ---------------------------------------------------------

def answer_question(question: str):
    # Retrieve documents
    results = hybrid_retrieve(question, top_k=3)

    if not results:
        return "No relevant documents found."

    # Extract text from retrieved docs
    all_text = " ".join(r["content"] for r in results)

    # Split into chunks
    chunks = split_into_chunks(all_text)

    verses = extract_exact_verses(chunks)
    if verses:
        return "\n\n".join(verses)

    # Rank chunks by semantic similarity
    top_chunks = rank_chunks(question, chunks)

    if not top_chunks:
        return "No relevant passages found."

    # Summarize the most relevant chunks
    final_answer = summarize_chunks(top_chunks)
    return final_answer


# ---------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------

if __name__ == "__main__":
    question = input("Ask your question: ").strip()

    print("\n--- Retrieved Context Preview ---")
    results = hybrid_retrieve(question, top_k=3)
    for r in results:
        print(f"Source: {r['source']}")
        if r['source'] == "google_drive":
            print(f"Drive file: {r['name']} (id: {r['id']})")
        if r['source'] == "local":
            print(f"File: {r.get('file')}")
        print(f"Score: {r['score']:.4f}")
        print()

    print("\n--- Final Answer ---\n")
    print(answer_question(question))

