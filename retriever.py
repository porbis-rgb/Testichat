# retriever.py
import os
import heapq
import requests
import re
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# ---------------------------------------------------------
# 0. EMBEDDING MODEL
# ---------------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")
print(">>> USING RETRIEVER FILE:", __file__)

def embed(text: str) -> np.ndarray:
    return model.encode(text, convert_to_numpy=True)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if norm(a) == 0 or norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (norm(a) * norm(b)))


# ---------------------------------------------------------
# 1. BIBLE BOOK + CHAPTER DETECTION
# ---------------------------------------------------------

BOOK_KEYWORDS = {
    "proverbs": ["proverbs", "prov"],
    "ecclesiastes": ["ecclesiastes", "ecc"],
    "job": ["job"],
    "corinthians": ["corinthians", "cor"],
    "john": ["john"],
    "psalms": ["psalm", "psalms", "ps"],
    # add more as needed
}

def detect_bible_book(question: str):
    q = question.lower()
    for book, keys in BOOK_KEYWORDS.items():
        if any(k in q for k in keys):
            return book
    return None

def detect_chapter(question: str):
    q = question.lower()
    for token in q.split():
        if token.isdigit():
            return int(token)
        if ":" in token:
            try:
                return int(token.split(":")[0])
            except:
                pass
    return None


# ---------------------------------------------------------
# 2. LOCAL RETRIEVER (patched to allow non-Bible questions)
# ---------------------------------------------------------

LOCAL_DIR = "./data"

def load_local_file(path: str) -> str:
    full_path = os.path.join(LOCAL_DIR,path)
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def local_retrieve(question: str, min_score: float = 0.25) -> list:
    book = detect_bible_book(question)
    chapter = detect_chapter(question)

    q_vec = embed(question)
    heap = []

    for filename in os.listdir(LOCAL_DIR):
        print("LOCAL FILES:", os.listdir(LOCAL_DIR))
        print("Checking local file:", filename)
        if not filename.lower().endswith(".txt"):
            continue
        # Bible filtering ONLY if it's a Bible question
        if book:
            if book not in filename.lower():
                continue
        # Non-Bible question → allow all local files
        text = load_local_file(filename)
        if not text:
            continue
        doc_vec = embed(text)
        score = cosine(q_vec, doc_vec)
        print("Score for", filename, "=", score)
        # Soft penalty if chapter doesn't appear
        if book and chapter and f"{chapter}:" not in text:
            score *= 0.7

        if score < min_score:
            continue

        heapq.heappush(heap, (-score, filename, text))

    top = []
    for _ in range(min(3, len(heap))):
        score, filename, text = heapq.heappop(heap)
        top.append({
            "source": "local",
            "file": filename,
            "score": -score,
            "content": text
        })

    return top


# ---------------------------------------------------------
# 3. GOOGLE DRIVE RETRIEVER (patched to search Shared With Me)
# ---------------------------------------------------------

SERVICE_ACCOUNT_FILE = "service_account.json"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def build_drive_service():
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        print("Warning: Google Drive service account file not found.")
        return None
    creds = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)

def drive_search(question: str, min_score: float = 0.25) -> list:
    service = build_drive_service()
    if service is None:
        return []

    book = detect_bible_book(question)
    chapter = detect_chapter(question)

    try:
        results = service.files().list(
            q="mimeType='application/vnd.google-apps.document'",
            pageSize=50,
            fields="files(id, name)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            corpora="allDrives"
        ).execute()
    except Exception:
        return []

    files = results.get("files", [])
    docs = []

    for f in files:
        name = f["name"].lower()

        # Bible filtering
        if book and book not in name:
            continue

        try:
            content = service.files().export(
                fileId=f["id"],
                mimeType="text/plain"
            ).execute().decode("utf-8", errors="ignore")
        except Exception:
            continue

        docs.append({
            "id": f["id"],
            "name": f["name"],
            "content": content
        })

    if not docs:
        return []

    q_vec = embed(question)
    heap = []

    for d in docs:
        vec = embed(d["content"])
        score = cosine(q_vec, vec)

        if chapter and f"{chapter}:" not in d["content"]:
            score *= 0.7

        if score < min_score:
            continue

        heapq.heappush(heap, (-score, d))

    top = []
    for _ in range(min(3, len(heap))):
        score, d = heapq.heappop(heap)
        top.append({
            "source": "google_drive",
            "id": d["id"],
            "name": d["name"],
            "score": -score,
            "content": d["content"]
        })

    return top


# ---------------------------------------------------------
# 4. WEB SEARCH RETRIEVER (patched: HTML cleaned)
# ---------------------------------------------------------

WEB_SEARCH_API_KEY = "2763e007c4972a51b6678a7cc50561b0624f4ac49f7342752ba67cf4c8edf064"
WEB_SEARCH_ENDPOINT = "https://api.serpapi.com/search"

def clean_html(raw_html):
    raw_html = re.sub(r"<script.*?>.*?</script>", "", raw_html, flags=re.DOTALL)
    raw_html = re.sub(r"<style.*?>.*?</style>", "", raw_html, flags=re.DOTALL)
    raw_html = re.sub(r"<.*?>", " ", raw_html)
    raw_html = re.sub(r"\s+", " ", raw_html)
    return raw_html.strip()
def web_search(question: str, min_score: float = 0.25) -> list:
    print("Searching the web...")
    params = {"q": question, "api_key": WEB_SEARCH_API_KEY}
    print("before web try")
    try:
        resp = requests.get(WEB_SEARCH_ENDPOINT, params=params, timeout=10)
        print("[WEB] HTTP status:", resp.status_code)
        print("[WEB] Full response JSON:", resp.text[:500])
        data = resp.json()
        print("[WEB] Raw SerpAPI keys:", list(data.keys()))
        print("[WEB] Organic results count:", len(data.get("organic_results", [])))
    except Exception as e:
        print("[WEB] Exception:", type(e).__name__, str(e))
        return []
    print("\n[WEB] Query:", question)
    print("[WEB] Requesting:", WEB_SEARCH_ENDPOINT)

    organic = data.get("organic_results", [])
    pages = []
    print("after web try")
    for item in organic[:5]:
        url = item.get("link")
        print("[WEB] Found URL:", url)

        if not url:
            continue
        try:
            page = requests.get(url, timeout=10)
            print("[WEB] Fetched page:", url, "Status:", page.status_code)
            text = clean_html(page.text)
            print("[WEB] Cleaned text length:", len(text))
            print("[WEB] Sample cleaned text:", text[:200])
        except Exception:
            continue
        pages.append({"url": url, "content": text})

    if not pages:
        return []

    q_vec = embed(question)
    heap = []
    for p in pages:
        vec = embed(p["content"])
        score = cosine(q_vec, vec)
        print("[WEB] Score for", p["url"], "=", score)
        if score < min_score:
            print("[WEB] Discarded (below threshold):", score)
            continue
        heapq.heappush(heap, (-score, p))
    top = []
    for _ in range(min(3, len(heap))):
        score, p = heapq.heappop(heap)
        top.append({
            "source": "web",
            "url": p["url"],
            "score": -score,
            "content": p["content"]
        })

    return top


# ---------------------------------------------------------
# 5. HYBRID RETRIEVER
# ---------------------------------------------------------

def hybrid_retrieve(question: str, top_k: int = 3) -> list:
    results = []

    # Local
    results.extend(local_retrieve(question))

    # Drive
    results.extend(drive_search(question))
    #print("before searching web")
    # Web
    #results.extend(web_search(question))
    #print("after searching web")

    if not results:
        return []

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]

