import os, json, re
import httpx
import faiss
import numpy as np
from typing import List, Tuple, Optional

GEMINI_KEY   = os.getenv("GOOGLE_GENAI_API_KEY")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "text-embedding-004")

# --- utils ---
def normalize_text(txt: str) -> str:
    return re.sub(r"\s+", " ", txt).strip()

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Découpe un texte long en morceaux qui se chevauchent légèrement."""
    text = normalize_text(text)
    chunks = []
    i = 0
    while i < len(text):
        end = min(len(text), i + chunk_size)
        chunk = text[i:end]
        chunks.append(chunk)
        if end == len(text):
            break
        i = end - overlap
        if i < 0:
            i = 0
    return chunks

class SimpleRAG:
    """
    RAG minimal avec FAISS + embeddings Gemini + persistance disque.
    - add_docs(texts): ajoute des chunks
    - search(query, k): renvoie [(texte, distance, rank)]
    - save/load: persiste l'index et les docs
    """
    def __init__(self, dim: Optional[int] = None):
        self.dim   = dim
        self.index = None
        self.docs: List[str] = []

    # ---------- Embeddings ----------
    async def _embed_gemini(self, text: str, is_query: bool) -> np.ndarray:
        if not GEMINI_KEY:
            raise RuntimeError("GOOGLE_GENAI_API_KEY manquant (.env)")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent?key={GEMINI_KEY}"
        body = {
            "model": EMBED_MODEL,
            "taskType": "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT",
            "content": {"parts": [{"text": text}]}
        }
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, json=body)
            r.raise_for_status()
            data = r.json()

        # Formats possibles selon version d'API
        vec = None
        if "embedding" in data and "values" in data["embedding"]:
            vec = data["embedding"]["values"]
        elif "embeddings" in data and data["embeddings"]:
            vec = data["embeddings"][0].get("values")
        else:
            raise RuntimeError(f"Réponse embeddings inattendue: {data}")

        arr = np.array(vec, dtype="float32")
        if self.dim is None:
            self.dim = arr.shape[0]
            self.index = faiss.IndexFlatL2(self.dim)
        return arr

    # ---------- CRUD ----------
    async def add_docs(self, texts: List[str]):
        if not texts:
            return
        vecs = [await self._embed_gemini(t, is_query=False) for t in texts]
        mat = np.stack(vecs)
        if self.index is None:
            self.dim = mat.shape[1]
            self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(mat)
        self.docs.extend(texts)

    async def search(self, query: str, k: int = 3) -> List[Tuple[str, float, int]]:
        if self.index is None or not self.docs:
            return []
        qv = await self._embed_gemini(query, is_query=True)
        D, I = self.index.search(qv.reshape(1, -1), min(k, len(self.docs)))
        out = []
        for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
            if idx == -1:
                continue
            out.append((self.docs[idx], float(dist), rank))
        return out

    # ---------- Persistance ----------
    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "docs.json"), "w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False, indent=2)

    def load(self, folder: str):
        idx_path  = os.path.join(folder, "index.faiss")
        docs_path = os.path.join(folder, "docs.json")
        if os.path.exists(idx_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(idx_path)
            with open(docs_path, "r", encoding="utf-8") as f:
                self.docs = json.load(f)
            self.dim = self.index.d
