import os, json, time
from typing import List
from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
import httpx
import re
import shutil
import time
from collections import defaultdict


# ---- Chargement .env ----
ENV_PATH = find_dotenv()
load_dotenv(dotenv_path=ENV_PATH, override=True)

PROVIDER = os.getenv("PROVIDER", "openai").lower()   # openai | gemini | claude
MODEL    = os.getenv("MODEL", "gpt-4o-mini")

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
GOOGLE_GENAI_KEY  = os.getenv("GOOGLE_GENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# RAG (embeddings Gemini + FAISS + chunking + persistance)
from rag import SimpleRAG, chunk_text
STORE_DIR = "store"
rag_engine = SimpleRAG()

app = FastAPI(title="AI/ML Applied Starter (multi-provider + RAG)", version="0.3.0")

@app.middleware("http")
async def metrics_middleware(request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    dt_ms = (time.perf_counter() - t0) * 1000
    METRICS["requests_total"] += 1
    METRICS["by_path"][request.url.path] += 1
    METRICS["latencies_ms"].append(dt_ms)
    if len(METRICS["latencies_ms"]) > 1000:  # borne la taille
        METRICS["latencies_ms"] = METRICS["latencies_ms"][-500:]
    return response

@app.get("/metrics")
def metrics():
    lat = METRICS["latencies_ms"]
    p50 = sorted(lat)[int(0.50*(len(lat)-1))] if lat else 0.0
    p90 = sorted(lat)[int(0.90*(len(lat)-1))] if lat else 0.0
    p99 = sorted(lat)[int(0.99*(len(lat)-1))] if lat else 0.0
    uptime_s = int(time.time() - METRICS["start_ts"])
    return {
        "uptime_s": uptime_s,
        "requests_total": METRICS["requests_total"],
        "by_path": dict(METRICS["by_path"]),
        "latency_ms": {"p50": round(p50,2), "p90": round(p90,2), "p99": round(p99,2)}
    }


# ------- Schémas simples -------
class ChatIn(BaseModel):
    message: str

class ChatOut(BaseModel):
    reply: str
    provider: str
    model: str

# ------- Hooks de cycle de vie -------
@app.on_event("startup")
async def _startup():
    print(">>> FastAPI démarré")
    print(f">>> CWD: {os.getcwd()}")
    print(f">>> __file__: {__file__}")
    print(f">>> .env found: {bool(ENV_PATH)} at {ENV_PATH}")
    try:
        rag_engine.load(STORE_DIR)
        print(f">>> RAG chargé ({len(rag_engine.docs)} chunks)")
    except Exception as e:
        print(f">>> RAG load skipped: {e}")

@app.on_event("shutdown")
async def _shutdown():
    try:
        rag_engine.save(STORE_DIR)
        print(">>> RAG sauvegardé")
    except Exception as e:
        print(f">>> RAG save failed: {e}")

# ------- Endpoints utilitaires -------
@app.get("/health")
def health():
    return {"status": "ok", "provider": PROVIDER, "model": MODEL}

@app.get("/debug/env")
def debug_env():
    return {
        "cwd": os.getcwd(),
        "env_path_found": bool(ENV_PATH),
        "env_path": ENV_PATH or None,
        "provider": PROVIDER,
        "model": MODEL,
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_google_key": bool(GOOGLE_GENAI_KEY),
        "has_anthropic_key": bool(ANTHROPIC_API_KEY),
    }

@app.get("/debug/ping")
async def debug_ping():
    try:
        if PROVIDER == "gemini":
            if not GOOGLE_GENAI_KEY:
                raise HTTPException(400, "GOOGLE_GENAI_API_KEY manquant dans .env")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GOOGLE_GENAI_KEY}"
            body = {"contents":[{"parts":[{"text":"ping"}]}]}
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(url, json=body)
                r.raise_for_status()
            data = r.json()
            sample = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"ok": True, "provider": "gemini", "model": MODEL, "sample": sample}

        elif PROVIDER == "claude":
            if not ANTHROPIC_API_KEY:
                raise HTTPException(400, "ANTHROPIC_API_KEY manquant dans .env")
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {"model": MODEL, "max_tokens": 16, "messages":[{"role":"user","content":"ping"}]}
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(url, headers=headers, json=body)
                r.raise_for_status()
            data = r.json()
            sample = data["content"][0]["text"]
            return {"ok": True, "provider": "claude", "model": MODEL, "sample": sample}

        else:
            if not OPENAI_API_KEY:
                raise HTTPException(400, "OPENAI_API_KEY manquant dans .env")
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"}
            body = {"model": MODEL, "messages":[{"role":"user","content":"ping"}], "max_tokens":8, "temperature":0}
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(url, headers=headers, json=body)
                r.raise_for_status()
            data = r.json()
            sample = data["choices"][0]["message"]["content"]
            return {"ok": True, "provider": "openai", "model": MODEL, "sample": sample}

    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"{PROVIDER} API error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        raise HTTPException(500, str(e))

# ------- Client LLM générique -------
async def llm_chat(message: str) -> str:
    try:
        if PROVIDER == "gemini":
            if not GOOGLE_GENAI_KEY:
                raise HTTPException(400, "GOOGLE_GENAI_API_KEY manquant (.env).")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GOOGLE_GENAI_KEY}"
            body = {"contents":[{"parts":[{"text": message}]}]}
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(url, json=body)
                r.raise_for_status()
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        elif PROVIDER == "claude":
            if not ANTHROPIC_API_KEY:
                raise HTTPException(400, "ANTHROPIC_API_KEY manquant (.env).")
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {"model": MODEL, "max_tokens": 300, "messages":[{"role":"user","content": message}]}
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(url, headers=headers, json=body)
                r.raise_for_status()
            data = r.json()
            return data["content"][0]["text"].strip()

        else:
            if not OPENAI_API_KEY:
                raise HTTPException(400, "OPENAI_API_KEY manquant (.env).")
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"}
            body = {
                "model": MODEL,
                "messages":[
                    {"role":"system","content":"Tu es un assistant concis et utile."},
                    {"role":"user","content": message}
                ],
                "temperature": 0.7,
                "max_tokens": 300
            }
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(url, headers=headers, json=body)
                r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()

    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"{PROVIDER} API error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        raise HTTPException(500, str(e))

# ------- Endpoints Chat -------
@app.post("/chat", response_model=ChatOut)
async def chat(payload: ChatIn):
    reply = await llm_chat(payload.message)
    return ChatOut(reply=reply, provider=PROVIDER, model=MODEL)

# Bypass JSON (pratique pour tester depuis le navigateur)
@app.get("/chat_qs")
async def chat_qs(message: str = Query(..., example="Bonjour ! Donne 1 idée de mini-projet IA.")):
    reply = await llm_chat(message)
    return {"reply": reply, "provider": PROVIDER, "model": MODEL}

# ------- Endpoints RAG -------
@app.post("/rag/add_text")
async def rag_add_text(
    text: str = Body(..., example="Colle ici un article, un manuel, etc."),
    chunk_size: int = 800, overlap: int = 120
):
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    await rag_engine.add_docs(chunks)
    rag_engine.save(STORE_DIR)
    return {"added_chunks": len(chunks), "total_chunks": len(rag_engine.docs)}

@app.post("/rag/query")
async def rag_query(query: str = Body(..., example="Question ?"), k: int = 5):
    matches = await rag_engine.search(query, k=k)
    return {
        "query": query,
        "matches": [
            {"rank": rank, "distance": dist, "text": text}
            for (text, dist, rank) in matches
        ]
    }


@app.post("/rag/answer")
async def rag_answer(
    question: str = Body(..., example="Explique en 3 phrases."),
    k: int = 5,
    debug: int = Query(0, description="1 pour renvoyer aussi le prompt")
):
    # 1) Recherche FAISS
    matches = await rag_engine.search(question, k=k)
    if not matches:
        return {"answer": "Aucune connaissance ingérée.", "sources": [], "citations": []}

    # 2) Construit les sources et le contexte numéroté
    sources = []
    ctx_lines = []
    for (text, dist, rank) in matches:
        sources.append({"rank": rank, "distance": dist, "text": text})
        ctx_lines.append(f"[{rank}] {text}")
    context = "\n\n".join(ctx_lines)

    # 3) Prompt RAG
    prompt = (
        "Tu es un assistant RAG. Tu ne dois répondre QU'À PARTIR des extraits fournis ci-dessous.\n"
        "EXIGENCES DE SORTIE :\n"
        "1) Réponds en français, en 3–5 phrases maximum.\n"
        "2) Cite AU MOINS une source en utilisant strictement ce format : [1], [2], etc. (les numéros correspondent aux extraits ci-dessous).\n"
        "3) N'invente aucune information qui n'apparaît pas dans les extraits.\n"
        "4) Si les extraits sont insuffisants, dis : « Contexte insuffisant. »\n\n"
        "EXEMPLE DE FORMAT :\n"
        "Le Soleil est ... [1]. Il ... [2].\n\n"
        f"EXTRAITS :\n{context}\n\n"
        f"QUESTION : {question}\n"
        "RÉPONSE (avec sources entre crochets) :"
    )

    # 4) Appel LLM
    reply = await llm_chat(prompt)

    # 5) Extraire les citations réellement utilisées dans la réponse
    # Gère [1], [2] mais aussi [1, 2, 3] et (1), (2)
    import re

    nums_set = set()

    # 5.1 — Toutes les occurrences entre crochets/parenthèses
    for m in re.findall(r"\[(.*?)\]|\((.*?)\)", reply):
        # m est un tuple (grp1, grp2) — on en prend le non vide
        inside = next((g for g in m if g), "")
        # 5.2 — Récupérer tous les nombres à l’intérieur (gère "1, 2, 3")
        for n in re.findall(r"\d+", inside):
            nums_set.add(int(n))

    nums = sorted(nums_set)

    # Filet de sécurité : si rien n’est détecté, on force au moins les deux premières sources
    if not nums and sources:
        nums = [s["rank"] for s in sources[:2]]
        suffix = "Sources: " + ", ".join(f"[{n}]" for n in nums)
        if "Sources:" not in reply:
            reply = reply.rstrip() + ("\n\n" if not reply.endswith("\n") else "") + suffix

    citations = []
    for n in nums:
        m = next((s for s in sources if s["rank"] == n), None)
        if m:
            citations.append({"rank": n, "text": m["text"]})

    # 6) Retour enrichi
    result = {"answer": reply, "sources": sources, "citations": citations}
    if debug == 1:
        result["prompt"] = prompt
    return result

@app.post("/rag/clear")
def rag_clear():
    """
    Réinitialise le store RAG (supprime les fichiers persistés et vide l'index en mémoire).
    """
    try:
        shutil.rmtree(STORE_DIR, ignore_errors=True)   # supprime store/ (index.faiss + docs.json)
        rag_engine.index = None
        rag_engine.docs = []
        os.makedirs(STORE_DIR, exist_ok=True)
        return {"cleared": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

METRICS = {
    "start_ts": time.time(),
    "requests_total": 0,
    "by_path": defaultdict(int),
    "latencies_ms": []
}