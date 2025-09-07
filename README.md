<<<<<<< HEAD
# AI-ML-Engineer-applied-
=======
# AI/ML Applied – RAG API (FastAPI + Gemini + FAISS)

Une API **de démonstration** qui combine :
- **FastAPI** pour l’exposition des endpoints,
- **Google Gemini** pour la génération de texte,
- **FAISS** pour l’indexation et la recherche vectorielle (RAG),
- **Docker** + **Cloud Run** pour le déploiement cloud.

Ce projet illustre une compétence clé d’**AI/ML Engineer appliqué** :  
👉 construire et déployer une API de bout en bout qui exploite l’IA de manière concrète.

---

## 🚀 Fonctionnalités

- `/chat` → conversation avec LLM
- `/chat_qs` → version simple (GET) sans JSON
- `/rag/add_text` → ingestion d’un texte (chunking automatique)
- `/rag/query` → recherche top-k passages
- `/rag/answer` → réponse augmentée avec citations `[1] [2]`
- `/rag/clear` → réinitialiser le store FAISS
- `/metrics` → monitoring (latence p50/p90/p99, total requêtes)

---

## 📦 Installation locale (sans Docker)

### Prérequis
- Python 3.10+
- Clé API Gemini (Google Generative AI)

### Étapes
```bash
git clone https://github.com/TON-UTILISATEUR/rag-api.git
cd rag-api

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
>>>>>>> 3443286 (RAG API initial commit)
