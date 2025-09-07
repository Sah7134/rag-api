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

- Endpoint `/chat` : conversation simple avec le modèle choisi.  
- Endpoint `/rag/add_text` : ajout de texte dans l’index vectoriel FAISS.  
- Endpoint `/rag/answer` : réponse à une question en s’appuyant sur les documents indexés (RAG).  
- Endpoint `/metrics` : statistiques (uptime, total de requêtes, latence p50/p90/p99).  
- Endpoint `/health` : vérification de l’état du service.  
- Documentation interactive via **Swagger UI** → `/docs`.

---

## ⚡ Installation locale (sans Docker)

### 1. Cloner le projet
```bash
git clone https://github.com/Sah7134/rag-api.git
cd rag-api
```

### 2. Créer un environnement virtuel
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Configurer l’environnement
Copier `.env.example` → `.env` et y ajouter votre clé Gemini :

```
GOOGLE_GENAI_API_KEY=ta_cle
PROVIDER=gemini
MODEL=gemini-1.5-flash
EMBED_MODEL=text-embedding-004
```

### 5. Lancer l’API
```bash
uvicorn main:app --reload
```

### 6. Tester
- Swagger UI : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
- Health check : [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 🐳 Installation avec Docker

### 1. Construire l’image
```bash
docker build -t rag-api:latest .
```

### 2. Lancer un conteneur
```bash
docker run -p 8000:8000 rag-api:latest
```

### 3. Accéder
- Swagger UI : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## ☁️ Déploiement Cloud Run

1. Construire et pousser l’image vers Artifact Registry :  
```bash
docker build -t $IMAGE .
docker push $IMAGE
```

2. Déployer sur Cloud Run :  
```bash
gcloud run deploy rag-api   --image $IMAGE   --region $REGION   --platform managed   --allow-unauthenticated   --port 8080   --set-env-vars "PROVIDER=gemini,MODEL=gemini-1.5-flash,EMBED_MODEL=text-embedding-004"   --set-secrets "GOOGLE_GENAI_API_KEY=google-genai-key:latest"
```

3. Récupérer l’URL publique :  
```bash
gcloud run services describe rag-api --region $REGION --format "value(status.url)"
```

---

## ✅ Tests (pytest)

Deux tests simples (`test_smoke.py`) sont inclus :  
- Vérification du `/health`.  
- Vérification du `/chat_qs` avec un mock.

Exécuter :  
```bash
pytest -q
```

---

## 📊 Exemple de requête RAG

### Ajouter un texte
```bash
curl -X POST "http://127.0.0.1:8000/rag/add_text" -H "Content-Type: application/json" -d "Le Soleil est une étoile."
```

### Poser une question
```bash
curl -X POST "http://127.0.0.1:8000/rag/answer" -H "Content-Type: application/json" -d "Explique le Soleil en 3 phrases."
```

Réponse attendue :  
```json
{
  "answer": "Le Soleil est une étoile au centre du système solaire...",
  "sources": [...],
  "citations": [1,2,3]
}
```

---

## 📌 Points clés du projet

- **Serverless** : déploiement via Cloud Run (scaling auto, aucun serveur à maintenir).  
- **Sécurité** : clé API stockée dans Secret Manager.  
- **Reproductible** : `.env.example`, `requirements.txt`, Dockerfile, README clair.  
- **Portfolio** : projet démonstratif d’AI/ML Engineer appliqué.

---

## 👤 Auteur
**Amin Hassaïni**  
📧 contact@aminhassaini.be  
💻 [GitHub](https://github.com/Sah7134)  
