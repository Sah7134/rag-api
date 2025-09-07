# ===== 1) Build stage =====
FROM python:3.12-slim AS build

# Evite les prompts interactifs et garde l'image légère
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Installer dépendances système minimales (faiss-cpu a besoin de libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de dépendances en premier (meilleur cache)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# ===== 2) Runtime stage =====
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# libgomp pour faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier seulement le nécessaire
COPY --from=build /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=build /usr/local/bin /usr/local/bin
COPY . /app

# Port exposé (Cloud Run ignore EXPOSE mais c'est utile en local)
EXPOSE 8080

# Uvicorn écoute 0.0.0.0:8080 (Cloud Run utilise $PORT=8080)
ENV PORT=8080
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
