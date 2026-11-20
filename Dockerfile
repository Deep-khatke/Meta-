# ---- Stage 1: build the React/Three.js frontend ----
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
ARG VITE_GOOGLE_CLIENT_ID=""
ENV VITE_GOOGLE_CLIENT_ID=${VITE_GOOGLE_CLIENT_ID}
RUN npm run build

# ---- Stage 2: FastAPI + MuJoCo backend that also serves the built frontend ----
FROM python:3.11-slim

# MuJoCo headless rendering libraries for Cloud Run Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=osmesa
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# CPU-only PyTorch for the PPO / imitation training (the default CUDA wheel is ~2 GB larger; Cloud Run has no GPU).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=frontend /fe/dist ./frontend/dist

# Writable runtime dirs for trained policies
RUN mkdir -p models/balance models/scene_tasks models/imitation models/rl

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
