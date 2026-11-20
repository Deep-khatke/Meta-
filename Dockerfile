# ==========================================
# Stage 1: Build Vite Frontend
# ==========================================
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# ==========================================
# Stage 2: Python Backend + MuJoCo Headless
# ==========================================
FROM python:3.11-slim

# MuJoCo headless rendering libraries for Linux container
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=osmesa
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application
COPY . .

# Copy compiled frontend from Stage 1 into frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

CMD ["sh", "-c", "exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
