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

# CPU-only PyTorch for PPO/imitation training (avoids 2.5GB CUDA bloat)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full application (backend + prebuilt frontend/dist)
COPY . .

# Writable runtime dirs for trained policies
RUN mkdir -p models/balance models/scene_tasks models/imitation models/rl

CMD ["sh", "-c", "exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
