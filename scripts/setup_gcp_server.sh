#!/usr/bin/env bash
# ==============================================================================
# RoboForge - Google Cloud Platform (GCP) Compute Engine Automated Setup
# Target OS: Ubuntu 24.04 LTS / 22.04 LTS
# ==============================================================================
set -e

echo "🚀 [1/6] Updating system packages and installing prerequisites..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git curl nginx certbot python3-certbot-nginx build-essential

echo "📦 [2/6] Installing Node.js 20 LTS..."
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt install -y nodejs
fi
echo "Node version: $(node -v)"
echo "NPM version: $(npm -v)"

echo "📂 [3/6] Setting up repository workspace..."
sudo mkdir -p /var/www
sudo chown -R $USER:$USER /var/www

APP_DIR="/var/www/meta-project"
if [ ! -d "$APP_DIR" ]; then
    git clone https://github.com/Deep-khatke/Meta-.git "$APP_DIR"
fi

cd "$APP_DIR"

echo "🐍 [4/6] Setting up Python virtual environment & backend dependencies..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" websockets mujoco numpy pydantic httpx python-dotenv

echo "⚛️ [5/6] Building frontend production bundle..."
cd "$APP_DIR/frontend"
npm install
npm run build
cd "$APP_DIR"

echo "⚙️ [6/6] Configuring systemd service and Nginx reverse proxy..."

# 1. Systemd Service
sudo tee /etc/systemd/system/roboforge.service > /dev/null <<EOF
[Unit]
Description=RoboForge Robotics Platform & Simulation Backend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/.venv/bin:/usr/local/bin:/usr/bin"
EnvironmentFile=-$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable roboforge
sudo systemctl restart roboforge

# 2. Nginx Configuration
sudo tee /etc/nginx/sites-available/roboforge > /dev/null <<'EOF'
server {
    listen 80;
    server_name roboforge.duckdns.org localhost;

    # Frontend Single Page App
    root /var/www/meta-project/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Backend REST API
    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Backend Swagger Docs
    location /docs {
        proxy_pass http://127.0.0.1:8000/docs;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /openapi.json {
        proxy_pass http://127.0.0.1:8000/openapi.json;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }

    # Simulation & Telemetry WebSockets
    location /ws/ {
        proxy_pass http://127.0.0.1:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }
}
EOF

sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -sf /etc/nginx/sites-available/roboforge /etc/nginx/sites-enabled/roboforge
sudo nginx -t
sudo systemctl reload nginx

echo "=================================================================="
echo "✅ RoboForge GCP Setup Completed Successfully!"
echo "Backend Status: $(sudo systemctl is-active roboforge)"
echo "Nginx Status:   $(sudo systemctl is-active nginx)"
echo ""
echo "Next Steps:"
echo "1. Update DuckDNS (roboforge.duckdns.org) to point to this VM's External IP."
echo "2. Run Certbot for free HTTPS:"
echo "   sudo certbot --nginx -d roboforge.duckdns.org"
echo "=================================================================="
