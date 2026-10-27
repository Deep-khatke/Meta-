# RoboForge Launcher Script for PowerShell
Write-Host "Starting RoboForge Robotics Simulation..." -ForegroundColor Cyan

# Start Backend
$backend = Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "Write-Host 'Starting RoboForge FastAPI Backend...'; .venv\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000" -PassThru

# Start Frontend
$frontend = Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "Write-Host 'Starting RoboForge Frontend...'; cd frontend; npm run dev" -PassThru

Write-Host "RoboForge services launched!" -ForegroundColor Green
Write-Host "Frontend: http://127.0.0.1:5173" -ForegroundColor Yellow
Write-Host "Backend API: http://127.0.0.1:8000" -ForegroundColor Yellow
