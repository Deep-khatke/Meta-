# RoboForge Launcher Script for PowerShell
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "     ROBOFORGE - INDUSTRIAL ROBOTICS SIMULATION STUDIO    " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check Python environment
$pythonPath = if (Test-Path "$scriptDir\.venv\Scripts\python.exe") { "$scriptDir\.venv\Scripts\python.exe" } else { "python" }

# Start Backend
Write-Host "`n[1/3] Starting RoboForge FastAPI Physics Server..." -ForegroundColor Yellow
$backend = Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "Write-Host 'RoboForge Physics Backend (:8000)'; & '$pythonPath' -m uvicorn backend.app:app --host 127.0.0.1 --port 8000" -PassThru

# Start Frontend
Write-Host "[2/3] Starting RoboForge Vite Frontend..." -ForegroundColor Yellow
$frontend = Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "Write-Host 'RoboForge Vite Frontend (:5173)'; Set-Location '$scriptDir\frontend'; npm run dev" -PassThru

# Wait 3 seconds
Write-Host "[3/3] Waiting for servers to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

Write-Host "`n[SUCCESS] RoboForge is live!" -ForegroundColor Green
Write-Host "  * Landing Page:    http://localhost:5173/" -ForegroundColor Cyan
Write-Host "  * Studio:          http://localhost:5173/studio" -ForegroundColor Cyan
Write-Host "  * FastAPI Backend: http://127.0.0.1:8000/" -ForegroundColor Cyan
Write-Host "  * Swagger Docs:    http://127.0.0.1:8000/docs" -ForegroundColor Cyan

# Launch Browser
Write-Host "`nLaunching web browser..." -ForegroundColor Green
Start-Process "http://localhost:5173/"
