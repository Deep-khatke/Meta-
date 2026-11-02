@echo off
chcp 65001 >nul
title RoboForge Studio Controller
cd /d "%~dp0"
color 0F

echo =====================================================================
echo           ROBOFORGE - INDUSTRIAL ROBOTICS SIMULATION STUDIO
echo =====================================================================
echo.
echo  [1/3] Initializing MuJoCo Physics Backend (FastAPI on port 8000)...

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

start "RoboForge_Backend" cmd /k "title RoboForge Backend && "%PYTHON_EXE%" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"

echo  [2/3] Initializing Vite 3D Engineering Frontend (port 5173)...
start "RoboForge_Frontend" cmd /k "title RoboForge Frontend && cd /d "%~dp0frontend" && npm run dev"

echo  [3/3] Waiting for services to initialize...
timeout /t 3 /nobreak >nul

echo.
echo =====================================================================
echo  [SUCCESS] All RoboForge services are active and running:
echo    * 3D Landing Page:    http://localhost:5173/
echo    * Engineering Studio: http://localhost:5173/studio
echo    * Physics API Hub:    http://127.0.0.1:8000/
echo    * Swagger API Docs:   http://127.0.0.1:8000/docs
echo =====================================================================
echo.
echo  Opening default web browser to Landing Page...
start http://localhost:5173/
echo.
echo  Keep this controller window open while using RoboForge.
echo  Press any key in this window to stop all servers and exit...
pause >nul

echo.
echo  Shutting down RoboForge servers...
taskkill /F /FI "WINDOWTITLE eq RoboForge Backend*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq RoboForge Frontend*" >nul 2>&1
taskkill /F /IM "uvicorn.exe" >nul 2>&1
echo  [DONE] All services stopped cleanly.
timeout /t 2 >nul
exit

