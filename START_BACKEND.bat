@echo off
title PNG Decomposer Backend
cd /d "e:\Antigravity project\photoshop plugin\backend"

echo ================================================
echo   PNG to Photoshop Layers - Backend Server
echo ================================================
echo.
echo Starting FastAPI server on http://127.0.0.1:8000
echo Press CTRL+C to stop the server.
echo.
echo API Docs: http://127.0.0.1:8000/docs
echo Health:   http://127.0.0.1:8000/api/v1/health
echo.

.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

pause

