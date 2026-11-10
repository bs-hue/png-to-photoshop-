# PNG Decomposer - Backend Startup Script (PowerShell)
# Double-click START_BACKEND.bat instead, or run this in PowerShell.

Set-Location "e:\Antigravity project\photoshop plugin\backend"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  PNG to Photoshop Layers - Backend Server" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting FastAPI on http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Docs:   http://127.0.0.1:8000/docs"
Write-Host "Health: http://127.0.0.1:8000/api/v1/health"
Write-Host ""
Write-Host "Press CTRL+C to stop." -ForegroundColor Yellow
Write-Host ""

& ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

