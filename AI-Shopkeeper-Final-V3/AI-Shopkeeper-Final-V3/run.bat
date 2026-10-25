@echo off
cd /d "%~dp0"

rem Use the virtual environment if setup.bat created one.
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

rem Install requirements only if something is missing (fast start otherwise).
python -c "import fastapi, uvicorn, pandas, sklearn, scipy, lightgbm, shap, multipart" 2>nul
if errorlevel 1 (
    echo Installing requirements...
    python -m pip install -r requirements.txt
)

rem Optional: fetch Chart.js once so the dashboard works offline later.
if not exist "frontend\vendor\chart.umd.min.js" (
    powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -Uri 'https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js' -OutFile 'frontend\vendor\chart.umd.min.js' } catch { Write-Host 'Chart.js download skipped (offline). Built-in chart fallback will be used.' }"
)

echo.
echo Starting AI Shopkeeper at http://127.0.0.1:8000  (Ctrl+C to stop)
python -m uvicorn backend.app:app
pause
