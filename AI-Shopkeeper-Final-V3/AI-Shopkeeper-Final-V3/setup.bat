@echo off
cd /d "%~dp0"
echo Creating virtual environment...
python -m venv .venv
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
set /p LSTM="Install TensorFlow for the LSTM experiment (about 400 MB)? [y/N] "
if /i "%LSTM%"=="y" python -m pip install -r requirements-lstm.txt
if not exist "frontend\vendor\chart.umd.min.js" (
    powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -Uri 'https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js' -OutFile 'frontend\vendor\chart.umd.min.js' } catch { Write-Host 'Chart.js download skipped.' }"
)
echo.
echo Setup finished. Start the app with run.bat
pause
