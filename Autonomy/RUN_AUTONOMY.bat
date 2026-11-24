@echo off
setlocal
title HawkEye Autonomy - launcher
REM Starts backend, dashboard and the forest simulation, then the autonomous patrol (no keyboard needed).
set "ROOT=%~dp0.."
set "UE=C:\Program Files\Epic Games\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe"

powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) { exit 1 }"
if errorlevel 1 (
    echo Port 8000 is already in use by another program. Close it first.
    pause
    exit /b 1
)
echo [1/4] Backend ...
start "HawkEye-Backend" /min cmd /k "cd /d "%ROOT%\project\backend" && python -m uvicorn main:app --host 0.0.0.0 --port 8000"
echo [2/4] Dashboard ...
start "HawkEye-Dashboard" /min cmd /k "cd /d "%ROOT%\project\frontend" && npm run dev -- --port 5173 --strictPort"
echo [3/4] Forest simulation (1-2 minutes) ...
tasklist /FI "IMAGENAME eq UE4Editor.exe" 2>NUL | find /I "UE4Editor.exe" >NUL
if errorlevel 1 start "HawkEye-Simulation" /D "%ROOT%\AirSimDemos" "%UE%" "%ROOT%\AirSimDemos\ForestSimulation\ForestSim\ForestSim.uproject" -game -windowed -ResX=1280 -ResY=720 -settings="%ROOT%\AirSimDemos\airsim_settings.json"
powershell -NoProfile -Command "$t=0; while (-not (Get-NetTCPConnection -LocalPort 41451 -State Listen -ErrorAction SilentlyContinue)) { Start-Sleep 3; $t+=3; if ($t -gt 600) { exit 1 } }"
if errorlevel 1 (
    echo The simulator did not start within 10 minutes. If Windows Smart App Control blocks the AirSim
    echo plugin, see README.md - Troubleshooting.
    pause
    exit /b 1
)
echo [4/4] Autonomous patrol ...
start "" http://localhost:5173
cd /d "%~dp0"
python autonomous_patrol.py
pause
endlocal
