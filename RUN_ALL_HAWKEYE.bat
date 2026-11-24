@echo off
setlocal
title HawkEye AI - launcher
cd /d "%~dp0"

REM ============================================================================================
REM  HawkEye AI - starts the whole system:
REM    1. backend (detector + event log)        http://localhost:8000
REM    2. dashboard                             http://localhost:5173
REM    3. forest simulation (Unreal + AirSim), in game mode: no "Play" button to press
REM    4. drone capture (thermal.py) streaming to the backend
REM
REM  RUN_ALL_HAWKEYE.bat          manual flight (W/S/A/D, arrows, Space/Shift, R/F tilt, Z/X zoom)
REM  RUN_ALL_HAWKEYE.bat auto     hands-free: the drone orbits the soldier by itself
REM ============================================================================================

set "ROOT=%~dp0"
set "UE=C:\Program Files\Epic Games\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe"
set "MODE=%~1"

echo.
echo  HAWKEYE AI - starting all systems
echo  ---------------------------------

REM Port 8000 must be free for the backend (another local server on it would answer instead)
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) { exit 1 }"
if errorlevel 1 (
    echo.
    echo  [!] Port 8000 is already in use by another program. Close it first, or the dashboard will
    echo      talk to the wrong server. See README.md - Troubleshooting.
    echo.
    pause
    exit /b 1
)

echo  [1/4] Backend ...
start "HawkEye-Backend" /min cmd /k "cd /d "%ROOT%project\backend" && python -m uvicorn main:app --host 0.0.0.0 --port 8000"

echo  [2/4] Dashboard ...
start "HawkEye-Dashboard" /min cmd /k "cd /d "%ROOT%project\frontend" && npm run dev -- --port 5173 --strictPort"

echo  [3/4] Forest simulation (takes 1-2 minutes to load) ...
tasklist /FI "IMAGENAME eq UE4Editor.exe" 2>NUL | find /I "UE4Editor.exe" >NUL
if errorlevel 1 (
    start "HawkEye-Simulation" /D "%ROOT%AirSimDemos" "%UE%" "%ROOT%AirSimDemos\ForestSimulation\ForestSim\ForestSim.uproject" -game -windowed -ResX=1280 -ResY=720 -settings="%ROOT%AirSimDemos\airsim_settings.json"
) else (
    echo        Unreal is already running: using it. If it is the editor, press Play.
)

echo        waiting for the simulator and the detector ...
powershell -NoProfile -Command "$t=0; while (-not (Get-NetTCPConnection -LocalPort 41451 -State Listen -ErrorAction SilentlyContinue)) { Start-Sleep 3; $t+=3; if ($t -gt 600) { exit 1 } }"
if errorlevel 1 (
    echo  [!] The simulator did not start within 10 minutes.
    pause
    exit /b 1
)
powershell -NoProfile -Command "$t=0; while ($true) { try { $h = Invoke-RestMethod http://127.0.0.1:8000/health -TimeoutSec 3; if ($h.detector_ready) { break } } catch {}; Start-Sleep 3; $t+=3; if ($t -gt 300) { exit 1 } }"
if errorlevel 1 echo  [!] The backend is not ready yet; the capture will keep retrying.

echo  [4/4] Drone capture ...
if /I "%MODE%"=="auto" (
    start "HawkEye-Capture" cmd /k "cd /d "%ROOT%AirSimDemos" && python thermal.py --auto"
) else (
    start "HawkEye-Capture" cmd /k "cd /d "%ROOT%AirSimDemos" && python thermal.py"
)

start "" http://localhost:5173

echo.
echo  HawkEye AI is running.
echo.
echo    Dashboard    http://localhost:5173     (Live operations / Analyze media / Event log)
echo    Backend      http://localhost:8000/health
echo.
if /I "%MODE%"=="auto" (
    echo    The drone is orbiting the soldier by itself.
) else (
    echo    Fly the drone: click the simulation window, then
    echo      W / S  forward / back      A / D  turn        arrows  strafe
    echo      SPACE / SHIFT  up / down   R / F  camera tilt Z / X   zoom in / out
    echo      T  save a frame pair       ESC    land and stop the capture
)
echo.
echo  Press any key here to STOP everything (backend, dashboard, capture, simulation).
pause >nul

echo  Stopping ...
taskkill /F /FI "WINDOWTITLE eq HawkEye-Capture*" /T >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq HawkEye-Backend*" /T >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq HawkEye-Dashboard*" /T >nul 2>&1
taskkill /F /IM UE4Editor.exe >nul 2>&1
echo  Done.
endlocal
