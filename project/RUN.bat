@echo off
set "DIR=%~dp0"

echo ============================================
echo   HAWKEYE INTEGRATED SYSTEM
echo ============================================
echo.

echo [1/4] Starting Backend...
start "HawkEye-Backend" cmd /k "cd /d "%DIR%backend" && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
timeout /t 5 >nul

echo [2/4] Starting Frontend...
start "HawkEye-Frontend" cmd /k "cd /d "%DIR%frontend" && npm run dev"
timeout /t 3 >nul

echo [3/4] Starting Unreal Engine...
start "" "C:\Program Files\Epic Games\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe" "C:\RTR_Project\AirSimDemos\ForestSimulation\ForestSim\ForestSim.uproject"

echo.
echo ============================================
echo   WAITING FOR UNREAL ENGINE
echo ============================================
echo.
echo 1. Wait for Unreal Engine to load
echo 2. Press Alt+P to start simulation
echo 3. Wait for drone to appear
echo.
pause

echo.
echo [4/4] Starting Capture System...
echo.
echo ============================================
echo   CONTROLS (in capture window):
echo ============================================
echo   W/S      - Forward/Backward
echo   A/D      - Yaw Left/Right
echo   Arrows   - Strafe Left/Right
echo   Space    - Up
echo   Shift    - Down
echo   R/F      - Camera Up/Down
echo   ESC      - Exit
echo ============================================
echo.
echo FLY THE DRONE to see movement and detection!
echo.
pause

cd /d "%DIR%airsim_capture"
python integrated_capture.py
