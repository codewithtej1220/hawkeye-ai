@echo off
echo ========================================
echo Setting up AirSim settings.json
echo ========================================
echo.

REM Create AirSim directory in Documents
if not exist "%USERPROFILE%\Documents\AirSim" mkdir "%USERPROFILE%\Documents\AirSim"

REM Copy settings.json
copy /Y "C:\AirSimDemos\settings.json" "%USERPROFILE%\Documents\AirSim\settings.json"

if errorlevel 1 (
    echo ERROR: Failed to copy settings.json
    pause
    exit /b 1
)

echo.
echo ✓ Settings file created at: %USERPROFILE%\Documents\AirSim\settings.json
echo.
echo Configuration:
echo - SimMode: Multirotor (Drone)
echo - Image Resolution: 640x480
echo - Vehicle: SimpleFlight
echo.
echo ========================================
echo Next Steps:
echo ========================================
echo.
echo 1. In Unreal Engine, press ESC to stop simulation
echo 2. Go to Build menu → Build Lighting Only
echo 3. Wait for build to complete
echo 4. Press Alt+P to start simulation
echo 5. Drone should now be controllable!
echo.
echo Manual Controls:
echo - Arrow Keys: Move drone
echo - Page Up/Down: Altitude
echo - F1: Toggle camera view
echo - Backspace: Reset position
echo.
echo Python Control:
echo - Run: python C:\AirSimDemos\thermal.py
echo.
pause
