@echo off
echo ============================================
echo Launching AirSim Forest Simulation
echo ============================================
echo.

echo Copying settings.json to AirSim folder...
if not exist "%USERPROFILE%\Documents\AirSim" mkdir "%USERPROFILE%\Documents\AirSim"
copy /Y "C:\RTR_Project\AirSimDemos\settings.json" "%USERPROFILE%\Documents\AirSim\settings.json"

echo.
echo Starting Unreal Engine with ForestSim project...
start "" "C:\Program Files\Epic Games\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe" "C:\RTR_Project\AirSimDemos\ForestSimulation\ForestSim\ForestSim.uproject"

echo.
echo ============================================
echo Unreal Engine is starting...
echo ============================================
echo.
echo WAIT for Unreal Engine to fully load, then:
echo 1. Press Alt+P to start the simulation
echo 2. Once the drone appears, run: python thermal.py
echo.
echo If you see errors about missing plugin:
echo - The AirSim plugin needs to be built first
echo - This requires Visual Studio and can take 20-30 minutes
echo.
pause
