@echo off
echo ============================================
echo   HAWKEYE AI - AUTOMATIC SETUP
echo ============================================
echo.

echo Fixing Unreal Engine project...
cd /d "C:\RTR_Project\AirSimDemos\ForestSimulation\ForestSim"

echo Removing corrupted build files...
if exist Binaries rmdir /s /q Binaries 2>nul
if exist Intermediate rmdir /s /q Intermediate 2>nul
if exist Saved rmdir /s /q Saved 2>nul
if exist Plugins\AirSim\Binaries rmdir /s /q Plugins\AirSim\Binaries 2>nul
if exist Plugins\AirSim\Intermediate rmdir /s /q Plugins\AirSim\Intermediate 2>nul

echo.
echo Build files cleaned successfully!
echo.
timeout /t 2 >nul

echo ============================================
echo   STARTING ALL SYSTEMS
echo ============================================
echo.

cd /d "C:\RTR_Project\project"
call RUN.bat
