@echo off
echo ============================================
echo FINAL SETUP: AirSim Forest Environment
echo ============================================
echo.

cd /d "C:\AirSimDemos\ForestSimTemp\ForestSim"

echo Step 1: Checking AirSim plugin...
if not exist "Plugins\AirSim\Binaries" (
    echo ERROR: AirSim plugin not found!
    echo Running setup script...
    call setup_forest_airsim.bat
)

echo.
echo Step 2: Cleaning old build files...
if exist "Binaries" rmdir /s /q "Binaries"
if exist "Intermediate" rmdir /s /q "Intermediate"
if exist ".vs" rmdir /s /q ".vs"

echo.
echo Step 3: Generating Visual Studio project files...
"C:\Program Files\Epic Games\UE_4.27\Engine\Binaries\DotNET\UnrealBuildTool.exe" -projectfiles -project="C:\AirSimDemos\ForestSimTemp\ForestSim\ForestSim.uproject" -game -rocket -progress

echo.
echo Step 4: Building project in Visual Studio...
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" "ForestSim.sln" /t:Build /p:Configuration="Development Editor" /p:Platform=Win64 /m

if errorlevel 1 (
    echo Trying alternative MSBuild path...
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe" "ForestSim.sln" /t:Build /p:Configuration="Development Editor" /p:Platform=Win64 /m
)

echo.
echo Step 5: Opening Unreal Engine...
start "" "C:\Program Files\Epic Games\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe" "C:\AirSimDemos\ForestSimTemp\ForestSim\ForestSim.uproject"

echo.
echo ============================================
echo SETUP COMPLETE!
echo ============================================
echo.
echo NEXT STEPS IN UNREAL ENGINE:
echo 1. Wait for Unreal Engine to load
echo 2. Press Alt+P to start simulation
echo 3. Drone should spawn in the forest!
echo.
echo TO RUN YOUR THERMAL CAPTURE:
echo   cd C:\AirSimDemos
echo   python thermal.py
echo.
pause
