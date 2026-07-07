@echo off
echo ========================================
echo ForestSimulation AirSim Setup Script
echo ========================================
echo.

REM Step 1: Build AirSim
echo [1/4] Building AirSim plugin...
cd /d "c:\AirSimDemos\AirSim"
if not exist "build.cmd" (
    echo ERROR: build.cmd not found in AirSim directory
    pause
    exit /b 1
)
call build.cmd
if errorlevel 1 (
    echo ERROR: AirSim build failed
    pause
    exit /b 1
)
echo AirSim build completed successfully!
echo.

REM Step 2: Copy AirSim plugin
echo [2/4] Copying AirSim plugin to ForestSim project...
xcopy /E /I /Y "c:\AirSimDemos\AirSim\Unreal\Plugins\AirSim" "c:\AirSimDemos\ForestSimulation\ForestSim\Plugins\AirSim"
if errorlevel 1 (
    echo ERROR: Failed to copy AirSim plugin
    pause
    exit /b 1
)
echo Plugin copied successfully!
echo.

REM Step 3: Create settings.json
echo [3/4] Creating AirSim settings.json...
set SETTINGS_DIR=%USERPROFILE%\Documents\AirSim
if not exist "%SETTINGS_DIR%" mkdir "%SETTINGS_DIR%"

(
echo {
echo   "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
echo   "SettingsVersion": 1.2,
echo   "SimMode": "Multirotor",
echo   "ViewMode": "SpringArmChase",
echo   "Vehicles": {
echo     "Drone1": {
echo       "VehicleType": "SimpleFlight",
echo       "X": 0, "Y": 0, "Z": -2,
echo       "Yaw": 0
echo     }
echo   },
echo   "CameraDefaults": {
echo     "CaptureSettings": [
echo       {
echo         "ImageType": 0,
echo         "Width": 1920,
echo         "Height": 1080,
echo         "FOV_Degrees": 90
echo       }
echo     ]
echo   }
echo }
) > "%SETTINGS_DIR%\settings.json"
echo Settings file created at: %SETTINGS_DIR%\settings.json
echo.

REM Step 4: Generate Visual Studio project files
echo [4/4] Generating Visual Studio project files...
echo Please right-click on ForestSim.uproject and select "Generate Visual Studio project files"
echo Then open ForestSim.sln in Visual Studio 2022 and build the solution.
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Right-click ForestSim.uproject and select "Generate Visual Studio project files"
echo 2. Open ForestSim.sln in Visual Studio 2022
echo 3. Set configuration to "Development Editor" and "Win64"
echo 4. Build the solution (Ctrl+Shift+B)
echo 5. Double-click ForestSim.uproject to launch in Unreal Engine
echo.
pause
