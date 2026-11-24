@echo off
REM Re-applies the soldier patrol fix to the map (AI controller + navigation area). Only needed if
REM the map is restored from an older copy. Close Unreal first. The original map is kept in
REM AirSimDemos\ForestSimulation\map_backup_2026-11-10.
"C:\Program Files\Epic Games\UE_4.27\Engine\Binaries\Win64\UE4Editor-Cmd.exe" "%~dp0..\ForestSimulation\ForestSim\ForestSim.uproject" -EnablePlugins=PythonScriptPlugin,EditorScriptingUtilities -run=pythonscript -script="%~dp0fix_soldier_ai.py" -unattended -nosplash -nullrhi
echo Done.
pause
