# ForestSimulation Setup Guide for AirSim

## Prerequisites
- ✅ Unreal Engine 4.27.2 (installed)
- ✅ Visual Studio 2022 (installed)
- ✅ AirSim source code (at `c:\AirSimDemos\AirSim`)
- ✅ ForestSimulation environment (extracted)

## Step-by-Step Setup

### 1. Build AirSim Plugin

First, you need to build the AirSim plugin for Unreal Engine 4.27:

```cmd
cd c:\AirSimDemos\AirSim
build.cmd
```

This will:
- Build the AirSim plugin
- Create necessary binaries
- Set up dependencies

**Note:** This may take 20-30 minutes on first build.

### 2. Copy AirSim Plugin to ForestSim Project

After building, copy the AirSim plugin to your ForestSim project:

```cmd
xcopy /E /I /Y "c:\AirSimDemos\AirSim\Unreal\Plugins\AirSim" "c:\AirSimDemos\ForestSimulation\ForestSim\Plugins\AirSim"
```

### 3. Generate Visual Studio Project Files

Right-click on `ForestSim.uproject` and select:
- **"Generate Visual Studio project files"**

If this option doesn't appear:
1. Make sure Unreal Engine 4.27.2 is properly associated
2. Or run manually:

```cmd
cd c:\AirSimDemos\ForestSimulation\ForestSim
"C:\Program Files\Epic Games\UE_4.27\Engine\Build\BatchFiles\Build.bat" ForestSimEditor Win64 Development "c:\AirSimDemos\ForestSimulation\ForestSim\ForestSim.uproject" -waitmutex
```

### 4. Open and Build in Visual Studio

1. Open `ForestSim.sln` in Visual Studio 2022
2. Set build configuration to **"Development Editor"** and **"Win64"**
3. Build the solution (Ctrl+Shift+B)

### 5. Configure AirSim Settings

Create AirSim settings file at: `C:\Users\[YourUsername]\Documents\AirSim\settings.json`

**For Drone Simulation:**

```json
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ViewMode": "SpringArmChase",
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": 0, "Z": -2,
      "Yaw": 0
    }
  },
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,
        "Width": 1920,
        "Height": 1080,
        "FOV_Degrees": 90
      }
    ]
  }
}
```

### 6. Launch the Environment

1. Double-click `ForestSim.uproject` to open in Unreal Engine
2. Wait for shaders to compile (first time only)
3. Click **Play** button or press **Alt+P**

### 7. Control the Drone

Once the simulation starts:

**Keyboard Controls:**
- **F1**: Toggle FPV/Chase camera view
- **Backspace**: Reset drone position
- **Arrow Keys**: Manual control (if enabled)

**Python API Control:**

```python
import airsim

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff
client.takeoffAsync().join()

# Move to position
client.moveToPositionAsync(0, 0, -10, 5).join()

# Land
client.landAsync().join()
```

## Troubleshooting

### Issue: "Plugin 'AirSim' failed to load"
**Solution:** Rebuild the AirSim plugin and ensure it's copied to the Plugins folder.

### Issue: "Missing ForestSim.dll"
**Solution:** Build the project in Visual Studio before launching.

### Issue: Unreal Engine crashes on startup
**Solution:** 
1. Delete `Saved`, `Intermediate`, and `Binaries` folders
2. Regenerate project files
3. Rebuild in Visual Studio

### Issue: Drone doesn't appear
**Solution:** Check that `settings.json` has `SimMode` set to `"Multirotor"`.

## Additional Resources

- [AirSim Documentation](https://microsoft.github.io/AirSim/)
- [AirSim APIs](https://microsoft.github.io/AirSim/apis/)
- [Settings Documentation](https://microsoft.github.io/AirSim/settings/)

## Project Structure

```
ForestSimulation/
└── ForestSim/
    ├── ForestSim.uproject          # Main project file
    ├── Content/                     # Environment assets
    │   ├── Mixamo/                 # Character/animation assets
    │   └── StarterContent/         # UE4 starter content
    ├── Plugins/
    │   └── AirSim/                 # AirSim plugin
    └── Saved/                      # Generated files
```

## Next Steps

1. ✅ Build AirSim
2. ✅ Copy plugin to project
3. ✅ Generate VS project files
4. ✅ Build in Visual Studio
5. ✅ Create settings.json
6. ✅ Launch and test!

Good luck with your forest drone simulation! 🚁🌲
