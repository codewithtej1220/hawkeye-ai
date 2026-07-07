# Quick Start Guide - ForestSimulation with AirSim

## 🚀 Fast Setup (5 Steps)

### 1️⃣ Build AirSim
```cmd
cd c:\AirSimDemos\AirSim
build.cmd
```
⏱️ Takes ~20-30 minutes

### 2️⃣ Run Setup Script
```cmd
cd c:\AirSimDemos\ForestSimulation
setup.bat
```

### 3️⃣ Generate Project Files
- Right-click `ForestSim.uproject`
- Select **"Generate Visual Studio project files"**

### 4️⃣ Build in Visual Studio
- Open `ForestSim.sln`
- Configuration: **Development Editor | Win64**
- Press **Ctrl+Shift+B** to build

### 5️⃣ Launch!
- Double-click `ForestSim.uproject`
- Press **Alt+P** to play

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| **Alt+P** | Start/Stop simulation |
| **F1** | Toggle camera view |
| **Backspace** | Reset drone |
| **~** | Open console |

---

## 🐍 Python Control

### Install AirSim Python Package
```cmd
pip install airsim
```

### Run Test Script
```cmd
python test_drone.py
```

### Basic Python Commands
```python
import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff
client.takeoffAsync().join()

# Move
client.moveToPositionAsync(10, 0, -10, 5).join()

# Land
client.landAsync().join()
```

---

## 📁 Important Files

| File | Location |
|------|----------|
| **Project** | `c:\AirSimDemos\ForestSimulation\ForestSim\ForestSim.uproject` |
| **Settings** | `C:\Users\[You]\Documents\AirSim\settings.json` |
| **AirSim Source** | `c:\AirSimDemos\AirSim` |

---

## ⚠️ Troubleshooting

### Drone doesn't appear?
✅ Check `settings.json` has `"SimMode": "Multirotor"`

### Plugin failed to load?
✅ Rebuild AirSim: `cd c:\AirSimDemos\AirSim && build.cmd`

### Missing DLL error?
✅ Build project in Visual Studio first

### Crash on startup?
✅ Delete `Saved`, `Intermediate`, `Binaries` folders and rebuild

---

## 📚 Resources

- [AirSim Docs](https://microsoft.github.io/AirSim/)
- [Python API](https://microsoft.github.io/AirSim/apis/)
- [Settings Guide](https://microsoft.github.io/AirSim/settings/)

---

## ✅ Checklist

- [ ] AirSim built successfully
- [ ] Plugin copied to ForestSim/Plugins/AirSim
- [ ] settings.json created
- [ ] Visual Studio project generated
- [ ] Project built in VS
- [ ] Unreal Engine launches without errors
- [ ] Drone appears in simulation
- [ ] Python test script works

**Happy Flying! 🚁🌲**
