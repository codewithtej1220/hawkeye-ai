# ForestSimulation - AirSim Drone Environment 🚁🌲

A forest environment for AirSim drone simulation using Unreal Engine 4.27.2.

## 📋 What You Have

- ✅ **ForestSim Unreal Project** - Forest environment assets
- ✅ **AirSim Source Code** - Located at `c:\AirSimDemos\AirSim`
- ✅ **Unreal Engine 4.27.2** - Installed
- ✅ **Visual Studio 2022** - Installed

## 🎯 Quick Start

**New to this? Start here:** → [QUICK_START.md](QUICK_START.md)

**Need detailed instructions?** → [SETUP_GUIDE.md](SETUP_GUIDE.md)

## 📦 What's Included

```
ForestSimulation/
├── README.md                    # This file
├── QUICK_START.md              # Fast setup guide
├── SETUP_GUIDE.md              # Detailed setup instructions
├── setup.bat                   # Automated setup script
├── test_drone.py               # Python test script
├── settings.json.example       # AirSim configuration template
└── ForestSim/
    ├── ForestSim.uproject      # Unreal Engine project file
    ├── Content/                # Environment assets
    │   ├── Mixamo/            # Character/animation assets
    │   └── StarterContent/    # UE4 starter content
    └── Plugins/
        └── AirSim/            # AirSim plugin (copy here after build)
```

## 🚀 Setup in 3 Commands

```cmd
# 1. Build AirSim
cd c:\AirSimDemos\AirSim
build.cmd

# 2. Run setup script
cd c:\AirSimDemos\ForestSimulation
setup.bat

# 3. Generate VS project, build, and launch!
```

## 🎮 Usage

### Launch Simulation
1. Double-click `ForestSim.uproject`
2. Press **Alt+P** to start simulation

### Control with Python
```python
import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()
client.moveToPositionAsync(10, 0, -10, 5).join()
client.landAsync().join()
```

### Run Test Script
```cmd
pip install airsim
python test_drone.py
```

## 📖 Documentation

- **Quick Start**: [QUICK_START.md](QUICK_START.md) - Get running in 5 steps
- **Setup Guide**: [SETUP_GUIDE.md](SETUP_GUIDE.md) - Detailed instructions with troubleshooting
- **AirSim Docs**: https://microsoft.github.io/AirSim/
- **Python API**: https://microsoft.github.io/AirSim/apis/

## 🔧 System Requirements

- **OS**: Windows 10/11
- **Unreal Engine**: 4.27.2
- **Visual Studio**: 2022 (with C++ desktop development)
- **Python**: 3.6+ (for API control)
- **Disk Space**: ~50GB (UE4 + AirSim + Project)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GTX 1060 or better recommended

## 🎯 What Can You Do?

- ✈️ Fly drones in a realistic forest environment
- 📸 Capture RGB, depth, and segmentation images
- 🤖 Train AI/ML models for autonomous navigation
- 🧪 Test computer vision algorithms
- 🎮 Develop drone control algorithms
- 📊 Collect simulation data for research

## ⚠️ Common Issues

| Problem | Solution |
|---------|----------|
| Plugin failed to load | Rebuild AirSim and copy plugin |
| Missing DLL | Build project in Visual Studio |
| Drone doesn't appear | Check settings.json SimMode |
| Crash on startup | Delete Saved/Intermediate/Binaries folders |

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed troubleshooting.

## 📝 Configuration

AirSim settings location: `C:\Users\[YourUsername]\Documents\AirSim\settings.json`

See `settings.json.example` for a complete configuration template.

## 🤝 Support

- **AirSim Issues**: https://github.com/microsoft/AirSim/issues
- **Documentation**: https://microsoft.github.io/AirSim/
- **Community**: https://github.com/microsoft/AirSim/discussions

## 📄 License

This project uses:
- **AirSim**: MIT License
- **Unreal Engine**: Epic Games License
- **ForestSim Assets**: Check individual asset licenses

## 🎉 Ready to Fly?

1. Read [QUICK_START.md](QUICK_START.md)
2. Run `setup.bat`
3. Build in Visual Studio
4. Launch and enjoy! 🚁

**Happy Flying in the Forest! 🌲✨**
