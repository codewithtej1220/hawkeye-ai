# 🚁 HawkEye AI - Drone-Based Intelligent Surveillance System

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![React 18](https://img.shields.io/badge/react-18-61dafb.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Real-Time Human and Weapon Threat Detection using Thermal Imaging, YOLOv11m, and SAHI**

Autonomous drone-based thermal surveillance system combining Microsoft AirSim simulation, Unreal Engine 4 rendering, deep learning (YOLOv11m + SAHI), and a full-stack web application for real-time threat detection and classification.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Performance Metrics](#performance-metrics)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Team](#team)
- [License](#license)

---

## 🎯 Overview

HawkEye AI addresses critical failures in traditional surveillance systems:
- ❌ **Illumination Dependency**: RGB cameras fail in darkness
- ❌ **Human Fatigue**: Operators experience cognitive decline after 20-30 minutes
- ❌ **Fixed Blind Spots**: Static cameras create exploitable coverage gaps
- ❌ **Delayed Response**: Manual assessment introduces critical latency

### Our Solution

✅ **Thermal Imaging**: 24/7 operation independent of lighting conditions  
✅ **AI-Powered Detection**: Real-time human & weapon detection with 92% and 85% accuracy  
✅ **SAHI Integration**: +20% improvement in small object (weapon) detection  
✅ **Real-Time Dashboard**: Sub-300ms end-to-end latency with live WebSocket streaming  
✅ **4-Level Threat Classification**: Clear → Elevated → High → Critical  

---

## ✨ Key Features

### 🔍 AI Detection
- **SMART_DETECTOR**: Dual-model approach (RGB + Thermal models)
- **YOLOv11m Architecture**: 20M parameters, optimized for thermal images
- **SAHI (Slicing Aided Hyper Inference)**: 512×512 slicing with 30% overlap
- **Weapon Detection**: 85% accuracy (improved from 65% using SAHI)
- **Human Detection**: 92% accuracy with 98.3% precision

### 🎯 Threat Classification
- **4-Level System**: Clear / Elevated / High / Critical
- **Intelligent Association**: Weapon-to-person linking with 35% bbox expansion
- **Risk Scoring**: 0-10 scale with real-time assessment
- **Low False Positives**: Only 3.2% false alarm rate

### 🖥️ Web Dashboard
- **Live Feed**: Split-view (original + annotated thermal)
- **Real-Time Updates**: WebSocket streaming at 1 FPS
- **Color-Coded Alerts**: Green (human) / Red (weapon) bounding boxes
- **Detection Archive**: 50-entry rolling window with searchable history
- **Auto-Reconnection**: 3-second delay, 99.998% uptime

### 🎮 Simulation Environment
- **Microsoft AirSim 1.8.1**: High-fidelity drone simulation
- **Unreal Engine 4.27**: Photorealistic ForestSim environment (1km²)
- **Thermal Generation**: 6-step pipeline (segmentation → colormap → blur)
- **Keyboard Control**: W/A/S/D, Space, Shift, R/F controls

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    4-TIER ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   TIER 1     │───▶│   TIER 2     │───▶│   TIER 3     │───▶│   TIER 4     │
│  SIMULATION  │    │   CAPTURE    │    │   BACKEND    │    │  FRONTEND    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
  AirSim + UE4       Python Script       FastAPI + AI       React Dashboard
  Thermal Camera     Base64 Encode       YOLOv11m+SAHI     WebSocket Client
  640×480 @ 1 FPS    HTTP POST           Threat Logic      Live Visualization
```

### Data Flow (265ms end-to-end)
1. **Thermal Generation** (20ms): Unreal → AirSim → Distance transform → Colormap
2. **Transmission** (10ms): NumPy → JPEG → Base64 → HTTP POST
3. **AI Detection** (152ms): SAHI slicing → YOLOv11m inference → NMS merge
4. **Annotation** (30ms): Draw bounding boxes → Encode images
5. **Broadcasting** (5ms): JSON → WebSocket → All clients
6. **Visualization** (35ms): React state update → Dashboard render

---

## 📊 Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Human Detection** | 92% | 85% | ✅ +7% |
| **Weapon Detection** | 85% | 75% | ✅ +10% |
| **Threat Classification** | 92.5% | 95% | ⚠️ -2.5% |
| **AI Inference Time** | 152ms | 200ms | ✅ 24% faster |
| **End-to-End Latency** | 265ms | 500ms | ✅ 47% faster |
| **False Positive Rate** | 3.2% | <5% | ✅ 36% better |
| **System Uptime** | 99.998% | 99% | ✅ 0.998% above |

### Detailed Accuracy (80 test images)

**Human Detection:**
- Precision: 98.3% (116 TP, 2 FP)
- Recall: 91.3% (11 FN out of 127)
- F1 Score: 94.7%
- Avg Confidence: 89.2%

**Weapon Detection:**
- Precision: 95.5% (42 TP, 2 FP)
- Recall: 80.8% (10 FN out of 52)
- F1 Score: 87.5%
- Avg Confidence: 76.4%

**SAHI Impact:**
- Without SAHI: 65% weapon detection (baseline)
- With SAHI: 85% weapon detection (+20% improvement!)

---

## 🛠️ Tech Stack

### Simulation Layer
- **Microsoft AirSim 1.8.1**: Drone simulation with thermal camera
- **Unreal Engine 4.27**: High-fidelity 3D rendering
- **ForestSim (Custom)**: 1km² forest environment

### AI Detection Layer
- **YOLOv11m**: 20M parameter object detection model
- **SAHI 0.11.14**: Slicing-aided hyper inference
- **PyTorch 2.0+**: Deep learning framework
- **OpenCV 4.8+**: Image processing
- **CUDA 11.8+**: GPU acceleration

### Backend Layer
- **FastAPI 0.104+**: Async REST API framework
- **Uvicorn**: ASGI server with WebSocket support
- **Pydantic 2.0+**: Data validation

### Frontend Layer
- **React 18**: Component-based UI
- **TypeScript 5.0+**: Type-safe development
- **Vite**: Lightning-fast build tool
- **Tailwind CSS 3.3+**: Utility-first styling
- **Lucide React**: Icon library

### Hardware Requirements
- **CPU**: Intel Core i7 / AMD Ryzen 7 (8+ cores)
- **RAM**: 16GB (32GB recommended)
- **GPU**: NVIDIA RTX 3060+ (6GB+ VRAM, CUDA support)
- **Storage**: 50GB free SSD
- **OS**: Windows 10/11 64-bit

---

## 📥 Installation

### Prerequisites
```bash
# Install Python 3.10
python --version  # Should show 3.10.x

# Install Node.js 18+
node --version    # Should show 18.x or higher

# Install CUDA 11.8+ (for GPU support)
nvidia-smi        # Verify GPU and CUDA installation
```

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/hawkeye-ai.git
cd hawkeye-ai
```

### 2. Setup Backend
```bash
cd project/backend
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 3. Setup Frontend
```bash
cd project/frontend
npm install
```

### 4. Download Models
```bash
# Models are NOT included in Git (too large)
# Download from releases or train your own

# Place models in project/models/:
# - yolo11m_rgb_weapon_final.pt (38.7 MB)
# - yolo11m_thermal_weapon_ultimate.pt (38.6 MB)
```

### 5. Install AirSim (Optional - for simulation)
```bash
# Download AirSim 1.8.1
# https://github.com/microsoft/AirSim/releases

# Install Unreal Engine 4.27
# https://www.unrealengine.com/download
```

---

## 🚀 Usage

### Quick Start (All-in-One)
```bash
# From project root
cd project
RUN.bat  # Windows - starts backend + frontend

# Or manually:

# Terminal 1: Start Backend
cd project/backend
python -m uvicorn main:app --reload --port 8000

# Terminal 2: Start Frontend
cd project/frontend
npm run dev  # Opens at http://localhost:5173

# Terminal 3: Run AirSim Capture (if using simulation)
cd AirSimDemos
python thermal.py
```

### Access Dashboard
```
http://localhost:5173
```

### API Endpoints
```
Backend API:    http://localhost:8000
Health Check:   http://localhost:8000/health
WebSocket:      ws://localhost:8000/stream
API Docs:       http://localhost:8000/docs
```

### Drone Controls (in AirSim)
```
W/S:      Forward / Backward
A/D:      Yaw Left / Right
Arrows:   Strafe Left / Right
Space:    Ascend
Shift:    Descend
R/F:      Camera Pitch Up / Down
ESC:      Land and Exit
```

---

## 📁 Project Structure

```
RTR_Project/
├── AirSimDemos/                    # Simulation environment
│   ├── ForestSimulation/           # Unreal Engine project
│   │   └── ForestSim/              # 1km² forest environment
│   ├── HawkEye_Manual_Dataset/     # Test dataset (5 images)
│   ├── thermal.py                  # Thermal capture script
│   ├── settings.json               # AirSim configuration
│   └── launch_simulation.bat       # Simulation launcher
│
├── project/                        # Main application
│   ├── backend/                    # FastAPI server
│   │   ├── app/
│   │   │   ├── detector.py         # SMART_DETECTOR implementation
│   │   │   ├── config.py           # Configuration settings
│   │   │   └── schemas.py          # Pydantic models
│   │   ├── main.py                 # FastAPI entry point
│   │   └── requirements.txt        # Python dependencies
│   │
│   ├── frontend/                   # React dashboard
│   │   ├── src/
│   │   │   ├── app/
│   │   │   │   └── App.tsx         # Main dashboard component
│   │   │   └── main.tsx            # React entry point
│   │   ├── package.json            # Node dependencies
│   │   └── vite.config.ts          # Vite configuration
│   │
│   ├── models/                     # YOLOv11m models (not in Git)
│   │   ├── yolo11m_rgb_weapon_final.pt
│   │   └── yolo11m_thermal_weapon_ultimate.pt
│   │
│   ├── FINAL_PRESENTATIONS/        # Presentation materials
│   │   ├── README.txt              # Presentation guide
│   │   ├── 1_PROJECT_WORKFLOW.txt
│   │   ├── 2_ML_ENGINEER_SPEECH.txt
│   │   ├── 4_FRONTEND_BACKEND_SPEECH.txt
│   │   ├── 6_SAHI_SPEECH.txt
│   │   └── 8_SIMULATION_SPEECH.txt
│   │
│   └── RUN.bat                     # All-in-one launcher
│
├── display/                        # Display module
│   ├── hawkeye_detector.py         # Detection display script
│   └── README.md
│
├── model/                          # Model utilities
│
├── MY_FINAL_MODEL_FILES.md         # Model documentation
├── MY_MODEL_FILES_LIST.txt         # Model file list
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## 👥 Team

**RTR Project Team** - JNTUH University College of Engineering

| Name | Role | Contributions |
|------|------|---------------|
| **Amula Bhanuteja** | ML Engineer | Dataset creation, YOLOv11m training, SMART_DETECTOR, system integration |
| **Poreddy Varshitha** | SAHI Specialist | SAHI integration, NMS optimization, +20% weapon detection improvement |
| **Rompicharla Lahari** | Simulation Engineer | ForestSim environment, AirSim integration, thermal generation, weapon dataset |
| **Rahul Badam** | Full-Stack Engineer | FastAPI backend, React frontend, WebSocket streaming, dashboard UI |

**Supervisor:** Dr. D. Vasumathi (Professor of CSE & Director of DUFR)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Microsoft AirSim**: For the excellent drone simulation framework
- **Ultralytics YOLO**: For the YOLOv11m object detection model
- **SAHI Project**: For small object detection enhancement
- **FastAPI**: For the modern async Python framework
- **React Team**: For the UI library

---

## 📞 Contact

For questions or collaboration:
- Email: [your-email@example.com]
- GitHub Issues: [github.com/yourusername/hawkeye-ai/issues]

---

## 🚀 Future Work

- [ ] Physical drone deployment (DJI Matrice 300 RTK)
- [ ] Weapon type classification (rifle vs pistol vs knife)
- [ ] Multi-object tracking across frames (DeepSORT)
- [ ] Pose estimation (holstered vs ready)
- [ ] CRPF and military integration
- [ ] Edge deployment on NVIDIA Jetson Xavier NX
- [ ] Model quantization (INT8) for faster inference

---

**Built with ❤️ by RTR Project Team | JNTUH 2025-2026**

⭐ **Star this repo if you find it useful!**
