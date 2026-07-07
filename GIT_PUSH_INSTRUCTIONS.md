# 🚀 Git Push Instructions for HawkEye AI

## Quick Start (Copy & Paste)

### Step 1: Initialize Git Repository
```bash
cd C:\RTR_Project
git init
```

### Step 2: Configure Git (if first time)
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Step 3: Add All Files
```bash
git add .
```

### Step 4: Verify What Will Be Committed
```bash
# Check staged files
git status

# Check ignored files (should show models, node_modules, etc.)
git status --ignored
```

**Expected ignored files:**
- `project/models/*.pt` (model files)
- `project/frontend/node_modules/` (npm packages)
- `__pycache__/` directories
- `*.log` files

### Step 5: First Commit
```bash
git commit -m "Initial commit: HawkEye AI v1.0 - Drone-based thermal surveillance system with YOLOv11m + SAHI"
```

### Step 6: Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `hawkeye-ai` (or your choice)
3. Description: "Drone-based thermal surveillance system for real-time human and weapon detection using YOLOv11m and SAHI"
4. **DO NOT** initialize with README (we already have one)
5. Click "Create repository"

### Step 7: Link to GitHub and Push
```bash
# Replace 'yourusername' with your actual GitHub username
git remote add origin https://github.com/yourusername/hawkeye-ai.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

---

## ⚠️ Before Pushing - Final Checks

### Check Repository Size
```bash
# Check total size (should be ~2.8 GB)
# Most of this is Unreal Engine assets (ForestSim)
git count-objects -vH
```

### Verify Large Files Are Ignored
```bash
# Should NOT show any *.pt files
git ls-files | findstr ".pt"

# If it shows model files, they will be pushed (WARNING!)
# Make sure .gitignore is correct
```

### Test Clone (After First Push)
```bash
# In a different directory, test cloning
cd C:\Temp
git clone https://github.com/yourusername/hawkeye-ai.git
cd hawkeye-ai

# Verify models are NOT present
dir project\models
# Should show: "File Not Found" or empty directory
```

---

## 📦 Create GitHub Release for Models

After successfully pushing to GitHub, create a release with model files:

### Step 1: Create Release on GitHub
1. Go to your repository: `https://github.com/yourusername/hawkeye-ai`
2. Click "Releases" → "Create a new release"
3. Tag version: `v1.0.0`
4. Release title: `HawkEye AI v1.0.0 - Initial Release`

### Step 2: Upload Model Files
Drag and drop these files to the release assets:
- `C:\RTR_Project\project\models\daytime_human_model.pt` (38.65 MB)
- `C:\RTR_Project\project\models\nighttime_human_model.pt` (38.62 MB)
- `C:\RTR_Project\project\models\weapon_model.pt` (38.64 MB)

Or create a ZIP:
```bash
cd C:\RTR_Project\project\models
powershell Compress-Archive -Path *.pt -DestinationPath C:\RTR_Project\hawkeye_models_v1.0.zip
```

### Step 3: Write Release Notes
```markdown
# HawkEye AI v1.0.0 - Initial Release

🚁 First stable release of HawkEye AI - Drone-based thermal surveillance system

## 🎯 Features
- Real-time human detection (92% accuracy)
- Weapon detection with SAHI (85% accuracy)
- 4-level threat classification
- FastAPI backend with WebSocket streaming
- React dashboard with real-time updates
- AirSim + Unreal Engine simulation

## 📦 Model Files
Download and place in `project/models/`:
- `daytime_human_model.pt` (38.65 MB) - RGB human detection
- `nighttime_human_model.pt` (38.62 MB) - Thermal human detection  
- `weapon_model.pt` (38.64 MB) - Weapon detection

## 📊 Performance
- Human Detection: 92% accuracy (98.3% precision, 91.3% recall)
- Weapon Detection: 85% accuracy (95.5% precision, 80.8% recall)
- End-to-End Latency: 265ms
- SAHI Improvement: +20% weapon detection

## 🛠️ Installation
See [README.md](https://github.com/yourusername/hawkeye-ai#installation)

## 👥 Team
Amula Bhanuteja, Poreddy Varshitha, Rompicharla Lahari, Rahul Badam

## 📄 License
MIT License - See [LICENSE](https://github.com/yourusername/hawkeye-ai/blob/main/LICENSE)
```

### Step 4: Update README with Download Link
After creating release, update README.md:
```bash
cd C:\RTR_Project
# Edit README.md, replace "Download from releases" with actual link:
# https://github.com/yourusername/hawkeye-ai/releases/download/v1.0.0/daytime_human_model.pt
```

---

## 🔧 Common Issues

### Issue: "fatal: not a git repository"
**Solution:**
```bash
cd C:\RTR_Project
git init
```

### Issue: "remote origin already exists"
**Solution:**
```bash
git remote remove origin
git remote add origin https://github.com/yourusername/hawkeye-ai.git
```

### Issue: Model files are being committed
**Solution:**
```bash
# Check if .gitignore is working
git check-ignore project/models/*.pt
# Should output the file paths (meaning they're ignored)

# If not ignored, verify .gitignore exists and contains:
# *.pt
# *.pth
# project/models/*.pt

# Remove from staging if accidentally added
git rm --cached project/models/*.pt
git commit -m "Remove model files from tracking"
```

### Issue: Push rejected (repository too large)
**Solution:**
```bash
# GitHub has a 100 MB file limit and 1-5 GB repo size soft limit
# If ForestSim is too large, consider:
# 1. Using Git LFS for large Unreal Engine assets
# 2. Hosting ForestSim separately
# 3. Providing download link instead of including in repo

# Install Git LFS
git lfs install
git lfs track "*.uasset"
git lfs track "*.umap"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### Issue: node_modules was committed
**Solution:**
```bash
git rm -r --cached project/frontend/node_modules
git commit -m "Remove node_modules from tracking"
git push
```

---

## 📋 Post-Push Checklist

After successfully pushing to GitHub:

- [ ] Repository is public/private as intended
- [ ] README displays correctly on GitHub
- [ ] LICENSE file is recognized by GitHub
- [ ] .gitignore is working (no *.pt files visible)
- [ ] GitHub shows correct file count (~67K files)
- [ ] Created release v1.0.0 with model files
- [ ] Updated README with model download links
- [ ] Added repository description and topics on GitHub
- [ ] Added project URL to README if applicable
- [ ] Invited team members as collaborators (if needed)
- [ ] Verified clone and setup works from fresh directory

---

## 🌟 GitHub Repository Settings (Recommended)

### Topics to Add
Go to repository settings and add topics:
- `computer-vision`
- `object-detection`
- `yolov11`
- `thermal-imaging`
- `drone`
- `airsim`
- `fastapi`
- `react`
- `surveillance`
- `weapon-detection`
- `sahi`

### About Section
```
Drone-based thermal surveillance system for real-time human and weapon detection using YOLOv11m, SAHI, AirSim, and FastAPI
```

### Repository URL
Add if you have a demo/presentation URL

---

## 🎓 For Your Presentation

**Repository Status:**
- ✅ Clean, professional structure
- ✅ Comprehensive documentation
- ✅ MIT open-source license
- ✅ Ready to showcase

**Demo the following:**
1. Show GitHub repository with README
2. Highlight performance metrics table
3. Show architecture diagram
4. Mention open-source (MIT licensed)
5. Show team contributions in README

---

**Good luck with your presentation!** 🚀

Generated: July 7, 2026
