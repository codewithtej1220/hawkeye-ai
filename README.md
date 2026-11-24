# HawkEye AI — Real‑Time Drone Surveillance System

Autonomous thermal drone surveillance for India's defense systems

## 📋 Project Overview

HawkEye AI is a drone-based surveillance system that detects people and the weapons they carry,
in real time, from thermal and colour cameras. A drone flies over a forest (simulated in Unreal
Engine + AirSim), its cameras stream to a detection backend, and operators watch a live console
that also keeps a permanent log of every event for later review.

### Key features

- **People and weapons, live:** an ensemble of two YOLO11m models trained on real thermal and
  colour footage, plus a zoom pass and a distant-weapon stage.
- **Four camera modes, shown side by side:** RGB, thermal greyscale, thermal INFERNO and RGBT
  sensor fusion. The system measures light, fog, glare and temperature contrast and uses the
  mode that suits the conditions.
- **Built for range:** optical zoom on the drone camera, a small-person search, and an estimated
  range to every person. See [Detection range](#detection-range).
- **Operations console:** live view, drone telemetry (altitude, heading, camera tilt, zoom),
  detections with range and evidence, alerts.
- **Event log for officials:** every threat-level change, ongoing threat and media analysis is
  stored in a database with a snapshot; searchable, filterable, reviewable (who, when, note),
  exportable to CSV.
- **Analyse your own media:** upload photos or videos, or paste a link.
- **A living simulation:** the soldier patrols the forest with his rifle and cycles through
  actions (run, halt and aim, crouch-walk, crouch and aim, look around).
- **Autonomous patrol (NEW):** A reinforcement-learning (PPO) flight policy trained to navigate collision-free through the forest, with threaded depth sensing and a safety shield. The drone investigates detected people by approaching to 9–11 m, zooming, and circling. See [Autonomy/README.md](Autonomy/README.md) for training results (5% crash rate in training world, 0% in Unreal over 5-minute patrols).

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10** or later
- **Node.js 18+**
- **Unreal Engine 4.27** with the AirSim plugin (for simulation)
- **NVIDIA GPU** with CUDA support (recommended for real-time inference)
- **Git LFS 3.7+** (for downloading model files)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/codewithtej1220/hawkeye-ai.git
   cd hawkeye-ai
   ```

2. **Download model files** (after cloning)
   ```bash
   git lfs pull
   ```
   This pulls the detector models (~40 MB each) and the RL flight policy (~1.9 MB).

3. **Install Python dependencies**
   ```bash
   cd project\backend
   pip install -r requirements.txt
   ```
   
   For autonomous patrol, also install:
   ```bash
   pip install stable-baselines3==2.4.1 gymnasium==1.0.0
   ```

4. **Install frontend dependencies**
   ```bash
   cd ..\frontend
   npm install
   ```

### Running the System

**Quick start:**

```bash
RUN_ALL_HAWKEYE.bat          # Manual flight control
RUN_ALL_HAWKEYE.bat auto     # Autonomous patrol (drone circles automatically)
```

For autonomous patrol with reinforcement learning, see [Autonomy/README.md](Autonomy/README.md) or run:
```bash
Autonomy\RUN_AUTONOMY.bat
```

The launcher starts:

1. **Backend** (detector + event log) → http://localhost:8000 (opens API docs)
2. **Dashboard** → http://localhost:5173 (opens in browser)
3. **Forest simulation** (Unreal + AirSim) in game mode (loads in 1–2 minutes)
4. **Drone capture** (`AirSimDemos/thermal.py`) with the soldier's patrol routine

Press any key in the launcher window to stop everything.

To start the parts yourself:

```
cd project\backend   && python -m uvicorn main:app --host 0.0.0.0 --port 8000
cd project\frontend  && npm run dev
cd AirSimDemos       && python thermal.py            (or: python thermal.py --auto)
```

If you start the simulation from the Unreal editor instead, press Play before running
`thermal.py`.

---

## 🏗️ Architecture

```
  Forest simulation (Unreal Engine 4.27 + AirSim)
    soldier with rifle, patrol routine (sim_actor.py)
    drone camera 1280x960, optical zoom 1x-6x
         │  colour frame + segmentation
         ▼
  Drone capture (AirSimDemos/thermal.py)
    renders the thermal camera (white-hot / INFERNO, rifle as cold metal)
    adds telemetry: altitude, heading, camera tilt, zoom
         │  HTTP POST /detect  (thermal + colour + telemetry)
         ▼
  Backend (FastAPI, project/backend)
    1. measure conditions, pick the sensing method (RGB / thermal / RGBT fusion)
    2. detect: 2-model ensemble -> small-person search -> zoom on each person
       -> distant-weapon classifier -> tracking across frames
    3. estimate range, classify the threat level
    4. render the four camera views, write the event log (SQLite + snapshots)
         │  WebSocket /stream (frames, new log entries)     REST /logs, /analyze
         ▼
  Dashboard (React, project/frontend)
    Live operations  |  Analyze media  |  Event log
```

---

## 🤖 Machine Learning

### Primary Algorithm: YOLOv11m

**You Only Look Once v11 (Medium variant)**
- Single-stage object detector
- CSPDarknet backbone with PANet
- Real-time inference (30-60 FPS)
- Fine-tuned from COCO weights on **real** thermal and RGB data (see below)

### Real-data training (`training/`)

The detector `project/models/hawkeye_real_yolo11m.pt` is a single model with two classes,
`person` and `weapon`. It is trained on real imagery rather than on simulator frames:

- **HIT-UAV**: drone-mounted thermal camera; people seen from altitude
- **TCWD**: real thermal footage of weapons (metal shows up as cold, dark regions)
- **Weapon Detection Dataset**: 4k real RGB photos of handguns and knives

To reproduce it, run `training\RUN_TRAINING.bat`. Details are in [training/README.md](training/README.md).
The backend loads this model automatically and falls back to the legacy `weapon_model.pt`
if it is missing. When the new model is present, `AirSimDemos/thermal.py` renders weapons as
cold metal instead of the legacy blue marker, which is how a real thermal camera sees them.

### Four sensing methods, chosen by environment

| Method | Input | Strength |
|---|---|---|
| RGB (visible light) | colour camera | best detail in good light |
| Thermal greyscale | thermal camera, white-hot | darkness, smoke; best when temperature contrast is low |
| Thermal INFERNO | the same thermal data in the INFERNO palette | preferred thermal view: darkness, fog, smoke |
| RGBT sensor fusion | aligned visible + thermal | combines both; each sensor weighted by how reliable it is right now |

For every frame the backend (`project/backend/app/modalities.py`) does the following:

1. **Recognises the input type.** It tells colour images, INFERNO thermal and greyscale apart.
   A lone greyscale image may come from a thermal camera or a monochrome camera, so it is
   analysed as-is.
2. **Measures the conditions:** brightness, contrast, fog/smoke haze, glare, and thermal
   contrast. From these it estimates the reliability of each sensor.
3. **Picks the method:**
   - With both sensors, it fuses them.
   - If the visible image is unusable (night, dense fog/smoke), it uses thermal only (INFERNO).
   - If there is no temperature contrast, it uses RGB only.
   - With a thermal image alone, it uses INFERNO. At low contrast ("thermal crossover" on hot
     days) it applies automatic gain and the greyscale palette, which keeps more detail.
4. **Explains the choice.** Every result reports the method, the reasons and the measured
   conditions, and the dashboard displays them.

Fusion is **decision-level**: the shared detector runs on both views, and matched detections
reinforce each other. A 4-channel network was not used because there is no paired
visible+thermal training data that contains weapons. Training one would have lost the model's
weapon knowledge. The policy was chosen from measurements; see Measured performance.

### Concealed weapons (under clothing)

A visible-light camera cannot see through clothing, so **thermal is the only one of the four
methods that can reveal a hidden weapon**. Metal under a jacket blocks body heat, which leaves a
**cold patch** in the weapon's shape on the warm body (the TCWD training data is real footage of
this).

The backend marks a weapon **"CONCEALED WEAPON"** in two cases:

- **Thermal:** the weapon lies inside the body outline (torso/waist) and is colder than the
  surrounding body.
- **RGBT fusion:** thermal sees the weapon on the body, but the colour camera does not, even
  though it can currently see well.

A rifle held away from the body is not marked.

| Measured | Result |
|---|---|
| Real concealed weapons (TCWD, 182 frames): found / marked concealed | 178 (98%) / 134 of those (75%) |
| Openly carried weapons wrongly marked concealed | ~2% |

The cold patch is small: about 3–4 px for a pistol seen from 50 m. Hidden weapons are therefore
reliable only at close range; at drone range they fall under the distant-weapon stage below.

**Demo:** upload `demo_media/concealed_weapon_close_range_thermal.mp4` (or the `.jpg` still) on
the dashboard. It is real close-range thermal test footage that was never used in training.

### Distant weapons: second stage (switch: off / flag / alert)

A weapon a few pixels wide cannot be outlined reliably. After normal detection, the second
stage runs:

1. **Small-person search.** The frame is split into overlapping 2×2 and 3×3 tiles, each tile is
   enlarged, and the detector looks for **people only** in them. Thermal people at 32 px were
   found 6/30 times before, 26/30 after.
2. **Zoom.** Each person is cropped, enlarged and searched for weapons.
3. **Armed-person classifier.** Small people with no weapon found are judged armed or unarmed
   from the whole low-resolution person: a protrusion, the pose of holding something, or a
   cold patch. This is `project/models/armed_classifier.pt`, a YOLO11s-cls model trained on
   47k real person crops degraded to 16–90 px, in colour, greyscale and INFERNO.
4. **Evidence over time** (live stream and video). Each person is tracked. A person armed in
   2 of their last 6 frames is confirmed.

**How findings are reported:** the operator sees either a weapon or a person, never a "maybe".

- A weapon the detector or the zoom pass **sees** counts at once: the person is armed, threat
  **High**.
- A person with no weapon is threat **Elevated**.
- The weaker signals (this classifier, and the thermal-rifle specialist below) are **not shown on a
  single frame**. A person counts as armed on them only once tracking confirms it in 2 of their
  last 6 frames; then it is a weapon, threat High. On a single uploaded photo they cannot be
  confirmed, so only seen weapons count.

`distant_mode = "off"` in `config.py` (or the dashboard's **Distant-weapon search** selector)
switches the weaker signals off entirely.

The live stream runs the small-person search on every 3rd frame (about 0.6 s extra), and the
tracker carries its results in between.

**Measured on 6-frame clips** (real armed and unarmed test people shrunk to 32–120 px tall and
placed in real street scenes; `training/temporal_benchmark.py`):

| | `off` | **`flag` (default)** |
|---|---|---|
| Armed people raised to High | 15 / 300 (5%) | **38 / 300 (12.7%)** |
| Armed people flagged (High or possible) | 15 / 300 (5%) | **107 / 300 (35.7%)** |
| Unarmed people raised to High | 0 / 190 | **0 / 190** |
| Unarmed people shown as possible | 0 | 17 / 190 (8.9%) |

In `alert` mode, armed people are confirmed about 31% of the time, but 11% of unarmed clips
are raised to High.

**Limits:**

- **Optics:** a camera with optical zoom or a higher resolution would help more than any model.

### Stand-off weapon identification ("detect then zoom")

At surveillance range a weapon is only a few pixels wide. After the full-frame pass, each
detected person is cropped with surrounding context, enlarged, and re-examined for weapons.
This is the software equivalent of slewing a zoom camera onto a target, and it works without
the drone flying closer.

To keep false "armed" alerts low:
- Weapons use their own confidence threshold (0.35 for the full frame, 0.6 for zoom crops).
- Body-part fragments inside a stronger detection are suppressed.
- In uploaded videos, a weapon must persist across neighbouring samples (about 0.5 s apart)
  to count toward the verdict.

SAHI tiled inference is still available (`use_sahi` in `project/backend/app/config.py`) but
is **off by default**. It was measured to flag weapons in 62 of 80 weapon-free street images,
while whole-frame detection flagged none.

### Detection Classes

- **Person/Human** → Green bounding boxes
- **Weapons** (gun, rifle, pistol, knife) → Red bounding boxes

### Models

The detector is an **ensemble of two YOLO11m models** (`project/models/`):

- `hawkeye_real_yolo11m.pt` (v2): 7 real datasets; strongest on night-street RGB.
- `hawkeye_v3.pt` (v3): v2 fine-tuned with an INFERNO copy of every thermal image, 2,000 COCO
  photos of people in every posture, rotated thermal people, and everyday objects as hard
  negatives.

Both models run on every image and their confidences are averaged. When both agree, the
detection stays strong; one model's false alarm is halved. v3 alone was rejected because it was
about 0.02 AP worse than v2 on night-street RGB people. Weight interpolation and a recovery
fine-tune did not fix that; the ensemble did. To use v2 alone (about 2× faster), set
`ensemble_model_path=""` in `project/backend/app/config.py`.

**Thermal-rifle specialist** (`hawkeye_thermal_rifle.pt`). In the simulation the main models
found the soldier reliably but rarely his rifle: the real training data has almost no thermal
footage of an openly carried rifle. The simulator can produce exactly that with perfect labels:
`AirSimDemos/capture_sim_dataset.py` recorded 1,768 viewpoints of the armed soldier and 1,300 with
the rifle hidden, at random bearings, look-down angles, ranges and zoom levels.

- **First attempt, rejected:** both main models fine-tuned on the armed frames mixed with real
  data. Measured with `training/regression_check.py` on real test images, they found fewer people
  (1,442 vs 1,492) and fewer real weapons (351 vs 385), so they were not deployed.
- **Deployed:** v3 fine-tuned on armed and unarmed simulator frames plus 5,600 real images
  (mostly weapon-free thermal people), used only as a **second opinion**. It runs in the zoom pass,
  on thermal views, about a person on whom the main ensemble found no weapon.

Measured with `training/weapon_threshold_sweep.py` (`training/eval/specialist_sweep.txt`):

| At confidence 0.50 | Main models only | With the specialist |
|---|---|---|
| Simulator rifle found (held-out frames) | 6 / 52 | **18 / 52** |
| Real concealed weapons found (TCWD) | 145 / 150 | 145 / 150 |
| Real weapon-free thermal images with a weapon note (LLVIP + HIT-UAV) | 8 / 300 | 33 / 300 |

Because of the last row, its findings are never shown on a single frame: the person counts as armed
on its evidence only once tracking confirms it over several frames (see "How findings are
reported" above). To disable the specialist, set `weapon_specialist_paths=""` in `config.py`.

### Measured performance

All figures are on held-out test images. Reproduce them with `training/evaluate_modalities.py`
and the backend validation scripts; raw results are in `training/eval/`.

**Person AP50 per sensing method and environment**, on 840 aligned night-street RGB+thermal
pairs (v2 model):

| Environment | RGB | Thermal grey | Thermal INFERNO | RGBT fusion | **Auto** |
|---|---|---|---|---|---|
| Normal (night street) | 0.81 | 0.92 | 0.92 | 0.92 | **0.92** |
| Dark | 0.45 | 0.92 | 0.92 | 0.92 | **0.92** |
| Fog / smoke | 0.53 | 0.91 | 0.92 | 0.92 | **0.92** |
| Glare | 0.79 | 0.92 | 0.92 | 0.93 | **0.93** |
| Thermal crossover (hot day) | 0.81 | 0.84 | 0.83 | 0.85 | **0.86** |

**v2 vs ensemble at the deployed settings** (backend code, RTX 4050 Laptop GPU):

| Test | v2 alone | **Ensemble** |
|---|---|---|
| Weapon-free images falsely flagged (street, drone, everyday photos) | 40 / 440 | **8 / 440** |
| Everyday photos (phones, remotes, sports gear) falsely flagged | 34 / 200 | **6 / 200** |
| Weapon-free RGB+thermal pairs flagged (fusion path) | 8 / 840 | 8 / 840 |
| Thermal concealed weapons found (TCWD) | 77 / 80 | **78 / 80** |
| RGB weapons found / rifles found | 69 / 80, 50 / 80 | 69 / 80, **53 / 80** |
| Distant rifle carrier in AirSim (thermal) | 4 / 16 | **6 / 16** |
| People found, drone thermal / everyday photos / AirSim | 47 / 161 / 14 | **49 / 172 / 16** |
| People in any posture (COCO), person AP50 | 0.69 | **0.79** |
| Time per frame: single image / RGB+thermal pair | ~80–160 ms / 274 ms | ~110–285 ms / 435 ms |

**Limits:**

- **Lying-down people:** no real thermal data of lying people exists, so thermal coverage of
  lying-down people comes from rotated training images.

---

## The simulation

The forest map (`AirSimDemos/ForestSimulation/ForestSim`) contains one soldier carrying an FN FAL
rifle and the AirSim drone.

**The soldier.** His Unreal blueprint already contained a wander behaviour and 49 rifle
animations, but he stood frozen: he had no AI controller, and the map's navigation area was about
290 m away from him. `AirSimDemos/unreal_tools/fix_soldier_ai.py` (already applied to the map; the
original is kept in `ForestSimulation/map_backup_2026-11-10`) gives him an AI controller and a
120 m × 120 m navigation area around his start point. `AirSimDemos/sim_actor.py` then directs a
repeating routine over AirSim's console channel:

| Action | Animation | Speed |
|---|---|---|
| Advancing | run | about 3 m/s |
| Halt and aim | standing aim | stopped |
| Sneaking | crouch-walk | about 1 m/s |
| Crouch and aim | crouched aim | stopped |
| Looking around | idle | stopped |

`thermal.py` starts the routine automatically (`HAWKEYE_SOLDIER=off` disables it). The current
action is shown on the dashboard's live view.

**He stays where a demo can find him.** The map's navigation area is 30 m × 30 m around his post,
and the routine walks him back if he strays more than 15 m or gets stuck. The blueprint still
occasionally picks a spot on a slope it cannot reach (its own "ai path finding failed" note, about
one every 20 s); the routine then moves him back onto good ground. Unreal's on-screen debug text is
switched off during capture, so these notes no longer appear in the picture. On the dashboard,
the **Map** card always shows where he is (simulation ground truth), and **G** in the simulation
window turns the drone and camera onto him.

**The cameras.** `AirSimDemos/airsim_settings.json` sets the drone camera to 1280×960 for both the
colour image and the segmentation image (AirSim's default is only 256×144). AirSim reads
`settings.json` from the folder it is started in and from `Documents\AirSim`; the launcher passes
the file explicitly, and copies are kept in both places. The thermal camera is rendered from the
colour image and the segmentation: body warm with a soft gradient, terrain cool and low-contrast,
the rifle as cold metal (`AirSimDemos/sim_common.py`).

**Zoom.** Keys Z / X step the camera through 1×, 1.5×, 2.2×, 3.6× and 6× optical zoom (field of
view 90° to 15°).

**Gimbal.** The camera is stabilised like a drone gimbal: the body's pitch and roll are subtracted
every loop, so the view does not swing when the drone speeds up or turns. In auto mode the drone
really flies (velocity commands) rather than being teleported, and both modes send flight
commands continuously, so AirSim no longer prints "API call was not received, entering hover mode".

**Smooth flight.** Flight commands go out at a steady 20 per second and the speed ramps up and down
(3 m/s²) instead of jumping between 0 and full speed. In `--auto` the drone flies a steady circle
whose centre follows the soldier slowly, so it does not lurch when he moves. Measured: mean jerk
(how abruptly acceleration changes) 2.1 → 0.3 m/s³. Frames are captured at most 3 times a second
(colour and segmentation in parallel, 0.15 s per pair), because each capture costs Unreal two extra
full-resolution renders and capturing flat out made the simulation itself stutter.

**Captured frames on the dashboard.** The dashboard shows each analysed frame (about one a second)
with its own boxes, so the boxes always sit on the person and the weapon. The last 8 frames are kept
in a strip under the live view; click one to look at it, "Back to live" returns.

**Tools** (`AirSimDemos/`):

| Script | Purpose |
|---|---|
| `thermal.py` | fly the drone (or `--auto` orbit), stream frames and telemetry to the backend |
| `sim_actor.py` | the soldier's routine, on its own |
| `range_test.py` | measure detection at increasing ranges, look-down angles and zoom levels |
| `capture_sim_dataset.py` | record auto-labelled training frames from random viewpoints |

## Detection range

Measured in the simulation with `AirSimDemos/range_test.py`: the drone is placed at increasing
slant ranges from the soldier, 35° above him, looking from his side and from the front, with the
wide lens (90° field of view) and the zoom lens (25°, about 3.6×). Two frames per position; full
results in `training/eval/range_test_final.txt`.

| Slant range | Altitude | Person found: wide lens | Person found: zoom lens | Weapon flagged (zoom lens) | Range estimate (zoom) |
|---|---|---|---|---|---|
| 10 m | 6 m | 4 / 4 | 4 / 4 | 4 / 4 | 15 m |
| 20 m | 11 m | 4 / 4 | 4 / 4 | 2 / 4 | 20 m |
| 40 m | 23 m | 4 / 4 | 4 / 4 | 4 / 4 | 43–46 m |
| 60 m | 34 m | 4 / 4 | 4 / 4 | 4 / 4 | 59 m |
| 80 m | 46 m | 3 / 4 | 4 / 4 | 2 / 4 | 77–80 m |
| 100 m | 57 m | 0 / 4 | 4 / 4 | 1 / 4 | 89–91 m |
| 150 m | 86 m | 0 / 4 | 2 / 4 | 0 / 4 | 126 m |
| 200 m | 115 m | 0 / 4 | 3 / 4 | 0 / 4 | 173–182 m |

"Weapon flagged" counts frames where the soldier was reported armed or as a possible weapon. At
these ranges most findings are "possible" (amber) rather than confirmed; tracking across frames
then confirms them.

**What this means for flying:**

- **Find people:** the wide lens covers the most ground and finds people out to about 80 m; the
  zoom lens finds them out to 200 m.
- **Check for a weapon:** zoom in (Z) on the person. Weapons are flagged out to about 80–100 m
  with the zoom lens, and only to about 20 m with the wide lens.
- **Range estimates** are within about 15% from 20 m out. At very close range the person fills
  so much of the frame that the estimate reads high.
- **Limits:** below about 15 px tall a person is no longer found. Higher camera resolution or more
  zoom would extend every number in the table.

## Event log

Every event is written to `project/backend/data/hawkeye_events.db` (SQLite) with an annotated
snapshot in `project/backend/data/snapshots/`:

- **Level change:** the live threat level changed and held for two frames (flicker is ignored).
- **Ongoing:** every 15 s while a High/Critical threat persists, every 60 s while only unarmed
  people are in view.
- **Media analysis:** every uploaded or linked photo/video.

Each entry records the time, level, people / armed people / weapons, the sensing method and
conditions, drone altitude, camera zoom, the estimated range to the nearest person, and each
detection. On the dashboard's **Event log** tab officials can filter by level, source and time,
search, open an event with its snapshot, mark it reviewed with their name and a note, and export
the filtered list to CSV. Events are never edited or deleted from the dashboard; the oldest are
pruned beyond 20,000 (`event_max` in `config.py`).

---

## 📁 Project Structure

```
C:\RTR_Project\
├── RUN_ALL_HAWKEYE.bat              # launcher: starts backend, dashboard, simulation, capture
├── demo_media/                      # test footage (concealed weapon thermal video)
├── project/
│   ├── PREPARE_AND_RUN.bat          # alternative launcher
│   ├── backend/                     # FastAPI server
│   │   ├── main.py                  # API endpoints, live stream, event logging
│   │   ├── app/
│   │   │   ├── detector.py          # detection pipeline (ensemble, zoom, distant stage)
│   │   │   ├── modalities.py        # four sensing methods, conditions, fusion
│   │   │   ├── tracking.py          # people tracked across frames
│   │   │   ├── views.py             # per-camera live views, range estimate
│   │   │   ├── eventlog.py          # persistent event log
│   │   │   ├── media.py             # uploaded photo/video analysis
│   │   │   ├── config.py            # settings
│   │   │   └── schemas.py           # data structures
│   │   ├── requirements.txt         # Python dependencies
│   │   └── data/                    # event database + snapshots (created on first run)
│   ├── frontend/src/app/            # React dashboard (Vite + TypeScript)
│   │   ├── App.tsx                  # shell, tabs, live connection, alerts
│   │   ├── LiveOps.tsx              # live view, camera panels, telemetry, detections
│   │   ├── AnalyzeMedia.tsx         # upload / link analysis
│   │   ├── EventLog.tsx             # event log, review, export
│   │   └── shared.ts                # shared types and helpers
│   └── models/                      # detector weights (Git LFS)
│       ├── hawkeye_real_yolo11m.pt  # main detector v2 (~39 MB)
│       ├── hawkeye_v3.pt            # ensemble detector v3 (~39 MB)
│       ├── armed_classifier.pt      # distant armed-person classifier (~5 MB)
│       ├── hawkeye_thermal_rifle.pt # thermal rifle specialist (~39 MB)
│       └── weapon_model.pt          # legacy weapon detector (~39 MB)
├── AirSimDemos/
│   ├── thermal.py                   # drone capture and flight control
│   ├── sim_common.py                # thermal rendering, segmentation, telemetry
│   ├── sim_actor.py                 # soldier patrol routine
│   ├── range_test.py                # detection range measurement script
│   ├── capture_sim_dataset.py       # auto-labelled training data generator
│   ├── airsim_settings.json         # AirSim camera settings (1280×960)
│   ├── unreal_tools/                # Unreal Blueprint editing scripts
│   │   └── fix_soldier_ai.py        # adds AI controller + navigation to soldier
│   └── ForestSimulation/ForestSim/  # Unreal Engine 4.27 project (Git LFS)
├── Autonomy/                        # NEW: Reinforcement Learning autonomous patrol
│   ├── RUN_AUTONOMY.bat             # launcher for autonomous patrol mode
│   ├── autonomous_patrol.py         # main patrol loop with investigation behavior
│   ├── airsim_pilot.py              # AirSim drone interface (velocity control, gimbal)
│   ├── nav_core.py                  # collision avoidance, safety shield
│   ├── investigate.py               # person investigation behavior (approach, zoom, circle)
│   ├── train_policy.py              # PPO training script
│   ├── train_world.py               # simplified training environment
│   ├── test_offline.py              # offline policy evaluation
│   ├── models/
│   │   ├── nav_policy.zip           # trained PPO policy (1.9 MB, Git LFS)
│   │   └── *.json                   # training and evaluation reports
│   └── README.md                    # training details, results, usage
├── training/                        # real-data training and evaluation pipeline
│   ├── RUN_TRAINING.bat             # download datasets and train from scratch
│   ├── train.py                     # YOLO fine-tuning script
│   ├── evaluate.py                  # test-set evaluation
│   ├── evaluate_modalities.py       # multi-modal evaluation (RGB/thermal/RGBT)
│   ├── prepare_dataset.py           # dataset assembly and augmentation
│   ├── build_v3_dataset.py          # v3 training data with INFERNO copies
│   ├── build_armed_crops.py         # armed-person classifier dataset
│   ├── distance_benchmark.py        # small-person detection evaluation
│   ├── temporal_benchmark.py        # multi-frame weapon confirmation
│   ├── weapon_threshold_sweep.py    # specialist model threshold tuning
│   ├── sources.py                   # dataset download URLs
│   ├── requirements.txt             # training dependencies
│   ├── eval/                        # evaluation result JSON files
│   └── README.md                    # training pipeline documentation
└── README.md                        # this file

## What's NOT in the repository (excluded by .gitignore)

These are either too large, regenerated by scripts, or contain local runtime data:

- **HawkEye_Training/** (~4.9 GB) – training datasets and checkpoints. Download with `training/RUN_TRAINING.bat`.
- **node_modules/** – install with `npm install` in `project/frontend/`.
- **project/backend/data/** – event database and snapshots (created at runtime).
- **live_record/**, range_test_results/ – capture session outputs.
- **Intermediate/**, **Saved/**, **Binaries/** – Unreal Engine build artifacts (regenerated when you open the .uproject).
```

---

## Drone controls

Click the simulation window first.

| Key | Action |
|-----|--------|
| W / S | Forward / backward |
| A / D | Turn left / right |
| Arrow keys | Strafe left / right |
| Space / Shift | Ascend / descend |
| R / F | Camera tilt up / down |
| Z / X | Zoom in / out |
| G | Find the soldier: turn the drone and tilt the camera onto him |
| T | Save a colour + thermal frame pair |
| ESC | Land and stop the capture |

`python thermal.py --auto` needs no keys: the drone orbits the soldier
(`--radius`, `--altitude`, `--orbit-period`, `--fov`).

---

## Technical stack

Backend: FastAPI, Ultralytics YOLO, OpenCV, PyTorch, SQLite

Frontend: React, Vite, WebSocket, TailwindCSS

Simulation: Unreal Engine 4.27, AirSim, Python airsim client

---

## API endpoints

REST
- POST /detect: `{"image": <base64 thermal frame>, "rgb": <optional base64 visible frame of the
  same view>, "method": "auto", "telemetry": {altitude_m, heading_deg, camera_pitch_deg,
  camera_fov_deg, ...}}`. Runs detection, broadcasts the result (with the four camera views) to
  the dashboard and writes the event log. If the detector is busy, the frame is skipped rather
  than queued.
- POST /analyze/upload: multipart `file` (an image or video up to 500 MB, thermal or visible,
  recognised automatically), an optional `thermal` file of the same scene (enables RGBT fusion),
  and `method` (`auto` | `rgb` | `thermal_grey` | `thermal_inferno` | `rgbt`). Returns the threat
  verdict, detections, annotated key frames, the method used and why, the measured conditions
  and, for videos, a threat timeline.
- POST /analyze/url: `{"url": "https://.../file.mp4", "method": "auto"}`, a direct link to an image
  or video. Links to local or private network addresses are refused.
- GET /logs: the event log, newest first. Filters: `level`, `min_level`, `source`, `since`,
  `until`, `search`, `unacknowledged`, `limit`, `offset`.
- GET /logs/stats, GET /logs/{id}, GET /logs/{id}/snapshot, GET /logs/export.csv
- POST /logs/{id}/acknowledge: `{"by": "name", "note": "..."}` marks an event as reviewed.
- GET /health: service health, the models in use, live-stream state

WebSocket
- WS /stream: live frames (`frame`), new log entries (`log`) and reviews (`log_update`)

---

## 🎓 Algorithms Implemented

1. **YOLO11m ensemble** - two single-stage detectors, confidences averaged
2. **Environment-driven sensor selection and decision-level RGBT fusion**
3. **Detect-then-zoom** - each person is magnified and re-examined for weapons
4. **Small-person tile search and armed-person classifier** - the distant stage
5. **Tracking across frames** - evidence accumulates per person
6. **Monocular range estimation** - from apparent size, field of view and camera tilt
7. **Thermal rendering** - distance-transform heat gradient, cold-metal weapons, sensor noise
8. **Threat classification** - Clear / Elevated / High / Critical

---

## Future enhancements

- Multi-drone coordination
- Edge deployment on drone hardware
- Mobile apps for remote monitoring
- More characters and weapon types in the simulation
- Real thermal footage of openly carried rifles for training

---

## Team

HawkEye AI — Thermal Surveillance System
Created by my team during Real-time Research Project; now developed and maintained by me.

---

## License

See the included LICENSE file for licensing details.

---

## 🙏 Acknowledgments

- **Microsoft AirSim** - Drone simulation framework
- **Ultralytics** - YOLOv11 implementation
- **SAHI Team** - Sliced inference library
- **Unreal Engine** - 3D environment platform

---

## 📞 Contact & Support

For technical queries or demo requests, please refer to the presentation materials in `project/FINAL_PRESENTATIONS/`.

---

## Troubleshooting

- **Windows Smart App Control:** May block the unsigned AirSim plugin DLL. If the simulation crashes on start, go to Windows Security → App & browser control → Smart App Control, and temporarily disable it or allow the ForestSim.exe.

- **Port 8000 already in use:** Another app is on the backend port. Stop it, or run the backend on another port (`uvicorn main:app --port 8001`). Then create `project/frontend/.env.local` containing `VITE_BACKEND_PORT=8001` and restart the dashboard. Also set `BACKEND_URL` in `AirSimDemos/thermal.py` to the new port (`thermal.py` reads `HAWKEYE_BACKEND_URL`, e.g. `set HAWKEYE_BACKEND_URL=http://127.0.0.1:8001`).:8001`.)
- **"Detector failed to load: DLL load failed ... An Application Control policy has blocked this
  file":** Windows Smart App Control blocks the unsigned library of `optree` (installed with
  Keras). HawkEye does not need it and tells PyTorch it is not installed
  (`project/backend/app/__init__.py`), so this should not appear. If another library is blocked,
  the Windows event log (Code Integrity) names the file.
- **The dashboard shows "No drone stream":** the capture (`thermal.py`) is not running or cannot
  reach the simulator. It needs the simulation to be playing; the launcher waits for it.
- **The soldier is out of view:** check the Map card on the dashboard, then press G in the
  simulation window.
- **"Elevated" with nobody in view:** fixed. When the thermal picture had nothing warm in it, the
  detector sometimes outlined most of the frame as a person or weapon (15 of 80 empty-forest
  frames). On the drone stream such frame-wide boxes are now dropped (`live_max_person_width_pct`
  in `config.py`); measured on the same 80 frames: 0 false alarms.
