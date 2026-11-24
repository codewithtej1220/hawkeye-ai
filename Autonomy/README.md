# HawkEye Autonomy: learned, collision-free autonomous patrol

A separate module (nothing in the main project is changed): the drone flies a patrol route through
the forest **by itself**, avoiding trees and hills with a flight policy it **learned by reinforcement
learning**, while the HawkEye detector keeps checking every frame for people and weapons.

```
Autonomy/
├── nav_core.py           what the drone senses and commands (shared by training and flight), safety shield
├── train_world.py        fast training world: hills, random forests, depth-ray sensor, drone dynamics
├── train_policy.py       reinforcement learning (PPO) -> models/nav_policy.zip + training_report.json
├── airsim_pilot.py       real depth camera -> policy -> flight command, in the Unreal simulator
├── autonomous_patrol.py  the mission: patrol route + obstacle avoidance + continuous detection
├── RUN_AUTONOMY.bat      starts backend, dashboard, simulator and the patrol
└── models/               trained policy and reports
```

## How it works

**What the drone senses** (the same in training and in Unreal):

- 24 depth rays across the front camera's 90° view (nearest obstacle per sector) and 8 rays looking
  12° down (ground rising ahead), up to 25 m.
- Direction and distance to the next waypoint, height above ground (from the lowest rows of the same
  depth image), its own
  speed, and its previous command.

**What it commands:** forward speed (0–5 m/s, never backwards blind), turn rate (±60°/s), climb or
descend (±2 m/s).

**How it learned:** flying in Unreal runs at 10–20 control steps a second, far too slow to learn by
trial and error. So the policy learned in `train_world.py`, a fast world built to match the real
sensor and dynamics:

- **Terrain:** rolling hills, which it must follow at 7 m above ground (below the canopy, so trees
  are real obstacles).
- **Trees:** random forests each episode, from open ground to dense woods, with trunks and canopies
  0.3–2.5 m wide and 6–22 m tall.
- **Realism:** sensor noise, missing depth readings, and the drone's response lag are all randomised.

PPO (proximal policy optimisation, `stable-baselines3`) trained a 256×256 network over 10 parallel
worlds:

- **Rewarded:** progress toward the goal and reaching it.
- **Penalised:** flying near obstacles, leaving the target height, jerky commands, and above all
  crashing.

**Safety shield:** after the policy, `nav_core.safety_shield` overrides it if needed, during training
too (each intervention costs the policy a little reward, so it learns to keep its own distance).

- **Corridor:** it keeps a 3 m-wide corridor along the flight path clear for the drone's full
  stopping distance. That distance comes from its current speed, its response lag and how old the
  depth image is.
- **Slow, stop, turn:** it slows down when the corridor gets short, and stops and turns toward open
  space when it runs out.
- **Obstacle memory:** a short-term map of recently seen trunks stops the drone when its momentum
  carries it toward one that has left the camera's view.
- **Ground:** it climbs when the drone is less than 3 m above ground.

A learned policy can make mistakes; the shield is the guarantee.

**Timing in Unreal:**

- **Depth thread:** the simulator renders camera requests one after another, about 0.2 s each. So a
  sensor thread keeps fetching depth images.
- **Control loop:** the loop runs at 7–10 Hz on the latest image. It first shifts that image's
  obstacles to the drone's current position and heading.
- **Stale image:** with no fresh image for 1 s, the drone holds position rather than fly blind.

**Surveillance never stops:** a separate thread streams the front-right camera (thermal + colour,
with telemetry) to the HawkEye backend, independently of navigation. The dashboard shows it as usual,
and the event log records it.

- **While patrolling:** the camera turns to keep the centre of the patrol area in view.
- **Possible person:** one detection makes the drone hover for 2.5 s and look again. A single-frame
  detection that does not repeat is released, and the patrol continues.
- **Confirmed person:** seen in at least 2 of the last 3 frames (`investigate.py`).
  1. The drone works out where the person stands, from the box in the image, the camera angle and
     its height.
  2. It turns to face them, closes to 14 m and zooms in, so they are about 180 px tall: close
     enough for the weapon detector.
  3. It circles them slowly to see them from different angles. Forward motion still goes through
     the safety shield. When a tree blocks the circling direction, it reverses.
  4. After 45 s, or 10 s without seeing them, it resumes the patrol.
- **Backend down:** the patrol keeps flying, and detection resumes when the backend is back.

**Height above ground:** read from the front depth image every frame and from the downward camera
every third frame. The lower of the two is used. Flying downhill, the front image alone reads high.

## Running

```
RUN_AUTONOMY.bat                       everything: backend, dashboard, simulator, patrol
python autonomous_patrol.py            just the patrol (simulator playing, backend running)
python autonomous_patrol.py --minutes 5    timed run, then models/patrol_report.json
python train_policy.py                 retrain the policy (~30-60 min on 10 CPU cores)
python train_policy.py --evaluate      re-evaluate the saved policy
python test_offline.py --latency 3     the real Pilot code against a fake AirSim (no Unreal needed)
```

Requirements beyond the main project: `pip install stable-baselines3==2.4.1 gymnasium==1.0.0`.

## Results

See `models/training_report.json` (training world) and `models/patrol_report.json` (Unreal).

Latest numbers:

| Test | Result |
|---|---|
| Training world, 60 unseen forests (`training_report.json`) | 5% crash rate, 7.9 goals per episode (flying straight at the goal: 2.2) |
| Real `Pilot` code on a fake AirSim, depth images 0.3 s late, 8 forests | 1 crash, in the densest forest (0.016 trees/m²) |
| Unreal, 5-minute patrol (`patrol_report.json`) | 0 collisions, 13 waypoints (2 laps), control at 6.8 Hz, depth images 0.17 s old on average |
| Unreal surveillance | 155/155 frames analysed. The soldier with his rifle was detected (High); 5 confirmed sightings; 12 one-frame detections re-checked and released |
