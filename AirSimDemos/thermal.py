"""HawkEye drone capture: flies the AirSim drone, renders its camera as a thermal image, and
streams RGB + thermal frames with drone telemetry to the HawkEye backend.

    python thermal.py              manual flight (keys below)
    python thermal.py --auto       hands-free: the drone orbits the soldier by itself
    python thermal.py --auto --radius 40 --altitude 30 --no-windows

Keys (manual flight; click the Unreal window first):
    W / S        forward / backward        A / D     turn left / right
    ARROWS       strafe left / right       SPACE / SHIFT   up / down
    R / F        camera tilt up / down     Z / X     zoom in / out (optical zoom)
    G            find the soldier: turn the drone and tilt the camera onto him
    T            save an RGB + thermal frame pair to HawkEye_Manual_Dataset
    ESC          land and exit

Environment variables:
    HAWKEYE_BACKEND_URL     backend address (default http://127.0.0.1:8000)
    HAWKEYE_METHOD          auto | rgb | thermal_grey | thermal_inferno | rgbt (default auto)
    HAWKEYE_THERMAL_PALETTE inferno (default) | greyscale
    HAWKEYE_SEND_RGB        1 (default) | 0: also send the colour frame (RGB and RGBT fusion)
    HAWKEYE_SOLDIER         auto (default) | off: run the soldier's patrol routine (sim_actor.py)
    HAWKEYE_WEAPON_STYLE    realistic (default with the real-data model) | blue (legacy model)
"""
import argparse
import base64
import datetime
import math
import os
import sys
import threading
import time

import airsim
import cv2
import numpy as np
import requests

from sim_actor import SoldierDirector
from sim_common import CAMERA_FOV_DEG, ParallelCapture, find_actor, setup_segmentation, telemetry, thermal_from

# Windows consoles default to a legacy code page; status messages must not make printing fail
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--auto", action="store_true", help="orbit the soldier automatically (no keyboard needed)")
parser.add_argument("--radius", type=float, default=30.0, help="--auto: orbit radius in metres")
parser.add_argument("--altitude", type=float, default=22.0, help="--auto: height above the soldier in metres")
parser.add_argument("--orbit-period", type=float, default=60.0, help="--auto: seconds per orbit")
parser.add_argument("--fov", type=float, default=CAMERA_FOV_DEG, help="starting camera field of view (degrees)")
parser.add_argument("--no-windows", action="store_true", help="do not open the RGB / thermal preview windows")
parser.add_argument("--duration", type=float, default=0, help="stop after this many seconds (0 = until ESC)")
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
# 127.0.0.1, not "localhost": on Windows "localhost" tries IPv6 first and every request waits ~2 s
BACKEND_URL = os.environ.get("HAWKEYE_BACKEND_URL", "http://127.0.0.1:8000")
SEND_TO_BACKEND = True
REAL_MODEL_PATH = os.path.join(HERE, "..", "project", "models", "hawkeye_real_yolo11m.pt")
# "realistic": white-hot / INFERNO like a real thermal camera, weapon as cold metal (real-data model).
# "blue": legacy look for the synthetic-data weapon_model.pt (solid blue weapon).
WEAPON_STYLE = os.environ.get("HAWKEYE_WEAPON_STYLE") or ("realistic" if os.path.exists(REAL_MODEL_PATH) else "blue")
THERMAL_PALETTE = os.environ.get("HAWKEYE_THERMAL_PALETTE", "inferno").lower()
SEND_RGB = os.environ.get("HAWKEYE_SEND_RGB", "1") != "0"
DETECTION_METHOD = os.environ.get("HAWKEYE_METHOD", "auto")
RUN_SOLDIER = os.environ.get("HAWKEYE_SOLDIER", "auto").lower() != "off"

# Matches CaptureSettings in airsim_settings.json. Without them AirSim captures only 256x144.
TARGET_WIDTH, TARGET_HEIGHT = 1280, 960
ZOOM_STEPS = [90.0, 60.0, 40.0, 25.0, 15.0]   # camera field of view per zoom level (degrees)
SPEED, YAW_RATE = 4.0, 40.0
ACCEL, YAW_ACCEL = 3.0, 90.0   # m/s^2 and deg/s^2: speed ramps up and down instead of jumping
CONTROL_HZ = 20                # flight commands per second, at a steady rate
# Flight commands last longer than the loop period, so AirSim never runs out of commands between two
# loop passes ("API call was not received, entering hover mode for safety")
COMMAND_S = 0.3
GIMBAL_HZ = 10                 # camera stabilisation updates per second
# Frames are captured no faster than this. The dashboard shows analysed frames (~1 per second), and
# every capture costs the simulator two extra renders at 1280x960: capturing flat out made Unreal,
# and so the drone's flight, stutter.
CAPTURE_FPS = 3
ANGLE_STEP, GIMBAL_DEBOUNCE_DELAY = 20.0, 0.20

SESSION = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
RGB_FOLDER = os.path.join(HERE, "HawkEye_Manual_Dataset", "rgb", SESSION)
THERMAL_FOLDER = os.path.join(HERE, "HawkEye_Manual_Dataset", "thermal", SESSION)

print("=" * 60)
print(f"HawkEye capture | mode: {'AUTO orbit' if args.auto else 'manual flight'} | thermal: {THERMAL_PALETTE} "
      f"| RGB: {SEND_RGB} | method: {DETECTION_METHOD}")

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

state = {
    "pitch": -20.0,          # camera tilt (degrees, negative = down)
    "zoom": min(range(len(ZOOM_STEPS)), key=lambda i: abs(ZOOM_STEPS[i] - args.fov)),
    "frame_id": 0,
    "capture_counter": 0,
    "save_pair": False,
    "running": True,
    "rgb": None,
    "thermal": None,
    "last_gimbal": 0.0,
    "weapon_id": None,
}
active_keys = set()
frame_lock = threading.Lock()
send_lock = threading.Lock()
send_event = threading.Event()
pending = {"body": None}


def fov():
    return ZOOM_STEPS[state["zoom"]]


def apply_camera(c=client, set_fov=True):
    """Point the camera like a stabilised gimbal: state["pitch"] is its tilt relative to the horizon.
    The camera is mounted on the drone body, which tilts whenever it speeds up or turns; without
    compensation the view swings with every manoeuvre (and the dashboard's box placement, which uses
    the reported tilt, would be off). So the body's pitch and roll are subtracted each loop."""
    state["pitch"] = max(-90.0, min(0.0, state["pitch"]))
    body_pitch, body_roll, _ = airsim.to_eularian_angles(c.simGetVehiclePose().orientation)
    c.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0),
                                        airsim.to_quaternion(math.radians(state["pitch"]) - body_pitch, -body_roll, 0)))
    if set_fov:
        c.simSetCameraFov("0", fov())


def environment_setup():
    try:
        client.simEnableWeather(True)
        client.simSetTimeOfDay(True)
        # t.IdleWhenNotForeground 0: keep the simulation at full speed while another window
        # (the dashboard) has the focus; Unreal otherwise drops to a few frames per second
        for command in ("r.TonemapperFilm 0", "r.Tonemapper.Gamma 0", "show Fog 0", "r.Fog 0",
                        "t.IdleWhenNotForeground 0",
                        # hide on-screen debug text (blueprint "ai path finding failed" notes, the
                        # "lighting needs to be rebuilt" banner): it would show in a demo
                        "DisableAllScreenMessages"):
            client.simRunConsoleCommand(command)
    except Exception as exc:
        print(f"Environment setup skipped: {exc}")


def backend_sender():
    """Post the newest frame to the backend. Runs apart from capture, so a slow detection (the
    backend shares the GPU with Unreal) never stalls the camera windows."""
    sent = ok = 0
    backend_up = None
    while state["running"]:
        if not send_event.wait(timeout=0.5):
            continue
        with send_lock:
            body, pending["body"] = pending["body"], None
            send_event.clear()
        if body is None:
            continue
        # Frames are encoded here, only for the frames actually sent (detection takes ~1 s, so most
        # captured frames are only shown in the local windows)
        body = {k: (jpeg(v, 88) if k in ("image", "rgb") else v) for k, v in body.items()}
        try:
            response = requests.post(f"{BACKEND_URL}/detect", json=body, timeout=10)
            sent += 1
            if backend_up is not True:
                print(f"Streaming to backend {BACKEND_URL} (dashboard: http://localhost:5173)")
                backend_up = True
            if response.status_code == 200:
                ok += 1
            else:
                print(f"Backend returned {response.status_code}: {response.text[:200]}")
            if sent % 50 == 0:
                print(f"{sent} frames sent, {ok} analysed")
        except requests.exceptions.RequestException as exc:
            if backend_up is not False:
                print(f"Backend not reachable at {BACKEND_URL} ({type(exc).__name__}); retrying...")
                backend_up = False
            time.sleep(1.0)


def capture_loop():
    """Capture thread. An error (e.g. the simulator hiccups) is reported and the loop restarts,
    instead of the thread dying while the camera windows stay open."""
    while state["running"]:
        try:
            _capture_frames()
        except Exception as exc:
            print(f"Capture error: {type(exc).__name__}: {exc}. Restarting capture in 1 s...")
            time.sleep(1.0)


def _capture_frames():
    worker = airsim.MultirotorClient()
    worker.confirmConnection()
    camera = ParallelCapture(TARGET_WIDTH, TARGET_HEIGHT)
    target = find_actor(worker, "ThirdPersonCharacter")
    # Altitude is reported above the drone's spawn (take-off) point, the origin of AirSim's coordinates,
    # not above wherever this capture happened to start (the drone may already be flying)
    home_z = 0.0
    # The next pair is captured while this one is rendered and sent (capture ~140 ms, render ~90 ms)
    latest = {"pair": None}
    ready = threading.Event()

    def grab():
        while state["running"]:
            started = time.time()
            pair = camera.pair()
            if pair is not None:
                latest["pair"] = pair
                ready.set()
            time.sleep(max(0.0, 1.0 / CAPTURE_FPS - (time.time() - started)))

    grabber = threading.Thread(target=grab, daemon=True)
    grabber.start()
    while state["running"]:
        if not grabber.is_alive():
            raise RuntimeError("camera capture stopped")
        if not ready.wait(timeout=1.0):
            continue
        ready.clear()
        rgb, seg = latest["pair"]
        thermal, body_mask, weapon_mask, weapon_id = thermal_from(rgb, seg, "inferno" if WEAPON_STYLE == "blue" else THERMAL_PALETTE)
        if WEAPON_STYLE == "blue":
            thermal[weapon_mask > 0] = [255, 0, 0]
        if weapon_id != state["weapon_id"]:
            state["weapon_id"] = weapon_id
            print(f"Weapon in view: {'yes' if weapon_id is not None else 'no'} ({int(np.count_nonzero(weapon_mask))} px)")

        if SEND_TO_BACKEND:
            state["frame_id"] += 1
            data = None
            try:
                data = telemetry(worker, state["pitch"], target, home_z)
                data["camera_fov_deg"] = fov()
                data["zoom"] = round(CAMERA_FOV_DEG / fov(), 1)
                data["mode"] = "auto orbit" if args.auto else "manual"
                if soldier is not None:
                    data["target_action"] = soldier.action
                    data["patrol_north_m"], data["patrol_east_m"] = round(soldier.home[0], 1), round(soldier.home[1], 1)
                state["target_pos"] = (data.get("target_north_m"), data.get("target_east_m"),
                                       data.get("target_height_above_m"))
            except Exception:
                pass
            send_rgb = SEND_RGB and WEAPON_STYLE != "blue"
            # Thermal + RGB of the same view for detection: the backend picks RGB, thermal or RGBT
            # fusion for the conditions. A newer frame replaces one not yet sent.
            body = {"image": thermal, "frame_id": state["frame_id"], "method": DETECTION_METHOD,
                    "telemetry": data}  # images are encoded by the sender, only if this frame is sent
            if send_rgb:
                body["rgb"] = rgb
            with send_lock:
                pending["body"] = body
            send_event.set()

        with frame_lock:
            state["rgb"], state["thermal"] = rgb, thermal
            if state["save_pair"]:
                os.makedirs(RGB_FOLDER, exist_ok=True)
                os.makedirs(THERMAL_FOLDER, exist_ok=True)
                n = state["capture_counter"]
                cv2.imwrite(os.path.join(RGB_FOLDER, f"manual_{n:04d}.png"), rgb)
                cv2.imwrite(os.path.join(THERMAL_FOLDER, f"manual_{n:04d}.png"), thermal)
                print(f"Saved frame pair {n:04d}")
                state["capture_counter"], state["save_pair"] = n + 1, False
        time.sleep(0.01)


def jpeg(image, quality):
    return base64.b64encode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]).decode("utf-8")


def look_at_soldier():
    """G key: turn the drone toward the soldier and tilt the camera down onto him (sim only)."""
    target = state.get("target_pos")
    if not target or target[0] is None:
        print("Soldier position not known yet.")
        return
    pos = client.getMultirotorState().kinematics_estimated.position
    dn, de = target[0] - pos.x_val, target[1] - pos.y_val
    yaw = math.degrees(math.atan2(de, dn))
    horizontal = math.hypot(dn, de)
    # target_height_above_m is positive when he is below the drone (NED: z grows downward)
    state["pitch"] = -math.degrees(math.atan2(max(0.0, target[2] or 0.0), max(horizontal, 0.1)))
    state["yaw_target"] = yaw
    state["camera_dirty"] = True
    print(f"Soldier: {math.hypot(horizontal, target[2] or 0):.0f} m away, bearing {yaw % 360:.0f} deg; turning toward him")


def on_press(key):
    now = time.time()
    try:
        if hasattr(key, "char") and key.char:
            k = key.char.lower()
            active_keys.add(k)
            if k in "rfzx" and now - state["last_gimbal"] > GIMBAL_DEBOUNCE_DELAY:
                if k == "r":
                    state["pitch"] += ANGLE_STEP
                elif k == "f":
                    state["pitch"] -= ANGLE_STEP
                elif k == "z":
                    state["zoom"] = min(len(ZOOM_STEPS) - 1, state["zoom"] + 1)
                elif k == "x":
                    state["zoom"] = max(0, state["zoom"] - 1)
                state["camera_dirty"] = True
                state["last_gimbal"] = now
                print(f"Camera tilt {state['pitch']:.0f} deg | zoom {CAMERA_FOV_DEG / fov():.1f}x (FOV {fov():.0f} deg)")
            elif k == "t":
                with frame_lock:
                    state["save_pair"] = True
            elif k == "g":
                look_at_soldier()
        else:
            active_keys.add(key)
    except Exception:
        pass


def on_release(key):
    from pynput import keyboard
    try:
        if hasattr(key, "char") and key.char:
            active_keys.discard(key.char.lower())
        else:
            active_keys.discard(key)
        if key == keyboard.Key.esc:
            return False
    except Exception:
        pass


def orbit_step(t0, dt):
    """--auto: fly a steady circle around the soldier, nose and camera on him. The circle's centre
    follows the soldier slowly (he walks, and is occasionally moved back to his post), so the drone
    does not lurch when he moves; the velocity is the circle's own (tangential) speed plus a gentle
    correction toward the circle, ramped like a real drone's."""
    target = client.simGetObjectPose(orbit_target).position
    if math.isnan(target.x_val):
        return
    centre = state.setdefault("orbit_centre", [target.x_val, target.y_val, target.z_val])
    k = min(1.0, dt / 4.0)  # ~4 s time constant
    for axis, value in enumerate((target.x_val, target.y_val, target.z_val)):
        centre[axis] += (value - centre[axis]) * k
    pos = client.getMultirotorState().kinematics_estimated.position
    omega = 2 * math.pi / args.orbit_period
    angle = omega * (time.time() - t0)
    goal = (centre[0] + args.radius * math.cos(angle), centre[1] + args.radius * math.sin(angle),
            centre[2] - args.altitude)
    tangential = (-args.radius * omega * math.sin(angle), args.radius * omega * math.cos(angle), 0.0)
    wanted = [t + 0.4 * (g - c) for t, g, c in zip(tangential, goal, (pos.x_val, pos.y_val, pos.z_val))]
    norm = math.sqrt(sum(x * x for x in wanted))
    if norm > 6.0:
        wanted = [x * 6.0 / norm for x in wanted]
    v = ramp(state.setdefault("orbit_v", [0.0, 0.0, 0.0]), wanted, ACCEL * dt)
    yaw = math.degrees(math.atan2(centre[1] - pos.y_val, centre[0] - pos.x_val))
    client.moveByVelocityAsync(v[0], v[1], v[2], duration=COMMAND_S, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                               yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw))
    horizontal = math.hypot(centre[0] - pos.x_val, centre[1] - pos.y_val)
    state["pitch"] = -math.degrees(math.atan2(max(0.0, centre[2] - pos.z_val), max(horizontal, 0.1)))


def ramp(current, wanted, step):
    """Move each component of `current` toward `wanted` by at most `step` (in place); returns it."""
    for i, w in enumerate(wanted):
        current[i] += max(-step, min(step, w - current[i]))
    return current


environment_setup()
setup_segmentation(client)
apply_camera()

soldier = None
if RUN_SOLDIER:
    try:
        soldier = SoldierDirector(verbose=True).start()
    except Exception as exc:
        print(f"Soldier routine not started: {exc}")

threading.Thread(target=capture_loop, daemon=True).start()
if SEND_TO_BACKEND:
    threading.Thread(target=backend_sender, daemon=True).start()

listener = None
orbit_target = find_actor(client, "ThirdPersonCharacter") if args.auto else None
if args.auto and orbit_target is None:
    sys.exit("--auto needs the soldier (ThirdPersonCharacter) in the scene.")
if not args.auto:
    from pynput import keyboard
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print(__doc__.split("Keys")[1].split("Environment")[0].replace("(manual flight; click the Unreal window first):", "Controls:"))
else:
    print(f"Taking off, then orbiting the soldier: radius {args.radius:.0f} m, {args.altitude:.0f} m above him. Ctrl+C to stop.")
    client.takeoffAsync().join()

started = time.time()
last_tick = last_gimbal = time.time()
manual = {"v": [0.0, 0.0, 0.0], "yaw": [0.0]}
try:
    while (listener is None or listener.running) and (not args.duration or time.time() - started < args.duration):
        now = time.time()
        dt, last_tick = max(1e-3, now - last_tick), now
        if args.auto:
            orbit_step(started, dt)
        else:
            from pynput import keyboard
            wanted = [SPEED * (("w" in active_keys) - ("s" in active_keys)),
                      SPEED * ((keyboard.Key.right in active_keys) - (keyboard.Key.left in active_keys)),
                      SPEED * ((keyboard.Key.shift in active_keys) - (keyboard.Key.space in active_keys))]
            vx, vy, vz = ramp(manual["v"], wanted, ACCEL * dt)  # smooth start and stop
            yaw_speed = ramp(manual["yaw"], [YAW_RATE * (("d" in active_keys) - ("a" in active_keys))], YAW_ACCEL * dt)[0]
            if abs(yaw_speed) < 0.5:
                yaw_speed = 0.0
            yaw_target = state.pop("yaw_target", None) if not yaw_speed else None
            if yaw_target is not None:  # G: turn to face the soldier
                yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=yaw_target)
                state["turning_until"] = time.time() + 1.5
                state["turn_yaw"] = yaw_target
            elif not yaw_speed and time.time() < state.get("turning_until", 0):
                yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=state["turn_yaw"])
            else:
                yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_speed)
            client.moveByVelocityBodyFrameAsync(vx, vy, vz, duration=COMMAND_S,
                                                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                yaw_mode=yaw_mode)
            if state.get("camera_dirty"):
                apply_camera(set_fov=state.pop("camera_dirty"))

        if not args.no_windows:
            with frame_lock:
                rgb, thermal = state["rgb"], state["thermal"]
            if rgb is not None:
                cv2.imshow("HawkEye RGB", cv2.resize(rgb, (640, 480)))
            if thermal is not None:
                cv2.imshow("HawkEye Thermal", cv2.resize(thermal, (640, 480)))
            cv2.waitKey(1)
        if now - last_gimbal >= 1.0 / GIMBAL_HZ:  # gimbal stabilisation
            apply_camera(set_fov=False)
            last_gimbal = now
        time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.time() - now)))
except KeyboardInterrupt:
    pass
finally:
    print("\nShutting down safely...")
    state["running"] = False
    if soldier is not None:
        soldier.stop()
    if listener is not None:
        listener.stop()
    cv2.destroyAllWindows()
    try:
        client.simSetCameraFov("0", CAMERA_FOV_DEG)
        if not args.auto:
            client.hoverAsync().join()
            client.landAsync().join()
        client.enableApiControl(False)
    except Exception:
        pass
    print("Done.")
