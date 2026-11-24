
import airsim
import cv2
import numpy as np
import os
import time
import threading
from pynput import keyboard
import requests
import base64
import sys

# Windows consoles default to a legacy code page; the emoji in the status messages below would
# otherwise make printing fail or stall
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# =========================================================
# HAWKEYE THERMAL SYSTEM
# AUTO-DETECT BLUE WEAPON VERSION
# =========================================================

print("═" * 60)

# Auto-generate session name with timestamp
import datetime
CURRENT_ANIMATION_NAME = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
print(f"📁 Session folder: {CURRENT_ANIMATION_NAME}")

# Backend API Configuration
# 127.0.0.1, not "localhost": on Windows "localhost" tries IPv6 first and every request waits ~2 s
# before falling back to the backend's IPv4 address
BACKEND_URL = os.environ.get("HAWKEYE_BACKEND_URL", "http://127.0.0.1:8000")
SEND_TO_BACKEND = True  # Set to False to disable backend communication

# Thermal rendering style:
#   "realistic" - white-hot greyscale like a real thermal camera (FLIR etc.): warm body
#                 bright with a soft heat gradient, ambient terrain mid-grey, weapons dark
#                 cold metal. Matches the real-data model hawkeye_real_yolo11m.pt.
#   "blue"      - legacy INFERNO (orange body) + solid-blue weapon that the synthetic-data
#                 weapon_model.pt was trained on
# Defaults to whichever matches the model the backend will load.
REAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "project", "models", "hawkeye_real_yolo11m.pt")
WEAPON_STYLE = os.environ.get("HAWKEYE_WEAPON_STYLE") or ("realistic" if os.path.exists(REAL_MODEL_PATH) else "blue")
print(f"🌡️ Thermal render style: {WEAPON_STYLE}")

# Thermal palette for the realistic render: "inferno" (preferred) or "greyscale" (white-hot)
THERMAL_PALETTE = os.environ.get("HAWKEYE_THERMAL_PALETTE", "inferno").lower()
# Also send the RGB camera frame so the backend can use RGB or RGBT sensor fusion
SEND_RGB = os.environ.get("HAWKEYE_SEND_RGB", "1") != "0"
# auto (environment-based choice) | rgb | thermal_grey | thermal_inferno | rgbt
DETECTION_METHOD = os.environ.get("HAWKEYE_METHOD", "auto")
print(f"🎨 Thermal palette: {THERMAL_PALETTE} | sending RGB: {SEND_RGB} | method: {DETECTION_METHOD}")

# White-hot grey levels (0-255). Tuned on captured frames against hawkeye_real_yolo11m.pt.
WHITEHOT_BG_BASE = 110       # mean ambient terrain level
WHITEHOT_BG_CONTRAST = 0.2   # terrain texture strength (real thermal backgrounds are low contrast)
WHITEHOT_SKIN_CORE = 225     # warmest part of the body
WHITEHOT_SKIN_EDGE = 185     # body outline (clothing / limbs are cooler)
WHITEHOT_WEAPON = 45         # cold metal, darker than terrain
_noise_rng = np.random.default_rng()


def render_whitehot(gray, body_mask, weapon_mask):
    """Render the AirSim scene as a white-hot thermal image (3-channel greyscale)."""
    # Background: low-contrast ambient terrain, softened like a microbolometer sensor
    bg = cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32)
    out = WHITEHOT_BG_BASE + WHITEHOT_BG_CONTRAST * (bg - 127.5)

    if np.any(body_mask):
        # Warm core fading to cooler outline, with clothing variation from scene shading
        dist = cv2.distanceTransform(body_mask, cv2.DIST_L2, 5)
        core = np.clip(dist / max(dist.max() * 0.5, 1.0), 0, 1)
        shading = (gray.astype(np.float32) / 255.0 - 0.5) * 24
        heat = WHITEHOT_SKIN_EDGE + (WHITEHOT_SKIN_CORE - WHITEHOT_SKIN_EDGE) * core + shading
        # Heat halo bleeding a couple of pixels into the surroundings
        halo = cv2.GaussianBlur((body_mask > 0).astype(np.float32), (0, 0), 2.0)
        warm = np.where(body_mask > 0, heat, out + (WHITEHOT_SKIN_EDGE - out) * halo * 0.6)
        out = np.maximum(out, warm)

    if np.any(weapon_mask):
        w = cv2.GaussianBlur((weapon_mask > 0).astype(np.float32), (0, 0), 0.8)
        out = out * (1 - w) + WHITEHOT_WEAPON * w

    # Optics blur + sensor noise
    out = cv2.GaussianBlur(out, (0, 0), 0.9)
    out += _noise_rng.normal(0, 2.0, out.shape)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

print("═" * 60)

# =========================================================
# DATASET PATHS
# =========================================================

BASE_FOLDER = "HawkEye_Manual_Dataset"

RGB_FOLDER = os.path.join(
    BASE_FOLDER,
    "rgb",
    CURRENT_ANIMATION_NAME
)

THERMAL_FOLDER = os.path.join(
    BASE_FOLDER,
    "thermal",
    CURRENT_ANIMATION_NAME
)

os.makedirs(RGB_FOLDER, exist_ok=True)
os.makedirs(THERMAL_FOLDER, exist_ok=True)

# =========================================================
# CAMERA SETTINGS
# =========================================================

# Matches CaptureSettings in airsim_settings.json (copy it to Documents\AirSim\settings.json).
# Without them AirSim captures only 256x144, and a rifle seen from the air is a 2-3 px smudge.
TARGET_WIDTH = 1280
TARGET_HEIGHT = 960

# =========================================================
# AIRSIM
# =========================================================

client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True)
client.armDisarm(True)

# =========================================================
# GLOBALS
# =========================================================

BODY_ID = 255

current_camera_pitch = -20.0

capture_counter = 0     # frames saved to disk with T
stream_frame_id = 0     # every frame streamed to the backend

active_keys = set()

last_gimbal_press_time = 0.0

GIMBAL_DEBOUNCE_DELAY = 0.20
ANGLE_STEP = 20.0

frame_lock = threading.Lock()

latest_rgb_frame = None
latest_thermal_frame = None

trigger_capture_flag = False

is_running = True

# =========================================================
# ENVIRONMENT SETUP
# =========================================================

def stable_environment_setup():

    print("\n⏰ Synchronizing environment...")

    try:

        client.simEnableWeather(True)

        client.simSetTimeOfDay(True)

        client.simRunConsoleCommand("r.TonemapperFilm 0")
        client.simRunConsoleCommand("r.Tonemapper.Gamma 0")

        client.simRunConsoleCommand("show Fog 0")
        client.simRunConsoleCommand("r.Fog 0")

        print("☀️ Environment Ready!")

    except Exception as e:

        print(f"⚠️ Setup skipped: {e}")

# =========================================================
# SEGMENTATION SETUP
# =========================================================

def setup_segmentation():

    print("\n🎯 ASSIGNING SEGMENTATION IDS...\n")

    try:

        # RESET ALL
        client.simSetSegmentationObjectID(
            ".*",
            0,
            True
        )

        # CHARACTER BODY
        client.simSetSegmentationObjectID(
            ".*ThirdPersonCharacter.*",
            BODY_ID,
            True
        )

        # WEAPON
        client.simSetSegmentationObjectID(
            ".*BP_Weapon.*",
            85,
            True
        )

        print("✅ Segmentation assignment complete!")

    except Exception as e:

        print(e)

# =========================================================
# CAMERA GIMBAL
# =========================================================

def update_camera_gimbal():

    global current_camera_pitch

    current_camera_pitch = max(
        -90.0,
        min(0.0, current_camera_pitch)
    )

    q = airsim.to_quaternion(
        np.radians(current_camera_pitch),
        0,
        0
    )

    client.simSetCameraPose(
        "0",
        airsim.Pose(
            airsim.Vector3r(0, 0, 0),
            q
        )
    )

# =========================================================
# IMAGE PIPELINE
# =========================================================

last_weapon_id = None
pending_backend_frame = None
send_lock = threading.Lock()
send_event = threading.Event()


def backend_sender():
    """Post the newest frame to the backend. Runs apart from capture, so a slow detection (the
    backend shares the GPU with Unreal) never stalls the camera windows."""
    global pending_backend_frame
    sent = ok = 0
    backend_up = None
    while is_running:
        if not send_event.wait(timeout=0.5):
            continue
        with send_lock:
            body, pending_backend_frame = pending_backend_frame, None
            send_event.clear()
        if body is None:
            continue
        try:
            response = requests.post(f"{BACKEND_URL}/detect", json=body, timeout=10)
            sent += 1
            if backend_up is not True:
                print(f"📡 Streaming to backend {BACKEND_URL} (dashboard: http://localhost:5173)")
                backend_up = True
            if response.status_code == 200:
                ok += 1
                detections = response.json().get("detections", 0)
                if detections > 0:
                    print(f"🎯 Backend detected: {detections} objects")
            else:
                print(f"⚠️ Backend returned {response.status_code}: {response.text[:200]}")
            if sent % 50 == 0:
                print(f"📡 {sent} frames sent, {ok} analysed")
        except requests.exceptions.RequestException as exc:
            if backend_up is not False:
                print(f"⚠️ Backend not reachable at {BACKEND_URL} ({type(exc).__name__}); retrying...")
                backend_up = False
            time.sleep(1.0)


def background_image_pipeline():
    """Capture loop. An error (e.g. the PC briefly running out of memory) is reported and the
    loop restarts, instead of the capture thread dying while the camera windows stay open."""
    while is_running:
        try:
            _image_pipeline_loop()
        except Exception as exc:
            print(f"⚠️ Capture error: {type(exc).__name__}: {exc}. Restarting capture in 1 s...")
            time.sleep(1.0)


def _image_pipeline_loop():

    global latest_rgb_frame
    global latest_thermal_frame
    global trigger_capture_flag
    global capture_counter
    global stream_frame_id
    global is_running
    global last_weapon_id
    global pending_backend_frame

    worker_client = airsim.MultirotorClient()
    worker_client.confirmConnection()

    while is_running:

        try:

            responses = worker_client.simGetImages([

                airsim.ImageRequest(
                    "0",
                    airsim.ImageType.Scene,
                    False,
                    False
                ),

                airsim.ImageRequest(
                    "0",
                    airsim.ImageType.Segmentation,
                    False,
                    False
                )

            ])

        except Exception:

            time.sleep(0.01)
            continue

        if len(responses) < 2:
            continue

        if not responses[0].image_data_uint8:
            continue

        if not responses[1].image_data_uint8:
            continue

        resp_rgb = responses[0]
        resp_seg = responses[1]

        try:

            rgb_raw = np.frombuffer(
                resp_rgb.image_data_uint8,
                dtype=np.uint8
            )

            seg_raw = np.frombuffer(
                resp_seg.image_data_uint8,
                dtype=np.uint8
            )

            rgb = rgb_raw.reshape(
                resp_rgb.height,
                resp_rgb.width,
                3
            )

            seg = seg_raw.reshape(
                resp_seg.height,
                resp_seg.width,
                3
            )

        except Exception:
            continue

        # =====================================================
        # RESIZE - Ensure both images are same size
        # =====================================================

        if rgb.shape[1] != TARGET_WIDTH or rgb.shape[0] != TARGET_HEIGHT:
            rgb = cv2.resize(
                rgb,
                (TARGET_WIDTH, TARGET_HEIGHT)
            )

        if seg.shape[1] != TARGET_WIDTH or seg.shape[0] != TARGET_HEIGHT:
            seg = cv2.resize(
                seg,
                (TARGET_WIDTH, TARGET_HEIGHT),
                interpolation=cv2.INTER_NEAREST
            )

        # =====================================================
        # FIX RGB
        # =====================================================

        if rgb.shape[2] == 4:

            rgb = cv2.cvtColor(
                rgb,
                cv2.COLOR_BGRA2BGR
            )

        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_BGR2GRAY
        )

        # =====================================================
        # SEGMENTATION
        # =====================================================

        stencil = seg[:, :, 2]

        unique_ids = np.unique(stencil)

        # BODY MASK
        body_mask = cv2.inRange(
            stencil,
            BODY_ID,
            BODY_ID
        )

        # =====================================================
        # AUTO DETECT WEAPON ID
        # =====================================================

        weapon_id_detected = None

        for uid in unique_ids:

            if uid != 0 and uid != BODY_ID:

                weapon_id_detected = int(uid)
                break

        weapon_mask = np.zeros_like(
            stencil,
            dtype=np.uint8
        )

        if weapon_id_detected is not None:

            weapon_mask = cv2.inRange(
                stencil,
                weapon_id_detected,
                weapon_id_detected
            )

        # Printing every frame floods the console and slows capture; report changes only
        if weapon_id_detected != last_weapon_id:
            print(f"🔫 Weapon segmentation ID: {weapon_id_detected} ({np.count_nonzero(weapon_mask)} px)")
            last_weapon_id = weapon_id_detected

        # =====================================================
        # BODY THERMAL
        # =====================================================

        thermal_base = np.zeros_like(
            gray,
            dtype=np.float32
        )

        if np.any(body_mask):

            dist_transform = cv2.distanceTransform(
                body_mask,
                cv2.DIST_L2,
                5
            )

            max_dist = np.max(dist_transform)

            if max_dist == 0:
                max_dist = 1.0

            normalized = dist_transform / max_dist

            sigmoid = 1.0 / (
                1.0 +
                np.exp(
                    -14.0 * (normalized - 0.15)
                )
            )

            # Ensure gray has same dimensions as sigmoid
            if gray.shape != sigmoid.shape:
                gray_resized = cv2.resize(gray, (sigmoid.shape[1], sigmoid.shape[0]))
                luminance = gray_resized.astype(np.float32) / 255.0
            else:
                luminance = gray.astype(np.float32) / 255.0

            shaded = sigmoid * (
                0.40 +
                0.60 * luminance
            )

            thermal_base = (
                shaded * 65.0
            ) + 190.0

            thermal_base[
                body_mask == 0
            ] = 0

        character_heat = np.clip(
            thermal_base,
            0,
            255
        ).astype(np.uint8)

        # =====================================================
        # BACKGROUND
        # =====================================================

        bg_mask = (
            (255 - body_mask) // 255
        ).astype(np.uint8)

        # Ensure bg_mask has the same dimensions as gray
        if len(bg_mask.shape) != len(gray.shape):
            bg_mask = bg_mask.reshape(gray.shape)

        bg_raw = cv2.multiply(
            gray,
            bg_mask
        )

        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )

        enhanced_bg = clahe.apply(bg_raw)

        ambient = cv2.convertScaleAbs(
            enhanced_bg,
            alpha=0.25,
            beta=25
        )

        # Ensure bg_mask is uint8 for multiplication
        bg_mask_uint8 = bg_mask.astype(np.uint8)

        background_cold = cv2.multiply(
            ambient,
            bg_mask_uint8
        )

        combined_gray = cv2.add(
            background_cold,
            character_heat
        )

        # =====================================================
        # BLUR
        # =====================================================

        sharp = cv2.GaussianBlur(
            combined_gray,
            (3, 3),
            0
        )

        bloom = cv2.GaussianBlur(
            combined_gray,
            (7, 7),
            0
        )

        final_gray = cv2.addWeighted(
            sharp,
            0.85,
            bloom,
            0.15,
            0
        )

        # =====================================================
        # THERMAL COLORMAP
        # =====================================================

        thermal = cv2.applyColorMap(
            final_gray,
            cv2.COLORMAP_INFERNO
        )

        # =====================================================
        # BLUE WEAPON
        # =====================================================

        if WEAPON_STYLE == "blue":
            thermal[
                weapon_mask > 0
            ] = [255, 0, 0]
        else:
            thermal = render_whitehot(gray, body_mask, weapon_mask)
            if THERMAL_PALETTE == "inferno":
                thermal = cv2.applyColorMap(cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)

        # =====================================================
        # SEND TO BACKEND FOR AI DETECTION
        # =====================================================

        if SEND_TO_BACKEND:
            # Thermal frame plus the RGB frame of the same camera view: the backend picks RGB,
            # thermal or RGBT fusion depending on conditions (light, fog, thermal contrast).
            # The sender thread posts it; a newer frame replaces one not yet sent (no queue build-up).
            _, buffer = cv2.imencode('.jpg', thermal, [cv2.IMWRITE_JPEG_QUALITY, 85])
            stream_frame_id += 1
            body = {"image": base64.b64encode(buffer).decode('utf-8'), "frame_id": stream_frame_id,
                    "method": DETECTION_METHOD}
            if SEND_RGB and WEAPON_STYLE != "blue":
                _, rgb_buffer = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                body["rgb"] = base64.b64encode(rgb_buffer).decode('utf-8')
            with send_lock:
                pending_backend_frame = body
            send_event.set()

        # =====================================================
        # SAVE
        # =====================================================

        with frame_lock:

            latest_rgb_frame = rgb.copy()

            latest_thermal_frame = thermal.copy()

            if trigger_capture_flag:

                rgb_file = os.path.join(
                    RGB_FOLDER,
                    f"manual_{capture_counter:04d}.png"
                )

                thermal_file = os.path.join(
                    THERMAL_FOLDER,
                    f"manual_{capture_counter:04d}.png"
                )

                cv2.imwrite(
                    rgb_file,
                    latest_rgb_frame
                )

                cv2.imwrite(
                    thermal_file,
                    latest_thermal_frame
                )

                print(
                    f"📸 Saved Frame Pair: {capture_counter:04d}"
                )

                capture_counter += 1

                trigger_capture_flag = False

        time.sleep(0.01)

# =========================================================
# KEYBOARD
# =========================================================

def on_press(key):

    global current_camera_pitch
    global trigger_capture_flag
    global last_gimbal_press_time

    current_time = time.time()

    try:

        if hasattr(key, 'char') and key.char:

            k = key.char.lower()

            active_keys.add(k)

            if k in ['r', 'f']:

                if (
                    current_time -
                    last_gimbal_press_time
                ) > GIMBAL_DEBOUNCE_DELAY:

                    if k == 'r':
                        current_camera_pitch += ANGLE_STEP

                    if k == 'f':
                        current_camera_pitch -= ANGLE_STEP

                    update_camera_gimbal()

                    print(
                        f"📸 Camera Pitch: {current_camera_pitch}"
                    )

                    last_gimbal_press_time = current_time

            elif k == 't':

                with frame_lock:
                    trigger_capture_flag = True

        else:

            active_keys.add(key)

    except Exception:
        pass

def on_release(key):

    try:

        if hasattr(key, 'char') and key.char:
            active_keys.discard(key.char.lower())
        else:
            active_keys.discard(key)

        if key == keyboard.Key.esc:
            return False

    except Exception:
        pass

# =========================================================
# INITIALIZATION
# =========================================================

stable_environment_setup()

setup_segmentation()

update_camera_gimbal()

pipeline_thread = threading.Thread(
    target=background_image_pipeline,
    daemon=True
)

pipeline_thread.start()

if SEND_TO_BACKEND:
    threading.Thread(target=backend_sender, daemon=True).start()

print("\n🚀 HawkEye Flight Engine Active!\n")

print("👉 Controls:")
print("W / S        : Forward / Backward")
print("A / D        : Yaw Left / Right")
print("ARROWS       : Left / Right")
print("SPACE/SHIFT  : Up / Down")
print("R / F        : Camera Gimbal")
print("T            : Capture Frame")
print("ESC          : Exit")

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release
)

listener.start()

# =========================================================
# FLIGHT SETTINGS
# =========================================================

SPEED = 4.0
YAW_RATE = 40.0

# =========================================================
# MAIN LOOP
# =========================================================

try:

    while listener.running:

        vx = 0.0
        vy = 0.0
        vz = 0.0

        yaw_speed = 0.0

        if 'w' in active_keys:
            vx = SPEED

        if 's' in active_keys:
            vx = -SPEED

        if 'a' in active_keys:
            yaw_speed = -YAW_RATE

        if 'd' in active_keys:
            yaw_speed = YAW_RATE

        if keyboard.Key.left in active_keys:
            vy = -SPEED

        if keyboard.Key.right in active_keys:
            vy = SPEED

        if keyboard.Key.space in active_keys:
            vz = -SPEED

        if keyboard.Key.shift in active_keys:
            vz = SPEED

        client.moveByVelocityBodyFrameAsync(
            vx,
            vy,
            vz,
            duration=0.02,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(
                is_rate=True,
                yaw_or_rate=yaw_speed
            )
        )

        with frame_lock:

            local_rgb = (
                latest_rgb_frame.copy()
                if latest_rgb_frame is not None
                else None
            )

            local_thermal = (
                latest_thermal_frame.copy()
                if latest_thermal_frame is not None
                else None
            )

        if local_rgb is not None:

            cv2.imshow(
                "HawkEye RGB",
                cv2.resize(
                    local_rgb,
                    (640, 480)
                )
            )

        if local_thermal is not None:

            cv2.imshow(
                "HawkEye Thermal",
                cv2.resize(
                    local_thermal,
                    (640, 480)
                )
            )

        cv2.waitKey(1)

        time.sleep(0.01)

finally:

    print("\n🛑 Shutting down safely...")

    is_running = False

    listener.stop()

    pipeline_thread.join(timeout=1.0)

    cv2.destroyAllWindows()

    try:

        client.hoverAsync().join()

        client.landAsync().join()

        client.enableApiControl(False)

    except Exception:
        pass

    print("Done.")

