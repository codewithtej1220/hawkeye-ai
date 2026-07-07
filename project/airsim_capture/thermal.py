
import airsim
import cv2
import numpy as np
import os
import time
import threading
from pynput import keyboard

# =========================================================
# HAWKEYE THERMAL SYSTEM
# AUTO-DETECT BLUE WEAPON VERSION
# =========================================================

print("═" * 60)

CURRENT_ANIMATION_NAME = input(
    "👉 Enter folder name for this capture session: "
).strip()

if not CURRENT_ANIMATION_NAME:
    CURRENT_ANIMATION_NAME = "manual_session"

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

TARGET_WIDTH = 640
TARGET_HEIGHT = 480

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

capture_counter = 0

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

def background_image_pipeline():

    global latest_rgb_frame
    global latest_thermal_frame
    global trigger_capture_flag
    global capture_counter
    global is_running

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

        print("Detected IDs:", unique_ids)

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

        print(
            "Detected Weapon ID:",
            weapon_id_detected
        )

        print(
            "Weapon Pixels:",
            np.count_nonzero(weapon_mask)
        )

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

        thermal[
            weapon_mask > 0
        ] = [255, 0, 0]

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
                    (640, 360)
                )
            )

        if local_thermal is not None:

            cv2.imshow(
                "HawkEye Thermal",
                cv2.resize(
                    local_thermal,
                    (640, 360)
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

