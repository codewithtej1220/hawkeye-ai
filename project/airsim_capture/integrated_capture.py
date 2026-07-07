"""
INTEGRATED HAWKEYE SYSTEM
Captures thermal images from AirSim and sends them to the backend for real-time detection
"""

import airsim
import cv2
import numpy as np
import time
import threading
import requests
import base64
from pathlib import Path
from pynput import keyboard

# =========================================================
# CONFIGURATION
# =========================================================

BACKEND_URL = "http://localhost:8000"
CAPTURE_FOLDER = "live_captures"
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

# =========================================================
# GLOBALS
# =========================================================

client = None
current_camera_pitch = -10.0  # -10 degrees for better side view (was -20)
active_keys = set()
last_gimbal_press_time = 0.0
GIMBAL_DEBOUNCE_DELAY = 0.20
ANGLE_STEP = 10.0  # Smaller steps for finer control (was 20)
frame_lock = threading.Lock()
latest_rgb_frame = None
latest_thermal_frame = None
is_running = True
frame_counter = 0

# =========================================================
# SETUP
# =========================================================

Path(CAPTURE_FOLDER).mkdir(exist_ok=True)

print("=" * 60)
print("🎯 HAWKEYE INTEGRATED SYSTEM")
print("=" * 60)
print(f"Backend: {BACKEND_URL}")
print(f"Capture folder: {CAPTURE_FOLDER}")
print("=" * 60)

# =========================================================
# AIRSIM CONNECTION
# =========================================================

def connect_airsim():
    global client
    print("\n🔌 Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("✅ AirSim connected!")

# =========================================================
# ENVIRONMENT SETUP
# =========================================================

def setup_environment():
    print("\n⏰ Setting up environment...")
    try:
        client.simEnableWeather(True)
        client.simSetTimeOfDay(True)
        client.simRunConsoleCommand("r.TonemapperFilm 0")
        client.simRunConsoleCommand("r.Tonemapper.Gamma 0")
        client.simRunConsoleCommand("show Fog 0")
        client.simRunConsoleCommand("r.Fog 0")
        print("☀️ Environment ready!")
    except Exception as e:
        print(f"⚠️ Setup skipped: {e}")

# =========================================================
# SEGMENTATION SETUP
# =========================================================

def setup_segmentation():
    print("\n🎯 Setting up segmentation...")
    try:
        client.simSetSegmentationObjectID(".*", 0, True)
        client.simSetSegmentationObjectID(".*ThirdPersonCharacter.*", 255, True)
        client.simSetSegmentationObjectID(".*BP_Weapon.*", 85, True)
        print("✅ Segmentation ready!")
    except Exception as e:
        print(f"⚠️ Segmentation error: {e}")

# =========================================================
# CAMERA GIMBAL
# =========================================================

def update_camera_gimbal():
    global current_camera_pitch
    current_camera_pitch = max(-90.0, min(0.0, current_camera_pitch))
    q = airsim.to_quaternion(np.radians(current_camera_pitch), 0, 0)
    client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), q))

# =========================================================
# SEND TO BACKEND
# =========================================================

def send_to_backend(thermal_frame):
    """Send thermal frame to backend for detection"""
    try:
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', thermal_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send to backend detection endpoint
        response = requests.post(
            f"{BACKEND_URL}/detect",
            json={"image": img_base64},
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"⚠️ Backend error: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("⚠️ Backend not reachable. Make sure it's running!")
        return None
    except Exception as e:
        print(f"⚠️ Error sending to backend: {e}")
        return None

# =========================================================
# IMAGE PIPELINE
# =========================================================

def background_image_pipeline():
    global latest_rgb_frame, latest_thermal_frame, is_running, frame_counter
    
    worker_client = airsim.MultirotorClient()
    worker_client.confirmConnection()
    
    print("\n🎥 Starting image capture pipeline...")
    
    while is_running:
        try:
            responses = worker_client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
            ])
            
            if len(responses) < 2 or not responses[0].image_data_uint8 or not responses[1].image_data_uint8:
                time.sleep(0.01)
                continue
            
            # Process RGB
            rgb_raw = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            rgb = rgb_raw.reshape(responses[0].height, responses[0].width, 3)
            
            # Process Segmentation
            seg_raw = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
            seg = seg_raw.reshape(responses[1].height, responses[1].width, 3)
            
            # Resize
            if rgb.shape[1] != TARGET_WIDTH or rgb.shape[0] != TARGET_HEIGHT:
                rgb = cv2.resize(rgb, (TARGET_WIDTH, TARGET_HEIGHT))
            if seg.shape[1] != TARGET_WIDTH or seg.shape[0] != TARGET_HEIGHT:
                seg = cv2.resize(seg, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)
            
            # Fix RGB
            if rgb.shape[2] == 4:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)
            
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            stencil = seg[:, :, 2]
            
            # Body mask
            body_mask = cv2.inRange(stencil, 255, 255)
            
            # Weapon mask
            unique_ids = np.unique(stencil)
            weapon_id = None
            for uid in unique_ids:
                if uid != 0 and uid != 255:
                    weapon_id = int(uid)
                    break
            
            weapon_mask = np.zeros_like(stencil, dtype=np.uint8)
            if weapon_id is not None:
                weapon_mask = cv2.inRange(stencil, weapon_id, weapon_id)
            
            # Generate thermal
            thermal_base = np.zeros_like(gray, dtype=np.float32)
            
            if np.any(body_mask):
                dist_transform = cv2.distanceTransform(body_mask, cv2.DIST_L2, 5)
                max_dist = np.max(dist_transform)
                if max_dist == 0:
                    max_dist = 1.0
                normalized = dist_transform / max_dist
                sigmoid = 1.0 / (1.0 + np.exp(-14.0 * (normalized - 0.15)))
                
                if gray.shape != sigmoid.shape:
                    gray_resized = cv2.resize(gray, (sigmoid.shape[1], sigmoid.shape[0]))
                    luminance = gray_resized.astype(np.float32) / 255.0
                else:
                    luminance = gray.astype(np.float32) / 255.0
                
                shaded = sigmoid * (0.40 + 0.60 * luminance)
                thermal_base = (shaded * 65.0) + 190.0
                thermal_base[body_mask == 0] = 0
            
            character_heat = np.clip(thermal_base, 0, 255).astype(np.uint8)
            
            # Background
            bg_mask = ((255 - body_mask) // 255).astype(np.uint8)
            if len(bg_mask.shape) != len(gray.shape):
                bg_mask = bg_mask.reshape(gray.shape)
            
            bg_raw = cv2.multiply(gray, bg_mask)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_bg = clahe.apply(bg_raw)
            ambient = cv2.convertScaleAbs(enhanced_bg, alpha=0.25, beta=25)
            background_cold = cv2.multiply(ambient, bg_mask.astype(np.uint8))
            combined_gray = cv2.add(background_cold, character_heat)
            
            # Blur
            sharp = cv2.GaussianBlur(combined_gray, (3, 3), 0)
            bloom = cv2.GaussianBlur(combined_gray, (7, 7), 0)
            final_gray = cv2.addWeighted(sharp, 0.85, bloom, 0.15, 0)
            
            # Apply colormap
            thermal = cv2.applyColorMap(final_gray, cv2.COLORMAP_INFERNO)
            
            # Blue weapon
            thermal[weapon_mask > 0] = [255, 0, 0]
            
            with frame_lock:
                latest_rgb_frame = rgb.copy()
                latest_thermal_frame = thermal.copy()
                frame_counter += 1
                
                # Send to backend every 30 frames (~1 per second to reduce lag)
                if frame_counter % 30 == 0:
                    try:
                        _, buffer = cv2.imencode('.jpg', thermal, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        img_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        response = requests.post(
                            f"{BACKEND_URL}/detect",
                            json={"image": img_base64, "frame_id": frame_counter},
                            timeout=1
                        )
                        
                        if response.status_code == 200:
                            print(f"✅ Frame {frame_counter} sent to backend")
                    except Exception:
                        pass  # Silent fail
            
            time.sleep(0.03)  # ~30 FPS
            
        except Exception as e:
            print(f"⚠️ Pipeline error: {e}")
            time.sleep(0.1)

# =========================================================
# KEYBOARD CONTROLS
# =========================================================

def on_press(key):
    global current_camera_pitch, last_gimbal_press_time
    current_time = time.time()
    
    try:
        if hasattr(key, 'char') and key.char:
            k = key.char.lower()
            active_keys.add(k)
            
            if k in ['r', 'f']:
                if (current_time - last_gimbal_press_time) > GIMBAL_DEBOUNCE_DELAY:
                    if k == 'r':
                        current_camera_pitch += ANGLE_STEP
                    if k == 'f':
                        current_camera_pitch -= ANGLE_STEP
                    update_camera_gimbal()
                    print(f"📸 Camera Pitch: {current_camera_pitch}")
                    last_gimbal_press_time = current_time
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
# MAIN
# =========================================================

def main():
    global is_running
    
    # Connect and setup
    connect_airsim()
    setup_environment()
    setup_segmentation()
    update_camera_gimbal()
    
    # Start image pipeline
    pipeline_thread = threading.Thread(target=background_image_pipeline, daemon=True)
    pipeline_thread.start()
    
    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    print("\n🚀 HawkEye Integrated System Active!\n")
    print("👉 Controls:")
    print("W / S        : Forward / Backward")
    print("A / D        : Yaw Left / Right")
    print("ARROWS       : Left / Right")
    print("SPACE/SHIFT  : Up / Down")
    print("R / F        : Camera Gimbal")
    print("ESC          : Exit")
    print("\n📡 Streaming to backend for real-time detection...")
    
    SPEED = 4.0
    YAW_RATE = 40.0
    
    try:
        while listener.running:
            # Flight controls
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
                vx, vy, vz,
                duration=0.02,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_speed)
            )
            
            # Display frames
            with frame_lock:
                local_rgb = latest_rgb_frame.copy() if latest_rgb_frame is not None else None
                local_thermal = latest_thermal_frame.copy() if latest_thermal_frame is not None else None
            
            if local_rgb is not None:
                cv2.imshow("HawkEye RGB", cv2.resize(local_rgb, (640, 360)))
            
            if local_thermal is not None:
                cv2.imshow("HawkEye Thermal", cv2.resize(local_thermal, (640, 360)))
            
            cv2.waitKey(1)
            time.sleep(0.01)
            
    finally:
        print("\n🛑 Shutting down...")
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

if __name__ == "__main__":
    main()
