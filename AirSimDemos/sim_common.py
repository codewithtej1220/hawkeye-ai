"""Shared simulator code for HawkEye: thermal rendering of AirSim frames, segmentation setup,
drone telemetry. Used by thermal.py (manual flight), range_test.py and sim_actor.py."""
import math

import airsim
import cv2
import numpy as np

BODY_ID = 255          # segmentation ID given to the soldier
WEAPON_ID = 85         # segmentation ID given to the rifle
CHARACTER_REGEX = ".*ThirdPersonCharacter.*"
WEAPON_REGEX = ".*BP_Weapon.*"
CAMERA_FOV_DEG = 90.0  # matches CaptureSettings in airsim_settings.json

# White-hot grey levels (0-255). Tuned on captured frames against hawkeye_real_yolo11m.pt.
WHITEHOT_BG_BASE = 110       # mean ambient terrain level
WHITEHOT_BG_CONTRAST = 0.2   # terrain texture strength (real thermal backgrounds are low contrast)
WHITEHOT_SKIN_CORE = 225     # warmest part of the body
WHITEHOT_SKIN_EDGE = 185     # body outline (clothing / limbs are cooler)
WHITEHOT_WEAPON = 45         # cold metal, darker than terrain
_noise_rng = np.random.default_rng()


def render_whitehot(gray, body_mask, weapon_mask):
    """Render the AirSim scene as a white-hot thermal image (3-channel greyscale)."""
    # Blur radii follow the capture size (tuned at 640 px wide), so the look is the same at any resolution
    k = max(1.0, gray.shape[1] / 640)
    # Background: low-contrast ambient terrain, softened like a microbolometer sensor
    bg = cv2.GaussianBlur(gray, (0, 0), 1.0 * k).astype(np.float32)
    out = WHITEHOT_BG_BASE + WHITEHOT_BG_CONTRAST * (bg - 127.5)

    if np.any(body_mask):
        # Warm core fading to cooler outline, with clothing variation from scene shading
        dist = cv2.distanceTransform(body_mask, cv2.DIST_L2, 5)
        core = np.clip(dist / max(dist.max() * 0.5, 1.0), 0, 1)
        shading = (gray.astype(np.float32) / 255.0 - 0.5) * 24
        heat = WHITEHOT_SKIN_EDGE + (WHITEHOT_SKIN_CORE - WHITEHOT_SKIN_EDGE) * core + shading
        # Heat halo bleeding a couple of pixels into the surroundings
        halo = cv2.GaussianBlur((body_mask > 0).astype(np.float32), (0, 0), 2.0 * k)
        warm = np.where(body_mask > 0, heat, out + (WHITEHOT_SKIN_EDGE - out) * halo * 0.6)
        out = np.maximum(out, warm)

    if np.any(weapon_mask):
        w = cv2.GaussianBlur((weapon_mask > 0).astype(np.float32), (0, 0), 0.8)
        out = out * (1 - w) + WHITEHOT_WEAPON * w

    # Optics blur + sensor noise
    out = cv2.GaussianBlur(out, (0, 0), 0.9)
    out += _noise_rng.normal(0, 2.0, out.shape)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def masks_from_segmentation(seg):
    """(body_mask, weapon_mask, weapon_id) from an AirSim segmentation image."""
    stencil = seg[:, :, 2]
    body_mask = cv2.inRange(stencil, BODY_ID, BODY_ID)
    weapon_id = next((int(u) for u in np.unique(stencil) if u not in (0, BODY_ID)), None)
    weapon_mask = (cv2.inRange(stencil, weapon_id, weapon_id) if weapon_id is not None
                   else np.zeros_like(stencil, dtype=np.uint8))
    return body_mask, weapon_mask, weapon_id


def thermal_from(rgb, seg, palette="inferno"):
    """Thermal camera image (BGR) of an AirSim scene + segmentation pair, in the given palette."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    body_mask, weapon_mask, weapon_id = masks_from_segmentation(seg)
    thermal = render_whitehot(gray, body_mask, weapon_mask)
    if palette == "inferno":
        thermal = cv2.applyColorMap(cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)
    return thermal, body_mask, weapon_mask, weapon_id


def setup_segmentation(client):
    client.simSetSegmentationObjectID(".*", 0, True)
    client.simSetSegmentationObjectID(CHARACTER_REGEX, BODY_ID, True)
    client.simSetSegmentationObjectID(WEAPON_REGEX, WEAPON_ID, True)


def capture_pair(client, width=None, height=None):
    """(rgb, seg) BGR arrays from the drone's front camera, or None when a frame is not ready."""
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
    ])
    if len(responses) < 2 or not responses[0].image_data_uint8 or not responses[1].image_data_uint8:
        return None
    rgb = np.frombuffer(responses[0].image_data_uint8, np.uint8).reshape(responses[0].height, responses[0].width, 3)
    seg = np.frombuffer(responses[1].image_data_uint8, np.uint8).reshape(responses[1].height, responses[1].width, 3)
    if width and (rgb.shape[1] != width or rgb.shape[0] != height):
        rgb = cv2.resize(rgb, (width, height))
    if width and (seg.shape[1] != width or seg.shape[0] != height):
        seg = cv2.resize(seg, (width, height), interpolation=cv2.INTER_NEAREST)
    return rgb, seg


class ParallelCapture:
    """Colour + segmentation pair, requested at the same moment over two connections. AirSim serves
    one request per connection at a time; asking for both images in one call took 0.42 s per pair,
    two parallel calls take 0.15 s (measured at 1280x960), so the stream runs ~2.7x faster."""

    def __init__(self, width=None, height=None):
        from concurrent.futures import ThreadPoolExecutor
        self.width, self.height = width, height
        self._clients = [airsim.MultirotorClient(), airsim.MultirotorClient()]
        for c in self._clients:
            c.confirmConnection()
        self._pool = ThreadPoolExecutor(max_workers=2)

    @staticmethod
    def _one(client, image_type):
        r = client.simGetImages([airsim.ImageRequest("0", image_type, False, False)])
        if not r or not r[0].image_data_uint8:
            return None
        return np.frombuffer(r[0].image_data_uint8, np.uint8).reshape(r[0].height, r[0].width, 3)

    def pair(self):
        """(rgb, seg) BGR arrays, or None when a frame is not ready."""
        jobs = [self._pool.submit(self._one, c, t)
                for c, t in zip(self._clients, (airsim.ImageType.Scene, airsim.ImageType.Segmentation))]
        rgb, seg = (j.result() for j in jobs)
        if rgb is None or seg is None:
            return None
        if self.width and (rgb.shape[1] != self.width or rgb.shape[0] != self.height):
            rgb = cv2.resize(rgb, (self.width, self.height))
        if self.width and (seg.shape[1] != self.width or seg.shape[0] != self.height):
            seg = cv2.resize(seg, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return rgb, seg


def find_actor(client, keyword):
    """Name of the first scene object containing `keyword` (case-insensitive), or None."""
    return next((o for o in client.simListSceneObjects() if keyword.lower() in o.lower()), None)


def telemetry(client, camera_pitch_deg, target=None, home_ground_z=None):
    """Drone state for the dashboard and the event log.

    altitude_m is height above the take-off point (what a real drone's barometer/GPS reports).
    With `target` (a scene object name, simulator only) the true slant range to it is included as
    target_range_m, which lets the range estimate made from the image be checked."""
    state = client.getMultirotorState().kinematics_estimated
    p, v = state.position, state.linear_velocity
    _, _, yaw = airsim.to_eularian_angles(state.orientation)
    ground_z = home_ground_z if home_ground_z is not None else 0.0
    data = {
        "altitude_m": round(ground_z - p.z_val, 1),
        "north_m": round(p.x_val, 1),
        "east_m": round(p.y_val, 1),
        "speed_mps": round(math.sqrt(v.x_val ** 2 + v.y_val ** 2 + v.z_val ** 2), 1),
        "heading_deg": round(math.degrees(yaw) % 360, 0),
        "camera_pitch_deg": round(camera_pitch_deg, 0),
        "camera_fov_deg": CAMERA_FOV_DEG,
    }
    if target:
        t = client.simGetObjectPose(target).position
        if not math.isnan(t.x_val):
            data["target_range_m"] = round(math.sqrt((t.x_val - p.x_val) ** 2 + (t.y_val - p.y_val) ** 2
                                                     + (t.z_val - p.z_val) ** 2), 1)
            data["target_height_above_m"] = round(t.z_val - p.z_val, 1)
            # simulator ground truth, for the dashboard's map (a real drone would not know this)
            data["target_north_m"] = round(t.x_val, 1)
            data["target_east_m"] = round(t.y_val, 1)
    return data
