"""Bridge between the learned policy and the AirSim simulator: real depth camera -> the policy's rays and
height above ground, policy action -> flight command (behind the safety shield).

Cameras (AirSim multirotor defaults):
  "0" front_center  - navigation: depth image -> obstacle rays, and height above ground from the
                      image's lower rows (the ground ahead and below)
  "1" front_right   - surveillance: tilted down for the people/weapon detector (autonomous_patrol.py)

Timing: the simulator renders image requests one after another, about 0.2 s each, shared with the
surveillance camera. Waiting for a fresh depth image every control step would hold the control loop
to ~2 Hz. Instead a sensor thread (its own connection) keeps grabbing depth images, and the control
loop runs at the policy's 10 Hz on the latest one, after shifting it to the drone's current position
and heading (the obstacles seen 0.2 s ago are where they were; the drone has moved).
"""
import math
import threading
import time

import airsim
import numpy as np

from nav_core import (FOV_DEG, LOW_ANGLES, LOW_PITCH_DEG, MAX_RANGE, MID_BAND_DOWN_DEG, MID_BAND_UP_DEG, N_LOW,
                      N_RAYS, RAY_ANGLES, ObstacleMemory, action_to_command, build_observation, safety_shield)

NAV_CAMERA, DOWN_CAMERA = "0", "3"
DOWN_EVERY = 3          # frames between downward height checks
AGL_ROWS = 15            # lowest image rows used for height above ground
STALE_S = 1.0            # older depth than this: stop until a fresh image arrives


def depth_to_rays(depth, nose_down=0.0):
    """DepthPerspective image (metres along each pixel's ray, 90 degree horizontal FOV, camera fixed to
    the body) -> 24 middle-band rays and 8 lower-band rays, each the nearest obstacle in its sector.
    nose_down (radians): the drone's forward tilt; the horizon then sits higher in the image."""
    h, w = depth.shape
    f = (w / 2) / math.tan(math.radians(FOV_DEG / 2))
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, MAX_RANGE)
    horizon = h / 2 - f * math.tan(nose_down)
    band = max(2, int(0.03 * h))
    # obstacle band: from MID_BAND_UP_DEG above the horizon to MID_BAND_DOWN_DEG below (as in training)
    top = int(horizon - f * math.tan(math.radians(MID_BAND_UP_DEG)))
    bottom = int(horizon + f * math.tan(math.radians(MID_BAND_DOWN_DEG))) + 1
    top, bottom = min(max(0, top), h - 1), max(min(h, bottom), 1)
    mid_rows = depth[top:max(bottom, top + 1)]
    low_row = int(horizon + f * math.tan(math.radians(-LOW_PITCH_DEG)))
    low_row = min(max(0, low_row), h - 1)
    low_rows = depth[max(0, low_row - band // 2):min(h, low_row + band // 2 + 1)]

    def sectors(rows, n):
        edges = np.linspace(0, w, n + 1).astype(int)
        return np.array([rows[:, a:b].min() for a, b in zip(edges[:-1], edges[1:])])
    return np.clip(sectors(mid_rows, N_RAYS), 0.2, MAX_RANGE), np.clip(sectors(low_rows, N_LOW), 0.2, MAX_RANGE)


def depth_to_agl(depth, nose_down=0.0):
    """Height above ground from the lowest rows of the front depth image: each pixel's distance times
    the downward component of its ray. The median over the middle columns is the ground (a tree in
    front reads lower, so the estimate errs toward 'too low', i.e. climbing)."""
    h, w = depth.shape
    f = (w / 2) / math.tan(math.radians(FOV_DEG / 2))
    v = (np.arange(h - AGL_ROWS, h) + 0.5 - h / 2)[:, None]
    u = (np.arange(w // 4, 3 * w // 4) + 0.5 - w / 2)[None, :]
    d = depth[h - AGL_ROWS:, w // 4:3 * w // 4]
    down = (v * math.cos(nose_down) + f * math.sin(nose_down)) / np.sqrt(f * f + u * u + v * v)
    drop = d * down
    ok = np.isfinite(drop) & (d > 0) & (d < 200)
    return float(np.median(drop[ok])) if ok.sum() > 20 else 50.0  # no ground in view: high up


def nose_down_of(orientation):
    pitch, _, _ = airsim.to_eularian_angles(orientation)
    return -pitch  # AirSim NED: positive pitch is nose up


def shift_rays(rays, angles, captured, now):
    """Re-express rays measured at pose `captured` (x, y, yaw) from pose `now`. Each ray end is a point
    in the world; seen from the new pose it has a new distance and bearing. Rays are resampled at the
    standard angles (outside the old view: the nearest edge reading)."""
    x0, y0, yaw0 = captured
    x1, y1, yaw1 = now
    a = yaw0 + angles
    px, py = x0 + rays * np.cos(a), y0 + rays * np.sin(a)
    d = np.hypot(px - x1, py - y1)
    b = (np.arctan2(py - y1, px - x1) - yaw1 + math.pi) % (2 * math.pi) - math.pi
    order = np.argsort(b)
    out = np.interp(angles, b[order], d[order])
    return np.clip(out, 0.2, MAX_RANGE)


class DepthSensor(threading.Thread):
    """Keeps the latest depth reading: (rays, low rays, agl, capture time, capture pose)."""

    def __init__(self, client):
        super().__init__(daemon=True)
        self.client = client
        self.running = True
        self.latest = None
        self.frames = 0
        self.error = None
        self.down = None  # (agl straight down, altitude then, time)

    def capture(self):
        state = self.client.getMultirotorState().kinematics_estimated
        r = self.client.simGetImages([airsim.ImageRequest(NAV_CAMERA, airsim.ImageType.DepthPerspective, True, False)])[0]
        depth = np.array(r.image_data_float, dtype=np.float32).reshape(r.height, r.width)
        tilt = nose_down_of(state.orientation)
        rays, low = depth_to_rays(depth, tilt)
        agl = depth_to_agl(depth, tilt)
        _, _, yaw = airsim.to_eularian_angles(state.orientation)
        p = state.position
        # The front image reads the ground ahead: flying downhill it is lower than the ground below, so
        # the estimate runs high and the drone sinks. Every DOWN_EVERY-th frame also measure straight down
        # and keep the lower (safer) of the two.
        if self.frames % DOWN_EVERY == 0:
            d = self.client.simGetImages([airsim.ImageRequest(DOWN_CAMERA, airsim.ImageType.DepthPlanar, True, False)])[0]
            down = np.array(d.image_data_float, dtype=np.float32).reshape(d.height, d.width)
            centre = down[d.height // 3:2 * d.height // 3, d.width // 3:2 * d.width // 3]
            centre = centre[np.isfinite(centre) & (centre > 0) & (centre < 200)]
            if centre.size:
                self.down = (float(np.median(centre)), -p.z_val, time.monotonic())
        if self.down and time.monotonic() - self.down[2] < 2.0:
            agl = min(agl, self.down[0] + (-p.z_val - self.down[1]))
        self.latest = (rays, low, agl, time.monotonic(), (p.x_val, p.y_val, yaw), -p.z_val)
        self.frames += 1

    def run(self):
        while self.running:
            try:
                self.capture()
            except Exception as exc:  # simulator hiccup: keep trying, the pilot stops on stale data
                self.error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.2)


class Pilot:
    def __init__(self, client, policy_path, sensor_client=None):
        """sensor_client: a second airsim client for the background depth thread (real simulator).
        None: sense synchronously every step (offline test with a fake simulator)."""
        from stable_baselines3 import PPO
        self.client = client
        self.model = PPO.load(policy_path, device="cpu")
        self.prev_action = np.zeros(3, dtype=np.float32)
        level = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0))
        client.simSetCameraPose(NAV_CAMERA, level)   # navigation camera looks straight ahead
        client.simSetCameraFov(NAV_CAMERA, FOV_DEG)
        self.rays = np.full(N_RAYS, MAX_RANGE)
        self.low = np.full(N_LOW, MAX_RANGE)
        self.agl = 0.0
        self.depth_age = 0.0
        self.memory = ObstacleMemory()
        self.sensor = DepthSensor(sensor_client or client)
        if sensor_client is not None:
            self.sensor.start()

    def sense(self, state=None):
        """Update rays and height above ground for the drone's current pose."""
        if not self.sensor.is_alive():
            self.sensor.capture()
        if self.sensor.latest is None:
            return False
        rays, low, agl, t, captured, alt_then = self.sensor.latest
        state = state or self.client.getMultirotorState().kinematics_estimated
        p = state.position
        _, _, yaw = airsim.to_eularian_angles(state.orientation)
        now = (p.x_val, p.y_val, yaw)
        self.rays = shift_rays(rays, RAY_ANGLES, captured, now)
        self.low = shift_rays(low, LOW_ANGLES, captured, now)
        self.agl = agl + (-p.z_val - alt_then)  # climbed or sank since the image
        self.depth_age = time.monotonic() - t
        return True

    def step(self, goal_ned, dt):
        """One control step toward goal (x, y in NED metres). Returns (command, shield reason, distance)."""
        state = self.client.getMultirotorState().kinematics_estimated
        fresh = self.sense(state) and self.depth_age < STALE_S
        p, v = state.position, state.linear_velocity
        _, _, yaw = airsim.to_eularian_angles(state.orientation)
        bearing = math.atan2(goal_ned[1] - p.y_val, goal_ned[0] - p.x_val) - yaw
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        dist = math.hypot(goal_ned[0] - p.x_val, goal_ned[1] - p.y_val)
        if not fresh:  # no recent depth image: do not fly blind
            self.hold(dt)
            return (0.0, 0.0, 0.0), "no fresh depth image: hold", dist
        c, s = math.cos(yaw), math.sin(yaw)
        v_body = (v.x_val * c + v.y_val * s, -v.x_val * s + v.y_val * c, v.z_val)
        yaw_rate = math.degrees(state.angular_velocity.z_val)
        obs = build_observation(self.rays, self.low, bearing, dist, self.agl, v_body, yaw_rate, self.prev_action)
        action, _ = self.model.predict(obs, deterministic=True)
        self.prev_action = np.asarray(action, dtype=np.float32)
        self.memory.update(p.x_val, p.y_val, yaw, self.rays, time.monotonic())
        alarm = self.memory.collision_ahead(p.x_val, p.y_val, v.x_val, v.y_val)
        command, reason = safety_shield(self.rays, self.low, self.agl, action_to_command(action),
                                        speed=math.hypot(v.x_val, v.y_val), memory_alarm=alarm,
                                        lag=0.6 + self.depth_age)  # older image: brake earlier
        forward, yaw_deg, vz = command
        # the command lasts a few control periods, so a late next step never leaves the drone uncommanded
        # for long, and a stalled script stops it soon after
        self.client.moveByVelocityBodyFrameAsync(forward, 0.0, vz, duration=max(dt * 3, 0.5),
                                                 drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                 yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_deg))
        return command, reason, dist

    def hold(self, dt):
        self.client.moveByVelocityBodyFrameAsync(0.0, 0.0, 0.0, duration=max(dt * 3, 0.5),
                                                 yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0))

    def stop(self):
        self.sensor.running = False

    @property
    def nearest_obstacle(self):
        return float(min(self.rays.min(), self.low.min()))
