"""Shared pieces of the autonomous-flight module: the policy's observation (what the drone senses) and
action (what it commands), defined once so the training world and the real simulator match exactly.

Observation (all scaled to about -1..1):
  * 24 depth rays across the front camera's 90 degree field of view, middle band  -> obstacles ahead
  * 8 depth rays from the lower band (looking slightly down)                   -> ground rising ahead
  * goal direction (sin, cos of the bearing relative to the nose) and distance
  * height above ground error (target height minus current)
  * current forward / sideways / vertical speed and yaw rate
  * previous action (so the policy can learn to fly smoothly)
Action (3 values in -1..1): forward speed, yaw rate, vertical speed.
"""
import math

import numpy as np

N_RAYS = 24            # middle band, left to right
N_LOW = 8              # lower band
FOV_DEG = 90.0
MAX_RANGE = 25.0       # metres; anything farther reads as "clear"
LOW_PITCH_DEG = -12.0  # the lower band looks this far below the nose
# The obstacle (middle) band covers this vertical slice of the camera's view, so a treetop just below
# the flight level counts as an obstacle (a level ray alone would pass over it while the body clips it)
MID_BAND_UP_DEG, MID_BAND_DOWN_DEG = 2.0, 8.0
TARGET_AGL = 7.0       # metres above ground: below the forest canopy, so trees are real obstacles
MAX_FWD, MAX_YAW, MAX_VZ = 5.0, 60.0, 2.0  # m/s, deg/s, m/s
CONTROL_HZ = 10
OBS_SIZE = N_RAYS + N_LOW + 3 + 1 + 4 + 3

RAY_ANGLES = np.radians(np.linspace(-FOV_DEG / 2, FOV_DEG / 2, N_RAYS))
LOW_ANGLES = np.radians(np.linspace(-FOV_DEG / 2, FOV_DEG / 2, N_LOW))


def action_to_command(action):
    """Policy output (-1..1) -> (forward m/s, yaw deg/s, vertical m/s, NED: positive = down).
    Forward speed is never negative: the drone does not fly backwards blind."""
    a = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
    forward = (a[0] + 1) / 2 * MAX_FWD
    return float(forward), float(a[1] * MAX_YAW), float(a[2] * MAX_VZ)


def build_observation(rays, low_rays, goal_bearing, goal_dist, agl, vel_body, yaw_rate, prev_action):
    """Pack sensor readings into the policy's observation vector.
    rays / low_rays: distances in metres (MAX_RANGE when nothing is seen);
    goal_bearing: radians, positive = goal to the right; vel_body: (forward, right, down) m/s."""
    obs = np.empty(OBS_SIZE, dtype=np.float32)
    obs[:N_RAYS] = np.clip(np.asarray(rays) / MAX_RANGE, 0, 1) * 2 - 1
    obs[N_RAYS:N_RAYS + N_LOW] = np.clip(np.asarray(low_rays) / MAX_RANGE, 0, 1) * 2 - 1
    i = N_RAYS + N_LOW
    obs[i], obs[i + 1] = math.sin(goal_bearing), math.cos(goal_bearing)
    obs[i + 2] = min(goal_dist, 60.0) / 30.0 - 1
    obs[i + 3] = float(np.clip((TARGET_AGL - agl) / 5.0, -1, 1))
    obs[i + 4] = vel_body[0] / MAX_FWD
    obs[i + 5] = float(np.clip(vel_body[1] / 2.0, -1, 1))
    obs[i + 6] = float(np.clip(vel_body[2] / MAX_VZ, -1, 1))
    obs[i + 7] = float(np.clip(yaw_rate / MAX_YAW, -1, 1))
    obs[i + 8:i + 11] = np.clip(prev_action, -1, 1)
    return obs


CORRIDOR_HALF_WIDTH = 1.5   # metres either side of the flight path the shield keeps clear
LOW_COS = math.cos(math.radians(LOW_PITCH_DEG))


def path_clearance(rays, low_rays):
    """Free distance along the drone's flight path: every ray is projected onto the forward axis, and
    only obstacles within CORRIDOR_HALF_WIDTH of the path count (a trunk just off to the side is
    clipped in a turn; one far to the side is not in the way)."""
    rays, low = np.asarray(rays), np.asarray(low_rays)
    lateral = rays * np.abs(np.sin(RAY_ANGLES))
    ahead = rays * np.cos(RAY_ANGLES)
    in_path = lateral < CORRIDOR_HALF_WIDTH
    clearance = float(ahead[in_path].min()) if in_path.any() else MAX_RANGE
    low_lateral = low * LOW_COS * np.abs(np.sin(LOW_ANGLES))
    low_ahead = low * LOW_COS * np.cos(LOW_ANGLES)
    low_in = low_lateral < CORRIDOR_HALF_WIDTH
    if low_in.any():
        clearance = min(clearance, float(low_ahead[low_in].min()))
    return clearance


class ObstacleMemory:
    """Short-term map of obstacles seen by the depth rays, in world coordinates (x, y). The camera only
    sees 90 degrees ahead; when the drone turns, its momentum carries it on for a moment toward trunks
    that have already left the picture. Remembering recent hits lets the shield check the actual
    direction of travel, not just the camera's view."""

    def __init__(self, keep_s=8.0, radius=18.0):
        self.keep_s, self.radius = keep_s, radius
        self.points = np.zeros((0, 3))  # x, y, time seen

    def update(self, x, y, yaw, rays, now):
        rays = np.asarray(rays)
        hit = rays < MAX_RANGE - 0.5
        if hit.any():
            d, a = rays[hit], yaw + RAY_ANGLES[hit]
            new = np.stack([x + d * np.cos(a), y + d * np.sin(a), np.full(d.shape, now)], axis=1)
            self.points = np.vstack([self.points, new])
        keep = (now - self.points[:, 2] < self.keep_s) & \
            (np.hypot(self.points[:, 0] - x, self.points[:, 1] - y) < self.radius)
        self.points = self.points[keep]
        if len(self.points) > 3000:
            self.points = self.points[-3000:]

    def collision_ahead(self, x, y, vx, vy, horizon=1.5, body=1.2):
        """True if continuing at the current velocity for `horizon` seconds passes within `body` metres
        of a remembered obstacle."""
        if not len(self.points):
            return False
        t = np.linspace(0.1, horizon, 8)
        path = np.stack([x + vx * t, y + vy * t], axis=1)
        d = np.hypot(path[:, None, 0] - self.points[None, :, 0], path[:, None, 1] - self.points[None, :, 1])
        return bool(d.min() < body)


def safety_shield(rays, low_rays, agl, command, speed=None, min_agl=3.0, margin=1.5, lag=0.6, decel=3.0,
                  memory_alarm=False):
    """Last line of defence, applied after the policy: whatever it asks for, the drone keeps a
    stopping distance clear along its flight path (from its current speed: response lag plus braking),
    slows down as obstacles approach, stops and turns toward open space before one, climbs if the
    ground comes up, and never descends below min_agl. Returns (command, reason or None)."""
    forward, yaw, vz = command
    speed = forward if speed is None else max(0.0, speed)
    braking = speed * lag + speed * speed / (2 * decel)
    clearance = path_clearance(rays, low_rays)
    reason = None
    if memory_alarm:
        # the drone's momentum is carrying it toward an obstacle it saw earlier (now out of view)
        forward, reason = 0.0, "remembered obstacle on the flight path: stop"
    elif clearance < braking + margin:
        forward, reason = 0.0, f"obstacle {clearance:.1f} m ahead: stop"
        if abs(yaw) < MAX_YAW * 0.3:
            # turn toward the side with more open space rather than hover in front of the obstacle
            left, right = np.asarray(rays)[:N_RAYS // 2].mean(), np.asarray(rays)[N_RAYS // 2:].mean()
            yaw = MAX_YAW * 0.6 * (1 if right >= left else -1)
    else:
        # the speed whose stopping distance (plus margin) fits in the clearance
        room = clearance - margin
        limit = max(0.0, (-lag + math.sqrt(lag * lag + 2 * room / decel)) * decel)
        if forward > limit:
            forward, reason = limit, f"obstacle {clearance:.1f} m ahead: slow"
    if agl < min_agl and vz > -0.5:
        vz, reason = -1.0, f"only {agl:.1f} m above ground: climb"
    return (forward, yaw, vz), reason
