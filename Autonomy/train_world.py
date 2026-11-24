"""Fast training world for the autonomous-flight policy (gymnasium environment).

Learning to fly inside Unreal would take days (the simulator runs ~10-20 control steps a second), so
the policy learns here first, millions of steps in an hour, then flies in Unreal with the real depth
camera. The world mirrors what matters for obstacle avoidance:
  * rolling terrain (smooth hills) the drone must follow at TARGET_AGL above ground,
  * trees as vertical cylinders (trunk + canopy radius, 6-22 m tall), random density each episode,
    from open ground to dense forest,
  * the same 24 + 8 depth rays the real front camera provides (nav_core), with noise and dropouts,
  * drone dynamics with response lag (a commanded speed is reached over ~0.4 s, as in AirSim).
Each episode, goals appear 25-70 m away; reaching one gives a new one, until a collision or time out.
"""
import math

import gymnasium as gym
import numpy as np

from nav_core import (CONTROL_HZ, LOW_ANGLES, LOW_PITCH_DEG, MAX_RANGE, MID_BAND_DOWN_DEG, N_LOW, N_RAYS, OBS_SIZE,
                      RAY_ANGLES, TARGET_AGL, ObstacleMemory, action_to_command, build_observation, safety_shield)

WORLD = 240.0          # metres, square
DRONE_RADIUS = 0.6
GOAL_RADIUS = 4.0
DT = 1.0 / CONTROL_HZ
STEP_SAMPLES = np.linspace(0.3, MAX_RANGE, 60)  # sample distances along each ray (terrain check)


class Terrain:
    """Smooth random hills: a sum of a few broad Gaussians and gentle waves."""

    def __init__(self, rng, roughness):
        n = 6
        self.cx, self.cy = rng.uniform(0, WORLD, n), rng.uniform(0, WORLD, n)
        self.h = rng.uniform(-1, 1, n) * 12 * roughness
        self.s = rng.uniform(25, 60, n)
        self.wave = rng.uniform(0, 3) * roughness, rng.uniform(0.02, 0.06), rng.uniform(0, 2 * math.pi)

    def height(self, x, y):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        z = np.zeros(np.broadcast(x, y).shape)
        for cx, cy, h, s in zip(self.cx, self.cy, self.h, self.s):
            z += h * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
        a, f, p = self.wave
        return z + a * np.sin(f * x + p) * np.cos(f * 0.8 * y)


class ForestWorld(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=1200, seed=None, shielded=True):
        """shielded: the policy's commands pass through the safety shield and obstacle memory exactly
        as in real flight, and every intervention costs a little reward, so the policy learns to keep
        its own distance rather than rely on the shield."""
        self.observation_space = gym.spaces.Box(-1.5, 1.5, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.shielded = shielded

    # ------------------------------------------------------------------ world generation
    def _make_world(self):
        rng = self.rng
        self.terrain = Terrain(rng, rng.uniform(0.2, 1.0))
        density = rng.choice([0.0, 0.002, 0.005, 0.008, 0.012, 0.016])  # trees per square metre
        n = int(density * WORLD * WORLD)
        tx, ty = rng.uniform(0, WORLD, n), rng.uniform(0, WORLD, n)
        self.trees = np.stack([tx, ty, rng.uniform(0.3, 2.5, n),            # radius (trunk..canopy)
                               self.terrain.height(tx, ty),                  # base height
                               rng.uniform(6, 22, n)], axis=1) if n else np.zeros((0, 5))
        self.trees[:, 4] += self.trees[:, 3]                                 # top height
        self.noise = rng.uniform(0.0, 0.06)                                  # depth noise (fraction)
        self.lag = rng.uniform(0.25, 0.6)                                    # response time constant (s)

    def _free(self, x, y, clearance):
        if not len(self.trees):
            return True
        d = np.hypot(self.trees[:, 0] - x, self.trees[:, 1] - y) - self.trees[:, 2]
        return d.min() > clearance

    def _new_goal(self):
        for _ in range(50):
            d, a = self.rng.uniform(25, 70), self.rng.uniform(0, 2 * math.pi)
            gx, gy = self.x + d * math.cos(a), self.y + d * math.sin(a)
            if 15 < gx < WORLD - 15 and 15 < gy < WORLD - 15 and self._free(gx, gy, 3.0):
                self.goal = (gx, gy)
                return
        self.goal = (WORLD / 2, WORLD / 2)

    # ------------------------------------------------------------------ sensing
    def _cast(self, angles, pitch, band_down=0.0):
        """Distances along rays from the drone (yaw + angles, fixed pitch); MAX_RANGE if nothing hit.
        band_down (radians): the ray stands for a vertical band reaching this far below it, so a tree
        whose top is inside the band counts as hit."""
        dirs = self.yaw + angles
        cx, sx = np.cos(dirs), np.sin(dirs)
        cp, sp = math.cos(pitch), math.sin(pitch)
        result = np.full(len(angles), MAX_RANGE)
        # terrain: march along each ray
        px = self.x + np.outer(cx * cp, STEP_SAMPLES)
        py = self.y + np.outer(sx * cp, STEP_SAMPLES)
        pz = self.z + sp * STEP_SAMPLES[None, :]
        under = pz < self.terrain.height(px, py)
        hit = under.any(axis=1)
        result[hit] = STEP_SAMPLES[under[hit].argmax(axis=1)]
        # trees: ray / vertical-cylinder intersection in the horizontal plane, then height check
        if len(self.trees):
            near = np.hypot(self.trees[:, 0] - self.x, self.trees[:, 1] - self.y) < MAX_RANGE + 3
            t = self.trees[near]
            if len(t):
                ox, oy = t[:, 0] - self.x, t[:, 1] - self.y           # tree centre relative to drone
                proj = np.outer(cx, ox) + np.outer(sx, oy)             # along-ray distance (horizontal)
                perp2 = (ox ** 2 + oy ** 2)[None, :] - proj ** 2
                r2 = (t[:, 2] ** 2)[None, :]
                inside = perp2 < r2
                along = proj - np.sqrt(np.clip(r2 - perp2, 0, None))   # entry point (horizontal)
                dist = along / max(cp, 1e-3)
                zh = self.z + sp * dist
                z_low = zh - along * math.tan(band_down)  # lower edge of the band at that distance
                valid = inside & (along > 0) & (zh >= t[:, 3][None, :]) & (z_low <= t[:, 4][None, :])
                dist = np.where(valid, dist, np.inf).min(axis=1)
                result = np.minimum(result, dist)
        # sensor noise and dropouts (missing depth reads as clear)
        result = result * (1 + self.rng.normal(0, self.noise, len(result)))
        drop = self.rng.random(len(result)) < self.noise * 0.3
        result[drop] = MAX_RANGE
        return np.clip(result, 0.2, MAX_RANGE)

    def _observe(self):
        self.rays = self._cast(RAY_ANGLES, 0.0, math.radians(MID_BAND_DOWN_DEG))
        self.low = self._cast(LOW_ANGLES, math.radians(LOW_PITCH_DEG))
        gx, gy = self.goal
        bearing = math.atan2(gy - self.y, gx - self.x) - self.yaw
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        self.goal_dist = math.hypot(gx - self.x, gy - self.y)
        self.agl = self.z - float(self.terrain.height(self.x, self.y))
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        v_body = (self.vx * c + self.vy * s, -self.vx * s + self.vy * c, -self.vz)
        return build_observation(self.rays, self.low, bearing, self.goal_dist, self.agl, v_body,
                                 math.degrees(self.yaw_rate), self.prev_action)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = self.np_random
        self._make_world()
        for _ in range(100):
            self.x, self.y = self.rng.uniform(30, WORLD - 30, 2)
            if self._free(self.x, self.y, 4.0):
                break
        self.z = float(self.terrain.height(self.x, self.y)) + TARGET_AGL + self.rng.uniform(-2, 2)
        self.yaw = self.rng.uniform(-math.pi, math.pi)
        self.vx = self.vy = self.vz = self.yaw_rate = 0.0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.steps = 0
        self.goals_reached = 0
        self.interventions = 0
        self.memory = ObstacleMemory()
        self._new_goal()
        obs = self._observe()
        self.prev_dist = self.goal_dist
        return obs, {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        forward, yaw_deg, vz_down = action_to_command(action)
        shield_reason = None
        if self.shielded:  # same safety layer as in real flight (airsim_pilot.py)
            self.memory.update(self.x, self.y, self.yaw, self.rays, self.steps * DT)
            alarm = self.memory.collision_ahead(self.x, self.y, self.vx, self.vy)
            (forward, yaw_deg, vz_down), shield_reason = safety_shield(
                self.rays, self.low, self.agl, (forward, yaw_deg, vz_down), speed=math.hypot(self.vx, self.vy),
                memory_alarm=alarm)
            self.interventions += shield_reason is not None
        # first-order response toward the command (AirSim's flight controller is similarly sluggish)
        k = 1 - math.exp(-DT / self.lag)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.vx += (forward * c - self.vx) * k
        self.vy += (forward * s - self.vy) * k
        self.vz += (-vz_down - self.vz) * k
        self.yaw_rate += (math.radians(yaw_deg) - self.yaw_rate) * k
        self.x += self.vx * DT
        self.y += self.vy * DT
        self.z += self.vz * DT
        self.yaw = (self.yaw + self.yaw_rate * DT + math.pi) % (2 * math.pi) - math.pi
        self.steps += 1

        obs = self._observe()
        ground = float(self.terrain.height(self.x, self.y))
        crashed = self.z < ground + 0.4 or not self._free_3d()
        outside = not (5 < self.x < WORLD - 5 and 5 < self.y < WORLD - 5) or self.agl > 30

        reward = (self.prev_dist - self.goal_dist) * 1.0           # progress toward the goal
        reward -= 0.003                                             # time
        reward -= 0.05 * min(abs(self.agl - TARGET_AGL) / 5.0, 2.0)  # hold height above ground
        front = float(self.rays[N_RAYS // 2 - 4:N_RAYS // 2 + 4].min())
        reward -= 0.15 * max(0.0, (5.0 - front) / 5.0)              # keep clear of obstacles
        reward -= 0.03 * float(np.sum((action - self.prev_action) ** 2))  # fly smoothly
        if shield_reason is not None:
            reward -= 0.05                                          # the shield had to step in
        self.prev_action = action
        if self.goal_dist < GOAL_RADIUS:
            reward += 10.0
            self.goals_reached += 1
            self._new_goal()
            obs = self._observe()
        self.prev_dist = self.goal_dist
        terminated = crashed or outside
        if crashed:
            reward -= 20.0
        truncated = self.steps >= self.max_steps
        return obs, float(reward), terminated, truncated, {"crashed": crashed, "goals": self.goals_reached,
                                                            "interventions": self.interventions}

    def _free_3d(self):
        if not len(self.trees):
            return True
        d = np.hypot(self.trees[:, 0] - self.x, self.trees[:, 1] - self.y) - self.trees[:, 2]
        hit = (d < DRONE_RADIUS) & (self.z >= self.trees[:, 3]) & (self.z <= self.trees[:, 4] + DRONE_RADIUS)
        return not hit.any()
