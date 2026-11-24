"""End-to-end test of the Unreal-side flight code without Unreal.

A fake AirSim client stands in for the simulator: it renders real depth images (front camera, 90 deg,
256x144, DepthPerspective) and the downward camera from the training world, reports the drone's state
the way AirSim does (NED coordinates, quaternion orientation) and applies velocity commands. The real
Pilot class (airsim_pilot.py) then flies patrol routes through random forests. This checks the parts a
pure training test cannot: depth image -> rays, image left/right and NED sign conventions, the
obstacle memory and the command path.

    python test_offline.py [--episodes 20]
"""
import argparse
import math
import time
import types

import numpy as np

import airsim
from airsim_pilot import AGL_ROWS, Pilot
from nav_core import (CONTROL_HZ, FOV_DEG, LOW_PITCH_DEG, MAX_RANGE, MID_BAND_DOWN_DEG, MID_BAND_UP_DEG,
                      TARGET_AGL)
from train_world import DT, ForestWorld

W, H = 256, 144
F = (W / 2) / math.tan(math.radians(FOV_DEG / 2))
COLUMN_ANGLES = np.arctan((np.arange(W) + 0.5 - W / 2) / F)  # left (negative) to right


class FakeClient:
    """The subset of airsim.MultirotorClient that Pilot uses, backed by a ForestWorld."""

    def __init__(self, world):
        self.w = world
        self.command = (0.0, 0.0, 0.0)

    def simSetCameraPose(self, *a, **k):
        pass

    def simSetCameraFov(self, *a, **k):
        pass

    def _depth_image(self):
        depth = np.full((H, W), MAX_RANGE * 4, np.float32)
        rows = set(range(int(H / 2 - F * math.tan(math.radians(MID_BAND_UP_DEG))),
                         int(H / 2 + F * math.tan(math.radians(MID_BAND_DOWN_DEG))) + 1, 2))
        low = int(H / 2 + F * math.tan(math.radians(-LOW_PITCH_DEG)))
        rows |= {low - 1, low, low + 1}
        rows |= set(range(H - AGL_ROWS, H))  # height above ground is read from the lowest rows
        for r in sorted(rows):
            pitch = -math.atan((r + 0.5 - H / 2) / F)
            d = self.w._cast(COLUMN_ANGLES, pitch)
            # _cast gives the horizontal-plane distance scaled for the row's pitch; along the pixel's
            # actual ray (column angle too) the distance is longer by 1/cos(column angle)
            depth[r] = d / np.cos(COLUMN_ANGLES)
        return depth

    def simGetImages(self, requests):
        front = self._depth_image()
        agl = self.w.z - float(self.w.terrain.height(self.w.x, self.w.y))
        down = np.full((H, W), agl, np.float32)
        mk = lambda img: types.SimpleNamespace(image_data_float=img.ravel().tolist(), height=H, width=W)
        return [mk(down if r.camera_name == "3" else front) for r in requests]

    def getMultirotorState(self):
        w = self.w
        # world: x = north, y = east, z up  ->  AirSim NED: z down
        q = airsim.to_quaternion(0, 0, w.yaw)
        k = types.SimpleNamespace(
            position=airsim.Vector3r(w.x, w.y, -w.z), linear_velocity=airsim.Vector3r(w.vx, w.vy, -w.vz),
            orientation=q, angular_velocity=airsim.Vector3r(0, 0, w.yaw_rate))
        return types.SimpleNamespace(kinematics_estimated=k)

    def moveByVelocityBodyFrameAsync(self, vx, vy, vz, duration, drivetrain=None, yaw_mode=None):
        self.command = (vx, yaw_mode.yaw_or_rate, vz)


def to_action(command, w):
    """Express the shielded command in the world's action space so ForestWorld.step applies it as-is."""
    from nav_core import MAX_FWD, MAX_VZ, MAX_YAW
    f, y, vz = command
    return np.array([f / MAX_FWD * 2 - 1, y / MAX_YAW, vz / MAX_VZ], dtype=np.float32)


def delay_sensor(pilot, steps):
    """Make the pilot see each depth image `steps` control steps after it was taken, as with the
    background sensor thread in Unreal (it must shift old readings to the current pose)."""
    queue, capture = [], pilot.sensor.capture

    def late_capture():
        capture()
        queue.append(pilot.sensor.latest)
        old = queue.pop(0) if len(queue) > steps else None
        if old is not None:  # the fake world runs faster than real time: state the image's simulated age
            rays, low, agl, _, pose, alt = old
            old = (rays, low, agl, time.monotonic() - steps * DT, pose, alt)
        pilot.sensor.latest = old
    pilot.sensor.capture = late_capture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--policy", default="models/nav_policy.zip")
    parser.add_argument("--latency", type=int, default=3,
                        help="depth images arrive this many control steps late (Unreal: ~0.2-0.4 s)")
    args = parser.parse_args()
    crashes = reached = total_steps = 0
    for e in range(args.episodes):
        world = ForestWorld(seed=20_000 + e, shielded=False)  # the Pilot applies its own shield
        world.reset(seed=20_000 + e)
        client = FakeClient(world)
        pilot = Pilot(client, args.policy)
        delay_sensor(pilot, args.latency)
        route = [(world.x + 30 * math.cos(a), world.y + 30 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 7)[:-1]]
        k, crashed = 0, False
        for step in range(900):
            world.goal = route[k]
            command, reason, dist = pilot.step(route[k], DT)
            _, _, term, _, info = world.step(to_action(client.command, world))
            if dist < 4.0:
                reached += 1
                k = (k + 1) % len(route)
            if term:
                crashed = info["crashed"]
                break
        crashes += crashed
        total_steps += step + 1
        print(f"episode {e:2d}: trees {len(world.trees):4d} | waypoints reached {reached:3d} total | "
              f"{'CRASH' if crashed else 'ok'} after {step + 1} steps", flush=True)
    print(f"\nPilot through fake AirSim: {args.episodes} routes, crash rate {crashes / args.episodes:.0%}, "
          f"{reached} waypoints reached in {total_steps / CONTROL_HZ / 60:.1f} simulated minutes")


if __name__ == "__main__":
    main()
