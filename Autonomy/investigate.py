"""Investigating a sighting: where the person is, and how the drone approaches and circles them.

When the detector confirms a person, the drone stops patrolling and:
  1. turns to face them,
  2. closes to STANDOFF_M (near enough for the detector to resolve a weapon, far enough to stay
     safe), zooming the camera in so the person fills ~PERSON_PX pixels of the image,
  3. circles them slowly at that distance, so the detector sees them from different angles,
  4. resumes the patrol after INVESTIGATE_S seconds, or when the person has been out of sight for
     LOST_S seconds.
Forward motion still goes through the safety shield; the circling motion is checked against the
obstacle memory and the depth rays at the edge of the view, and reverses direction before a tree.

    python investigate.py      # self-check of the camera geometry
"""
import math

import numpy as np

STANDOFF_M = 14.0      # distance kept from the person
BAND_M = 2.5           # +- tolerance around the standoff before closing in
APPROACH_MAX = 3.0     # m/s
ORBIT_SPEED = 1.2      # m/s sideways while circling
INVESTIGATE_S = 45.0
LOST_S = 10.0
COOLDOWN_S = 30.0      # after an investigation, patrol at least this long before the next one
PERSON_PX = 180        # zoom so a standing person is about this tall in the image
PERSON_H = 1.8
FOV_MIN, FOV_MAX = 25.0, 90.0


def person_position(box, image_w, image_h, fov_deg, cam_yaw_deg, cam_pitch_deg, drone_x, drone_y, drone_yaw,
                    agl, range_m=None):
    """World (x, y) of a detected person. box: x, y, width, height in % of the frame (top-left origin).
    The ray through the feet (bottom centre of the box) is intersected with the ground AGL metres
    below the drone; for a ray at or above the horizon the size-based range estimate is used."""
    f = (image_w / 2) / math.tan(math.radians(fov_deg) / 2)
    u = (box["x"] + box["width"] / 2) / 100 * image_w - image_w / 2
    v = (box["y"] + box["height"]) / 100 * image_h - image_h / 2
    p = math.radians(-cam_pitch_deg)                      # look-down angle of the optical axis
    fwd, right, down = f * math.cos(p) - v * math.sin(p), u, f * math.sin(p) + v * math.cos(p)
    heading = drone_yaw + math.radians(cam_yaw_deg) + math.atan2(right, fwd)
    horizontal_dir = math.hypot(fwd, right)
    if down > 0.17 * math.hypot(horizontal_dir, down):    # more than ~10 degrees below the horizon
        dist = agl * horizontal_dir / down
    elif range_m:
        dist = math.sqrt(max(range_m ** 2 - agl ** 2, 1.0))
    else:
        return None
    return drone_x + dist * math.cos(heading), drone_y + dist * math.sin(heading)


def zoom_fov(dist, image_w):
    """Field of view that makes a standing person about PERSON_PX pixels tall at this distance."""
    half = math.atan((image_w / 2) * PERSON_H / (PERSON_PX * max(dist, 1.0)))
    return float(np.clip(math.degrees(2 * half), FOV_MIN, FOV_MAX))


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def investigate_command(target, drone_x, drone_y, drone_yaw, orbit_dir):
    """(forward m/s, sideways m/s, yaw rate deg/s, distance) toward the standoff circle around target."""
    dx, dy = target[0] - drone_x, target[1] - drone_y
    dist = math.hypot(dx, dy)
    error = wrap(math.atan2(dy, dx) - drone_yaw)
    yaw_rate = float(np.clip(math.degrees(error) * 1.5, -40, 40))
    forward = side = 0.0
    if dist > STANDOFF_M + BAND_M:
        if abs(error) < math.radians(30):                 # approach only while facing the person
            forward = float(np.clip(0.5 * (dist - STANDOFF_M), 0.0, APPROACH_MAX))
    else:
        side = ORBIT_SPEED * orbit_dir                    # on the circle (or inside it): circle around
    return forward, side, yaw_rate, dist


def _self_check():
    """A person placed in the world, projected into the camera, must be found back at the same spot."""
    W, H, fov = 1280, 960, 60.0
    for off, pitch, yaw, agl, px, py in [(-8, -30, 0.3, 7, 20, 6), (12, -20, -1.2, 8, -5, -25), (-15, -45, 2.5, 6, -9, 4)]:
        cam_yaw = math.degrees(wrap(math.atan2(py, px) - yaw)) + off   # person `off` degrees from the image centre
        f = (W / 2) / math.tan(math.radians(fov) / 2)
        heading = math.atan2(py, px) - yaw - math.radians(cam_yaw)
        dist = math.hypot(px, py)
        fwd, right, down = dist * math.cos(heading), dist * math.sin(heading), agl  # camera-yawed frame
        p = math.radians(-pitch)
        c_f, c_d = fwd * math.cos(p) + down * math.sin(p), -fwd * math.sin(p) + down * math.cos(p)
        u, v = right / c_f * f, c_d / c_f * f
        box = {"x": (u + W / 2) / W * 100 - 1, "width": 2, "y": (v + H / 2) / H * 100 - 10, "height": 10}
        x, y = person_position(box, W, H, fov, cam_yaw, pitch, 0, 0, yaw, agl)
        assert math.hypot(x - px, y - py) < 0.05, (x, y, px, py)
    fwd, side, yaw_rate, dist = investigate_command((30, 0), 0, 0, 0, 1)
    assert fwd == APPROACH_MAX and side == 0 and dist == 30
    fwd, side, _, _ = investigate_command((14, 0), 0, 0, 0, -1)
    assert fwd == 0 and side == -ORBIT_SPEED
    assert FOV_MIN <= zoom_fov(14, 1280) < 60
    print("investigate.py self-check passed")


if __name__ == "__main__":
    _self_check()
