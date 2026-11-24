"""Autonomous patrol: the drone flies a patrol route through the forest by itself, avoiding trees and
terrain with the learned policy, while surveillance never stops: a separate thread streams the
surveillance camera (thermal + colour) to the HawkEye backend for people/weapon detection.

    python autonomous_patrol.py                  # patrol until Ctrl+C
    python autonomous_patrol.py --minutes 5      # timed run, then a report (models/patrol_report.json)
    python autonomous_patrol.py --radius 30 --points 6

Behaviour:
  1. take off and climb to the policy's height above ground (TARGET_AGL)
  2. fly waypoint to waypoint around the patrol area; the policy picks speed, turn and climb, the
     safety shield overrides it near obstacles; the surveillance camera keeps looking at the centre
     of the patrol area
  3. a detection makes the drone hover and look again; a person seen in at least 2 of the last 3
     frames is a confirmed sighting: the drone turns to them, closes to a safe distance with the
     camera zoomed in, and circles them to see them from different angles (investigate.py), then
     resumes the route
Needs: the simulator playing, and the HawkEye backend running (http://127.0.0.1:8000) for detection.
The patrol still flies without the backend; detection resumes as soon as it is reachable.
"""
import argparse
import base64
import collections
import json
import math
import os
import sys
import threading
import time

import airsim
import cv2
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "AirSimDemos"))
from sim_common import find_actor, setup_segmentation, thermal_from, telemetry  # noqa: E402

import investigate as inv  # noqa: E402
from airsim_pilot import Pilot  # noqa: E402
from nav_core import CONTROL_HZ, TARGET_AGL, safety_shield  # noqa: E402

BACKEND = os.environ.get("HAWKEYE_BACKEND_URL", "http://127.0.0.1:8000")
SURVEY_CAMERA = "1"
WAYPOINT_RADIUS = 4.0
RECHECK_S = 2.5            # hover this long to look again at an unconfirmed detection
RECHECK_COOLDOWN_S = 4.0   # then fly at least this long before re-checking again


class Surveillance(threading.Thread):
    """Streams the surveillance camera to the detector, independently of navigation. The main loop
    sets where the camera looks (self.camera: yaw / pitch relative to the drone, field of view); this
    thread applies it right before each capture, so every frame's telemetry and the position of a
    person found in it use the camera pose the frame was taken with."""

    def __init__(self, fps=2.0):
        super().__init__(daemon=True)
        self.fps = fps
        self.running = True
        self.camera = {"yaw": 0.0, "pitch": -35.0, "fov": 90.0}
        self.mode = "autonomous patrol"
        self.agl = TARGET_AGL
        self.sent = self.analysed = self.frames_with_people = 0
        self.last_detection_at = 0.0
        self.last_confirmed_at = 0.0
        self.confirmed_frames = 0
        self.recent = collections.deque(maxlen=3)  # people in each of the last frames?
        self.people = []        # world (x, y, armed, time) of people in the latest analysed frame
        self.armed_frames = 0
        self.backend_up = None
        self.soldier = None

    def run(self):
        c = airsim.MultirotorClient()
        c.confirmConnection()
        self.soldier = find_actor(c, "ThirdPersonCharacter")
        session = requests.Session()
        frame_id, applied = 0, None
        while self.running:
            started = time.time()
            try:
                cam = dict(self.camera)
                if cam != applied:
                    c.simSetCameraPose(SURVEY_CAMERA, airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(
                        math.radians(cam["pitch"]), 0, math.radians(cam["yaw"]))))
                    c.simSetCameraFov(SURVEY_CAMERA, cam["fov"])
                    applied = cam
                state = c.getMultirotorState().kinematics_estimated
                _, _, yaw = airsim.to_eularian_angles(state.orientation)
                pos, agl = state.position, self.agl
                r = c.simGetImages([airsim.ImageRequest(SURVEY_CAMERA, airsim.ImageType.Scene, False, False),
                                    airsim.ImageRequest(SURVEY_CAMERA, airsim.ImageType.Segmentation, False, False)])
                rgb = np.frombuffer(r[0].image_data_uint8, np.uint8).reshape(r[0].height, r[0].width, 3)
                seg = np.frombuffer(r[1].image_data_uint8, np.uint8).reshape(r[1].height, r[1].width, 3)
                if seg.shape != rgb.shape:
                    seg = cv2.resize(seg, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                thermal = thermal_from(rgb, seg)[0]
                data = telemetry(c, cam["pitch"], self.soldier, 0.0)
                data.update(camera_fov_deg=round(cam["fov"], 1), mode=self.mode,
                            zoom=round(math.tan(math.radians(45)) / math.tan(math.radians(cam["fov"] / 2)), 1),
                            heading_deg=round(math.degrees(yaw + math.radians(cam["yaw"])) % 360),
                            # the take-off point is not the ground below a drone flying over hills
                            altitude_m=round(agl, 1), altitude_reference="Above ground")
                frame_id += 1
                body = {"image": jpeg(thermal), "rgb": jpeg(rgb), "frame_id": frame_id, "method": "auto",
                        "telemetry": data}
                response = session.post(f"{BACKEND}/detect", json=body, timeout=10)
                self.sent += 1
                if self.backend_up is not True:
                    print(f"[surveillance] streaming to {BACKEND}")
                    self.backend_up = True
                result = response.json() if response.status_code == 200 else {}
                if result.get("status") == "ok":  # not "skipped" (detector busy)
                    self.analysed += 1
                    self._record(result, rgb.shape, cam, pos, yaw, agl)
            except requests.RequestException:
                if self.backend_up is not False:
                    print(f"[surveillance] backend not reachable at {BACKEND}; retrying (patrol continues)")
                    self.backend_up = False
                time.sleep(1.0)
            except Exception as exc:  # simulator hiccup: keep surveilling
                print(f"[surveillance] {type(exc).__name__}: {exc}")
                time.sleep(0.5)
            time.sleep(max(0.0, 1.0 / self.fps - (time.time() - started)))

    def _record(self, result, shape, cam, pos, yaw, agl):
        seen = result.get("detections", 0) > 0
        self.recent.append(seen)
        now = time.time()
        people = []
        for p in result.get("people", []):
            xy = inv.person_position(p, shape[1], shape[0], cam["fov"], cam["yaw"], cam["pitch"],
                                     pos.x_val, pos.y_val, yaw, agl, p.get("range_m"))
            if xy:
                people.append((xy[0], xy[1], p.get("armed", False), now))
        self.people = people
        if result.get("threat") in ("High", "Critical"):
            self.armed_frames += 1
        if seen:
            self.frames_with_people += 1
            self.last_detection_at = now
            # persistence: one stray frame is not a sighting; 2 of the last 3 frames is
            if sum(self.recent) >= 2:
                self.confirmed_frames += 1
                self.last_confirmed_at = now


def jpeg(image):
    return base64.b64encode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])[1]).decode()


def patrol_route(centre, radius, points):
    return [(centre[0] + radius * math.cos(2 * math.pi * k / points),
             centre[1] + radius * math.sin(2 * math.pi * k / points)) for k in range(points)]


def look_at(drone_x, drone_y, drone_yaw, agl, point):
    """Camera yaw / pitch (degrees, relative to the drone) that look at a ground point."""
    dx, dy = point[0] - drone_x, point[1] - drone_y
    yaw = math.degrees(inv.wrap(math.atan2(dy, dx) - drone_yaw))
    pitch = -math.degrees(math.atan2(agl, max(math.hypot(dx, dy), 1.0)))
    return float(np.clip(yaw, -110, 110)), float(np.clip(pitch, -60, -8))


class Investigation:
    """Approach and circle one person (see investigate.py)."""

    def __init__(self, target, now):
        self.target = target          # (x, y) world, smoothed
        self.started = self.last_seen = now
        self.orbit_dir = 1
        self.flip_allowed_at = now
        self.closest = 99.0

    def update_target(self, people, now):
        if not people:
            return
        best = min(people, key=lambda p: math.hypot(p[0] - self.target[0], p[1] - self.target[1]))
        if math.hypot(best[0] - self.target[0], best[1] - self.target[1]) < 12:  # same person
            self.target = (0.6 * self.target[0] + 0.4 * best[0], 0.6 * self.target[1] + 0.4 * best[1])
            self.last_seen = best[3]

    def done(self, now):
        return now - self.started > inv.INVESTIGATE_S or now - self.last_seen > inv.LOST_S

    def step(self, client, pilot, dt):
        """One control step; returns the shield's reason or None."""
        state = client.getMultirotorState().kinematics_estimated
        pilot.sense(state)
        p, v = state.position, state.linear_velocity
        _, _, yaw = airsim.to_eularian_angles(state.orientation)
        forward, side, yaw_rate, dist = inv.investigate_command(self.target, p.x_val, p.y_val, yaw, self.orbit_dir)
        self.closest = min(self.closest, dist)
        vz = float(np.clip(0.5 * (pilot.agl - TARGET_AGL), -1.0, 1.0))  # NED: positive is down
        (forward, _, vz), reason = safety_shield(pilot.rays, pilot.low, pilot.agl, (forward, yaw_rate, vz),
                                                 speed=math.hypot(v.x_val, v.y_val), lag=0.6 + pilot.depth_age)
        if side:
            # circling: the camera looks at the person, not along the sideways motion, so check the
            # rays at that edge of the view and the obstacle memory; turn back before a tree
            edge = pilot.rays[-4:] if side > 0 else pilot.rays[:4]
            c, s = math.cos(yaw), math.sin(yaw)
            vx, vy = -s * side, c * side
            blocked = edge.min() < 5.0 or pilot.memory.collision_ahead(p.x_val, p.y_val, vx, vy, horizon=2.0)
            if blocked:
                side = 0.0
                if time.time() > self.flip_allowed_at:
                    self.orbit_dir, self.flip_allowed_at = -self.orbit_dir, time.time() + 3.0
                    reason = reason or "circling blocked: reverse"
        pilot.memory.update(p.x_val, p.y_val, yaw, pilot.rays, time.monotonic())
        client.moveByVelocityBodyFrameAsync(forward, side, vz, duration=max(dt * 3, 0.5),
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate))
        return reason, dist, (p.x_val, p.y_val, yaw)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=0, help="stop after this long (0 = until Ctrl+C)")
    parser.add_argument("--radius", type=float, default=30.0, help="patrol circle radius (m)")
    parser.add_argument("--points", type=int, default=6, help="waypoints on the circle")
    parser.add_argument("--policy", default=os.path.join(HERE, "models", "nav_policy.zip"))
    args = parser.parse_args()

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    setup_segmentation(client)
    client.simRunConsoleCommand("t.IdleWhenNotForeground 0")
    client.simRunConsoleCommand("DisableAllScreenMessages")
    soldier = find_actor(client, "ThirdPersonCharacter")
    volume = find_actor(client, "NavMeshBoundsVolume")
    anchor = client.simGetObjectPose(volume or soldier).position if (volume or soldier) else client.simGetVehiclePose().position
    centre = (anchor.x_val, anchor.y_val)
    route = patrol_route(centre, args.radius, args.points)

    try:
        from sim_actor import SoldierDirector
        SoldierDirector(verbose=False).start()
    except Exception as exc:
        print(f"Soldier routine not started: {exc}")
    survey = Surveillance()
    survey.start()

    sensor_client = airsim.MultirotorClient()
    sensor_client.confirmConnection()
    pilot = Pilot(client, args.policy, sensor_client=sensor_client)
    print("Taking off...")
    client.takeoffAsync().join()
    dt = 1.0 / CONTROL_HZ
    stats = {"waypoints_reached": 0, "laps": 0, "collisions": 0, "shield_interventions": 0, "steps": 0,
             "min_obstacle_m": 99.0, "rechecks": 0, "confirmed_sightings": 0, "investigations": 0,
             "investigation_closest_m": [], "depth_holds": 0}
    depth_ages = []
    started = time.time()
    k = 0
    last_collision_stamp = client.simGetCollisionInfo().time_stamp
    hover_until = 0.0
    investigation, next_investigation_at = None, 0.0
    print(f"Patrolling {args.points} waypoints, {args.radius:.0f} m around the patrol area, ~{TARGET_AGL:.0f} m above ground")
    try:
        while not args.minutes or time.time() - started < args.minutes * 60:
            tick = time.time()
            survey.agl = pilot.agl
            # ---- decide: investigate a confirmed person, hover to re-check a possible one, or patrol
            if investigation is None and survey.last_confirmed_at > tick - 1.0 and tick > next_investigation_at \
                    and survey.people:
                state = client.getMultirotorState().kinematics_estimated
                me = (state.position.x_val, state.position.y_val)
                nearest = min(survey.people, key=lambda q: math.hypot(q[0] - me[0], q[1] - me[1]))
                investigation = Investigation((nearest[0], nearest[1]), tick)
                stats["confirmed_sightings"] += 1
                print(f"[patrol] person confirmed (2 of 3 frames) {math.hypot(nearest[0] - me[0], nearest[1] - me[1]):.0f} m "
                      f"away: approaching to {inv.STANDOFF_M:.0f} m and circling to look from all sides")
            elif investigation is None and survey.last_detection_at > tick - 1.0 and tick > hover_until + RECHECK_COOLDOWN_S:
                hover_until = tick + RECHECK_S
                stats["rechecks"] += 1
                print("[patrol] possible person: hovering to look again")

            if investigation is not None:
                investigation.update_target(survey.people, tick)
                if investigation.done(tick):
                    stats["investigations"] += 1
                    stats["investigation_closest_m"].append(round(investigation.closest, 1))
                    print(f"[patrol] investigation over (closest {investigation.closest:.1f} m); resuming the patrol")
                    investigation, next_investigation_at = None, tick + inv.COOLDOWN_S
                    survey.mode = "autonomous patrol"
                else:
                    survey.mode = "investigating person"
                    reason, dist, (x, y, yaw) = investigation.step(client, pilot, dt)
                    # camera straight ahead (the drone faces the person), aimed down at them, zoomed in
                    _, pitch = look_at(x, y, yaw, pilot.agl, investigation.target)
                    survey.camera = {"yaw": 0.0, "pitch": pitch, "fov": inv.zoom_fov(math.hypot(dist, pilot.agl), 1280)}
                    if reason:
                        stats["shield_interventions"] += 1
            elif tick < hover_until:
                pilot.hold(dt)
                pilot.sense()
            else:
                goal = route[k]
                command, reason, dist = pilot.step(goal, dt)
                if reason and reason.startswith("no fresh depth"):
                    stats["depth_holds"] += 1
                elif reason:
                    stats["shield_interventions"] += 1
                if dist < WAYPOINT_RADIUS:
                    stats["waypoints_reached"] += 1
                    k = (k + 1) % len(route)
                    if k == 0:
                        stats["laps"] += 1
                    print(f"[patrol] waypoint reached ({stats['waypoints_reached']} total); next {k + 1}/{len(route)}")
            if investigation is None:
                # patrolling or hovering: the surveillance camera watches the centre of the patrol area
                state = client.getMultirotorState().kinematics_estimated
                _, _, yaw = airsim.to_eularian_angles(state.orientation)
                cam_yaw, pitch = look_at(state.position.x_val, state.position.y_val, yaw, pilot.agl, centre)
                survey.camera = {"yaw": round(cam_yaw), "pitch": round(pitch), "fov": 90.0}
            depth_ages.append(pilot.depth_age)
            stats["steps"] += 1
            stats["min_obstacle_m"] = min(stats["min_obstacle_m"], pilot.nearest_obstacle)
            collision = client.simGetCollisionInfo()
            if collision.has_collided and collision.time_stamp != last_collision_stamp:
                last_collision_stamp = collision.time_stamp
                stats["collisions"] += 1
                print(f"[patrol] COLLISION with {collision.object_name}")
            time.sleep(max(0.0, dt - (time.time() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        survey.running = False
        pilot.stop()
        client.hoverAsync().join()
        stats.update(minutes=round((time.time() - started) / 60, 1), detection_frames_sent=survey.sent,
                     detection_frames_analysed=survey.analysed, frames_with_people=survey.frames_with_people,
                     frames_with_confirmed_people=survey.confirmed_frames, frames_armed=survey.armed_frames,
                     control_hz=round(stats["steps"] / max(time.time() - started, 1e-6), 1),
                     depth_frames=pilot.sensor.frames,
                     median_depth_age_s=round(float(np.median(depth_ages)), 2) if depth_ages else None,
                     min_obstacle_m=round(stats["min_obstacle_m"], 1))
        os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
        json.dump(stats, open(os.path.join(HERE, "models", "patrol_report.json"), "w"), indent=1)
        print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
