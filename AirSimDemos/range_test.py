"""Detection range test. Places the drone at increasing slant ranges from the soldier (camera
pointed at him), captures RGB + thermal frames with ground truth from the simulator's
segmentation, and runs the HawkEye detector on each.

    python range_test.py                      # default ranges, 35 and 60 degree look-down angles
    python range_test.py --ranges 20 50 100   # custom ranges (metres)
    python range_test.py --no-detect          # capture only

Needs the simulator running (Play in the editor, or RUN_ALL_HAWKEYE.bat). Results are written to
range_test_results/<time>/: frames, annotated frames, results.json and results.md."""
import argparse
import datetime
import json
import math
import os
import sys
import time

import airsim
import cv2
import numpy as np

from sim_common import CAMERA_FOV_DEG, capture_pair, find_actor, setup_segmentation, thermal_from

HERE = os.path.dirname(os.path.abspath(__file__))


def look_at(client, target, slant, depression_deg, bearing_deg=180.0):
    """Teleport the drone to `slant` metres from target, `depression_deg` above the horizon as seen
    from the target, and point the camera at it. bearing_deg is the compass direction from the
    target to the drone (world frame)."""
    dep = math.radians(depression_deg)
    horiz, up = slant * math.cos(dep), slant * math.sin(dep)
    b = math.radians(bearing_deg)
    pos = airsim.Vector3r(target.x_val + horiz * math.cos(b), target.y_val + horiz * math.sin(b), target.z_val - up)
    yaw = math.atan2(target.y_val - pos.y_val, target.x_val - pos.x_val)
    client.simSetVehiclePose(airsim.Pose(pos, airsim.to_quaternion(0, 0, yaw)), True)
    client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(-dep, 0, 0)))


def truth_boxes(body_mask, weapon_mask):
    out = {}
    for name, mask in (("person", body_mask), ("weapon", weapon_mask)):
        ys, xs = np.where(mask > 0)
        out[name] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
    # the rifle overlaps the body in the mask; the person box covers both
    if out["person"] and out["weapon"]:
        p, w = out["person"], out["weapon"]
        out["person"] = [min(p[0], w[0]), min(p[1], w[1]), max(p[2], w[2]), max(p[3], w[3])]
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ranges", type=float, nargs="+", default=[10, 20, 30, 40, 60, 80, 100, 130, 160, 200])
    parser.add_argument("--angles", type=float, nargs="+", default=[35, 60], help="look-down angles (degrees)")
    parser.add_argument("--views", nargs="+", default=["side"], choices=["front", "side", "behind"],
                        help="where the drone is relative to the way the soldier faces")
    parser.add_argument("--fov", type=float, nargs="+", default=[90, 25],
                        help="camera field of view (degrees): 90 = wide, smaller = optical zoom")
    parser.add_argument("--repeats", type=int, default=3, help="frames per position (sensor noise differs)")
    parser.add_argument("--no-detect", action="store_true")
    args = parser.parse_args()

    client = airsim.MultirotorClient()
    client.confirmConnection()
    setup_segmentation(client)
    character = find_actor(client, "ThirdPersonCharacter")
    if character is None:
        sys.exit("No ThirdPersonCharacter in the scene.")
    pose = client.simGetObjectPose(character)
    target = pose.position
    facing = math.degrees(airsim.to_eularian_angles(pose.orientation)[2])
    view_offset = {"front": 0.0, "side": 90.0, "behind": 180.0}
    home = client.simGetVehiclePose()

    out_dir = os.path.join(HERE, "range_test_results", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    detector = annotate = None
    if not args.no_detect:
        import albumentations  # noqa: F401  (must load before torch on this machine)
        sys.path.insert(0, os.path.join(HERE, "..", "project", "backend"))
        from app.config import get_settings
        from app.detector import ThreatDetector
        from app.media import annotate_frame as annotate
        from app.views import estimate_ranges
        detector = ThreatDetector(get_settings())
        detector.load()
        if not detector.ready:
            sys.exit(f"Detector not ready: {detector.status_message}")

    rows = []
    try:
        for fov, view in ((f, v) for f in args.fov for v in args.views):
            bearing = facing + view_offset[view]
            client.simSetCameraFov("0", fov)
            telemetry = {"camera_fov_deg": fov}
            for angle, slant in ((a, r) for a in args.angles for r in args.ranges):
                look_at(client, target, slant, angle, bearing)
                time.sleep(0.6)  # let the renderer catch up with the teleport
                row = {"range_m": slant, "angle_deg": angle, "altitude_m": round(slant * math.sin(math.radians(angle)), 1),
                       "fov_deg": fov, "view": view, "frames": []}
                telemetry["camera_pitch_deg"] = -angle
                for k in range(args.repeats):
                    look_at(client, target, slant, angle, bearing)  # hold position (no flight controller while testing)
                    time.sleep(0.25)
                    pair = capture_pair(client)
                    if pair is None:
                        continue
                    rgb, seg = pair
                    thermal, body_mask, weapon_mask, _ = thermal_from(rgb, seg)
                    truth = truth_boxes(body_mask, weapon_mask)
                    tag = f"fov{int(fov)}_{view}_a{int(angle)}_r{int(slant):03d}_{k}"
                    cv2.imwrite(os.path.join(out_dir, tag + "_rgb.png"), rgb)
                    cv2.imwrite(os.path.join(out_dir, tag + "_thermal.png"), thermal)
                    frame = {"tag": tag, "truth": truth,
                             "person_px": (truth["person"][3] - truth["person"][1]) if truth["person"] else 0,
                             "weapon_px": int((weapon_mask > 0).sum())}
                    if detector is not None:
                        payload, display = detector.process_views(rgb, thermal, 0, "range-test", return_display=True)
                        h, w = display.shape[:2]
                        boxes = [(d, [d.bbox.x / 100 * w, d.bbox.y / 100 * h, (d.bbox.x + d.bbox.width) / 100 * w,
                                      (d.bbox.y + d.bbox.height) / 100 * h]) for d in payload.detections]
                        estimate_ranges(payload.detections, {**telemetry, "image_width": w, "image_height": h})
                        people = [(d, b) for d, b in boxes if d.kind == "person"]
                        hit = [d for d, b in people if truth["person"] and iou(b, truth["person"]) >= 0.3]
                        frame.update(method=payload.method, level=payload.summary.level.value,
                                     person_found=bool(hit), person_conf=round(max((d.confidence for d in hit), default=0), 2),
                                     people_reported=payload.summary.total_people,
                                     armed=payload.summary.armed_people > 0,
                                     weapon_evidence=next((d.armed_evidence for d in hit if d.armed), None),
                                     possible_armed=payload.summary.possible_armed > 0,
                                     est_range_m=next((d.range_m for d in hit if getattr(d, "range_m", None)), None))
                        cv2.imwrite(os.path.join(out_dir, tag + "_detected.jpg"), annotate(display, payload.detections))
                    row["frames"].append(frame)
                rows.append(row)
                f = row["frames"]
                if f and detector is not None:
                    est = [x["est_range_m"] for x in f if x.get("est_range_m")]
                    print(f"fov {fov:.0f} {view:6s} {slant:5.0f} m @ {angle:.0f} deg | person {f[0]['person_px']:4d} px | "
                          f"est {(sum(est) / len(est)) if est else 0:5.0f} m | found "
                          f"{sum(x['person_found'] for x in f)}/{len(f)} | armed {sum(x['armed'] for x in f)}/{len(f)} | "
                          f"possible {sum(x['possible_armed'] for x in f)}/{len(f)} | reported people "
                          f"{[x['people_reported'] for x in f]} | {f[0]['method']}")
                elif f:
                    print(f"{slant:5.0f} m @ {angle:.0f} deg | person {f[0]['person_px']} px | rifle {f[0]['weapon_px']} px")
    finally:
        client.simSetCameraFov("0", CAMERA_FOV_DEG)
        client.simSetVehiclePose(home, True)
        client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(-20), 0, 0)))

    json.dump({"camera_fov_deg": CAMERA_FOV_DEG, "rows": rows}, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    if detector is not None:
        lines = ["| Lens (FOV) | View | Range | Look-down | Altitude | Person height | Person found | Armed (weapon seen) | Possible weapon flag |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for row in rows:
            f = row["frames"]
            if not f:
                continue
            n = len(f)
            lines.append(f"| {row['fov_deg']:.0f}° | {row['view']} | {row['range_m']:.0f} m | {row['angle_deg']:.0f}° | {row['altitude_m']:.0f} m | "
                         f"{f[0]['person_px']} px | {sum(x['person_found'] for x in f)}/{n} | "
                         f"{sum(x['armed'] for x in f)}/{n} | {sum(x['possible_armed'] for x in f)}/{n} |")
        open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("Results in", out_dir)


if __name__ == "__main__":
    main()
