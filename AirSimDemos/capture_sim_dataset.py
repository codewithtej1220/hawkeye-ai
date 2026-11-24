"""Capture an auto-labelled training set from the forest simulation.

The detector was trained on real footage, where an openly carried rifle seen by a thermal camera
is rare, so it missed the simulated soldier's rifle. The simulator can label such frames exactly:
its segmentation gives the soldier's and the rifle's pixels. This script moves the drone to random
viewpoints (all bearings, look-down angles, ranges and zoom levels) while the soldier performs his
routine, and saves thermal (white-hot and INFERNO) and colour frames with YOLO labels.

    python capture_sim_dataset.py --count 1500 --out ..\\HawkEye_Training\\sim_capture

The result is mixed with the real data by HawkEye_Training/build_sim_mix.py; it never replaces it.
Needs the simulator running."""
import argparse
import math
import os
import random
import time

import airsim
import cv2
import numpy as np

from sim_actor import SoldierDirector
from sim_common import CAMERA_FOV_DEG, capture_pair, find_actor, masks_from_segmentation, render_whitehot, setup_segmentation

ZOOMS = [90.0, 90.0, 60.0, 40.0, 25.0, 15.0]


def box_of(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def yolo_line(cls, box, w, h):
    x1, y1, x2, y2 = box
    return f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=1500, help="viewpoints to capture")
    parser.add_argument("--out", default=os.path.join("..", "HawkEye_Training", "sim_capture"))
    parser.add_argument("--negatives", type=float, default=0.1, help="share of frames with the soldier out of view")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--unarmed", action="store_true",
                        help="hide the rifle: unarmed-soldier frames, so a model does not learn that every "
                             "person seen by the thermal camera carries a weapon")
    parser.add_argument("--start-index", type=int, default=0, help="first file number (to add to an existing set)")
    args = parser.parse_args()
    random.seed(args.seed)

    client = airsim.MultirotorClient()
    client.confirmConnection()
    setup_segmentation(client)
    character = find_actor(client, "ThirdPersonCharacter")
    home = client.simGetVehiclePose()
    weapon = find_actor(client, "BP_Weapon")
    if args.unarmed and weapon:
        client.simSetObjectScale(weapon, airsim.Vector3r(0.001, 0.001, 0.001))
    director = SoldierDirector(verbose=False).start()

    for sub in ("images", "labels"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    saved = with_weapon = 0
    t0 = time.time()
    try:
        for i in range(args.count):
            target = client.simGetObjectPose(character).position
            fov = random.choice(ZOOMS)
            focal = 640 / math.tan(math.radians(fov) / 2)
            # ranges that give a person between ~12 px and ~500 px tall at this zoom
            slant = math.exp(random.uniform(math.log(max(6.0, 1.75 * focal / 500)), math.log(min(260.0, 1.75 * focal / 12))))
            dep = math.radians(random.uniform(12, 78))
            bearing = random.uniform(0, 2 * math.pi)
            horiz, up = slant * math.cos(dep), slant * math.sin(dep)
            pos = airsim.Vector3r(target.x_val + horiz * math.cos(bearing), target.y_val + horiz * math.sin(bearing),
                                  target.z_val - up)
            yaw = math.atan2(target.y_val - pos.y_val, target.x_val - pos.x_val)
            negative = random.random() < args.negatives
            # the soldier is not centred: aim off by up to ~35% of the field of view (negatives: look away)
            off = math.radians(fov) * (random.uniform(0.9, 1.6) * random.choice((-1, 1)) if negative else random.uniform(-0.35, 0.35))
            off_pitch = 0.0 if negative else math.radians(fov) * 0.75 * random.uniform(-0.3, 0.3)
            client.simSetCameraFov("0", fov)
            client.simSetVehiclePose(airsim.Pose(pos, airsim.to_quaternion(0, 0, yaw + off)), True)
            client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(-dep + off_pitch, 0, 0)))
            time.sleep(0.35)
            pair = capture_pair(client)
            if pair is None:
                continue
            rgb, seg = pair
            h, w = rgb.shape[:2]
            body_mask, weapon_mask, _ = masks_from_segmentation(seg)
            body_box, weapon_box = box_of(body_mask), box_of(weapon_mask)
            labels = []
            if body_box and (body_box[3] - body_box[1]) >= 8 and (body_box[2] - body_box[0]) >= 3:
                # a soldier cut off by the frame edge is still a person, but skip slivers
                labels.append(yolo_line(0, body_box, w, h))
                if weapon_box and int((weapon_mask > 0).sum()) >= 12 and max(weapon_box[2] - weapon_box[0], weapon_box[3] - weapon_box[1]) >= 8:
                    labels.append(yolo_line(1, weapon_box, w, h))
            elif body_box:
                continue  # too small / clipped to label reliably: do not teach it as background
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            whitehot = render_whitehot(gray, body_mask, weapon_mask)
            # real thermal cores are lower resolution than colour cameras: vary the sharpness
            scale = random.choice((1.0, 0.75, 0.5, 0.5))
            if scale < 1.0:
                whitehot = cv2.resize(whitehot, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            inferno = cv2.applyColorMap(cv2.cvtColor(whitehot, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)
            stem = f"{args.start_index + i:05d}"
            outputs = [("thermal__simgrey__" + stem, whitehot), ("thermal__siminferno__" + stem, inferno)]
            if random.random() < 0.35:
                outputs.append(("rgb__sim__" + stem, rgb))
            for name, image in outputs:
                cv2.imwrite(os.path.join(args.out, "images", name + ".jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                with open(os.path.join(args.out, "labels", name + ".txt"), "w") as f:
                    f.write("\n".join(labels) + ("\n" if labels else ""))
            saved += 1
            with_weapon += len(labels) == 2
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{args.count} viewpoints | saved {saved} | with rifle label {with_weapon} | "
                      f"soldier: {director.action} | {time.time() - t0:.0f} s", flush=True)
    finally:
        director.stop()
        if args.unarmed and weapon:
            client.simSetObjectScale(weapon, airsim.Vector3r(1, 1, 1))
        client.simSetCameraFov("0", CAMERA_FOV_DEG)
        client.simSetVehiclePose(home, True)
        client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(-20), 0, 0)))
    print(f"Done: {saved} viewpoints saved to {os.path.abspath(args.out)} ({with_weapon} with a rifle label)")


if __name__ == "__main__":
    main()
