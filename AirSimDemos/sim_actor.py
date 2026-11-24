"""Directs the soldier in the forest simulation: a repeating routine of patrol actions.

The soldier's Unreal blueprint (ThirdPersonCharacter) already contains the behaviours:
  * startWander        - walk to a random reachable point on the navigation mesh, wait, repeat
  * SetAnimationState  - play one of its 49 rifle animations (idle, aim, crouch, run, ...)
This script triggers them over AirSim's console channel and sets his movement speed to match, so
he runs, halts and aims, crouch-walks, crouches and aims, and looks around, in a loop.

    python sim_actor.py            # run the routine until Ctrl+C (the simulator must be playing)

thermal.py starts the same routine in the background (set HAWKEYE_SOLDIER=off to disable)."""
import math
import threading
import time

import airsim

# (action name, animation, index in the blueprint's Animation List, speed in cm/s, seconds)
ROUTINE = [
    # speeds match the distance each animation's steps cover (184 / 195 cm per 1 s cycle), so the feet
    # do not slide. These Mixamo animations move their root bone (Hips) forward and snap back each cycle;
    # they are set to "force root lock" in Unreal so they play in place. walk_forward was added to the
    # blueprint's list as index 49 (the list had no plain forward walk)
    ("patrolling",         "walk_forward",           49, 184, 14),
    ("halt and aim",       "idle_aiming",             9,   0,  6),
    ("sneaking (crouched)", "walk_crouching_forward", 39, 195, 12),
    ("crouch and aim",     "idle_crouching_aiming",  11,   0,  6),
    ("looking around",     "idle",                    8,   0,  4),
]
# A nudge is a teleport, which looks like a glitch, so it is the last resort: the blueprint's wander
# loop pauses at each point it reaches, and the navigation area keeps him near his post by itself.
STUCK_AFTER_S = 15.0   # moving action but no movement for this long: path finding failed, nudge
NUDGE_M = 3.0
LEASH_M = 25.0         # beyond this distance from his post he is nudged back toward it
NUDGE_EVERY_S = 10.0   # at most one nudge per this many seconds
NUDGE_LIFT_M = 2.5     # placed this far above his old height, he lands on a slope up to ~40 degrees
FELL_M = 30.0          # this far below his post's height: he has fallen through the terrain
RECOVER_DROP_M = 12.0


class SoldierDirector:
    def __init__(self, client=None, verbose=True):
        self.client = client or airsim.MultirotorClient()
        if client is None:
            self.client.confirmConnection()
        self.verbose = verbose
        self.action = "starting"
        self._stop = threading.Event()
        self._wander_started = False
        self._last_nudge = 0.0
        self._thread = None
        objects = self.client.simListSceneObjects()
        self.name = next((o for o in objects if "thirdpersoncharacter" in o.lower()), None)
        self.home = self._position() if self.name else None
        # his post is the centre of the patrol (navigation) area, wherever an earlier run left him
        volume = next((o for o in objects if "navmeshboundsvolume" in o.lower()), None)
        if self.home and volume:
            v = self.client.simGetObjectPose(volume).position
            if not math.isnan(v.x_val):
                # his own height is only trusted while he is near the area's height: an earlier run
                # may have left him fallen through the terrain (z thousands of metres down)
                z = self.home[2] if abs(self.home[2] - v.z_val) < FELL_M else v.z_val
                self.home = (v.x_val, v.y_val, z)
        if self.home and abs(self._position()[2] - self.home[2]) > FELL_M:
            self._recover()

    def _position(self):
        p = self.client.simGetObjectPose(self.name).position
        return (p.x_val, p.y_val, p.z_val)

    def _console(self, command):
        self.client.simRunConsoleCommand(command)

    def _perform(self, action, animation, index, speed):
        self.action = action
        self._console(f"set CharacterMovementComponent MaxWalkSpeed {speed}")
        self._console(f"ke * SetAnimationState {animation} {index}")
        if self.verbose:
            print(f"[soldier] {action}")

    def _nudge_home(self):
        """Path-finding fails when he steps off the edge of the nav mesh: move him a few metres back
        toward the middle of the patrol area and start walking again."""
        x, y, z = self._position()
        dx, dy = self.home[0] - x, self.home[1] - y
        dist = math.hypot(dx, dy) or 1.0
        pose = self.client.simGetObjectPose(self.name)
        step = min(NUDGE_M, dist)
        # placed above the spot he lands on the ground under it; placed level with his old
        # height he can end up inside a rising slope and fall through the terrain
        # relative to his own height: lifting him to his post's height when he is downhill of it
        # dropped him several metres every time
        pose.position = airsim.Vector3r(x + dx / dist * step, y + dy / dist * step, z - NUDGE_LIFT_M)
        self.client.simSetObjectPose(self.name, pose, True)
        self._last_nudge = time.monotonic()
        if self.verbose:
            print("[soldier] nudged back toward his post")

    def _recover(self):
        """He fell through the terrain (or out of the world): put him back above his post; gravity
        lands him on the ground."""
        pose = self.client.simGetObjectPose(self.name)
        pose.position = airsim.Vector3r(self.home[0], self.home[1], self.home[2] - RECOVER_DROP_M)
        self.client.simSetObjectPose(self.name, pose, True)
        if self.verbose:
            print("[soldier] fell out of the world: returned to his post")

    def _ensure_wandering(self):
        """Start the blueprint's wander loop if it is not running. It must be started only once:
        every extra startWander adds another loop, and the loops then fight over where he walks."""
        before = self._position()
        if self._stop.wait(2.5):
            return
        if math.dist(before, self._position()) < 0.3 and not self._wander_started:
            self._console("ke * startWander")
            self._wander_started = True

    def run(self):
        if self.name is None:
            print("[soldier] No ThirdPersonCharacter in the scene; routine not started.")
            return
        while not self._stop.is_set():
            for action, animation, index, speed, seconds in ROUTINE:
                if self._stop.is_set():
                    break
                try:
                    self._perform(action, animation, index, speed)
                    if speed > 0:
                        self._ensure_wandering()
                    last, still_since, end = self._position(), time.monotonic(), time.monotonic() + seconds
                    while time.monotonic() < end and not self._stop.wait(0.5):
                        now = self._position()
                        if math.isnan(now[2]) or now[2] - self.home[2] > FELL_M:
                            self._recover()
                            still_since, last = time.monotonic(), self._position()
                            continue
                        may_nudge = time.monotonic() - self._last_nudge > NUDGE_EVERY_S
                        if math.dist(now, last) > 0.3 or speed == 0:
                            still_since = time.monotonic()
                        if may_nudge and math.hypot(now[0] - self.home[0], now[1] - self.home[1]) > LEASH_M:
                            self._nudge_home()  # strayed too far from his post
                            still_since = time.monotonic()
                        elif may_nudge and time.monotonic() - still_since > STUCK_AFTER_S:
                            self._nudge_home()  # path finding failed (off the edge of the nav mesh)
                            still_since = time.monotonic()
                        last = now
                except Exception as exc:  # the simulator was stopped or restarted
                    if self.verbose:
                        print(f"[soldier] routine paused: {type(exc).__name__}: {exc}")
                    if self._stop.wait(3.0):
                        break

    def start(self):
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()


if __name__ == "__main__":
    director = SoldierDirector()
    print("Soldier routine running. Ctrl+C to stop.")
    try:
        director.run()
    except KeyboardInterrupt:
        director.stop()
