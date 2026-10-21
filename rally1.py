r"""
rally.py - DIKU RoboRally 2026 (REX exam assignment 6).

Track (seen from above, x to the right, y up):

    L2 ----------- 400 ----------- L4
     |  \                       /  |
    300     \   obstacles   /     300
     |   /                     \   |
    L1 ----------------------------- L3

Route L1 -> L2 -> L3 -> L4 -> L1.

Idea - every round of a leg is: FACE the target, LOOK, then ONE action.
  - Face: by the camera if we see the target, else by the particle-filter pose (Ex5).
  - Look: 3 frames, update the particle filter with the 4 known corner landmarks.
  - Act:  something in the way -> drive to a point beside it
          target close          -> last stretch blind, stop ~30 cm before the box (< 40 cm rule)
          otherwise             -> one step towards the target
  We look where we are about to drive. Obstacles we saw earlier are remembered, because
  closer than ~90 cm the camera cannot see them any more. The front sonar is the last safety net.
  After a visit the sonar tells us how far we are from the box -> all particles go there.
  If no particle fits the measurements, the filter is lost -> new particles on a ring
  around the closest landmark ("sensor resetting").

Race day:  python3 rally.py | tee run.log     (landmark IDs 1-4)
Practice:  python3 rally.py practice          (IDs 5, 3, 6, 10)
Calibrate: python3 rally.py calib             (12 x 30 deg = one full turn, then 100 cm)
"""
import sys
import time
import numpy as np
import particle, camera, robot

# ---- track ------------------------------------------------------------------
BOX_HALF = 20.0                                 # half landmark box side [cm]
W, H = 400 + 2 * BOX_HALF, 300 + 2 * BOX_HALF   # 400/300 are between the INNER sides -> centre to centre
L1, L2, L3, L4 = (5, 3, 6, 10) if "practice" in sys.argv else (1, 2, 3, 4)
LANDMARKS = {L1: (0.0, 0.0), L2: (0.0, H), L3: (W, 0.0), L4: (W, H)}
ROUTE = [L1, L2, L3, L4, L1]
NAME = {L1: "L1", L2: "L2", L3: "L3", L4: "L4"}

# ---- robot (check on race day) ------------------------------------------------
CAM_FRONT = 21.0                    # wheel axle -> camera [cm] (CAM_OFFSET, Ex4)
SONAR_FRONT = 28.0                  # wheel axle -> front sonar [cm]  MEASURE!
CAM_F, MARKER_SIZE = 1315.0, 0.145  # our Ex4 calibration (camera.py's f=1687 makes distances 1.33x too long)
SPEED, CM_PER_SEC, DEG_PER_SEC = 64, 46.0, 125.0

# ---- strategy ---------------------------------------------------------------------
VISIT_DIST = 30.0                   # stop distance front -> box side (rule: < 40 cm)
BLIND_DIST = 130.0                  # closer than this the camera loses the marker -> drive blind
BLIND_STOP = 45.0                   # blind stretch ends at this camera -> box distance
SEE_DIST = 160.0                    # from here on the camera should see the target
KEEP_OBS = 28.0 + 17.5 + 15        # path -> obstacle centre: robot radius + box radius + margin [cm]
KEEP_LM = 100.0                     # path -> centre of a WRONG landmark (never get within 40 cm of it)
STEP = 70.0                         # max drive between two looks [cm]
SONAR_STOP = 450                    # [mm] > 40 cm, so we never "visit" a wrong box by accident
N = 1000                            # particles
SIGMA_A = np.radians(10.0)          # bearing noise
FULL_SCAN = [30] * 11               # with the current view: all around
AHEAD_SCAN = [30, 30, -90, -30]     # +30, +60, -30, -60 deg

arlo = robot.Robot()
cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
cam.intrinsic_matrix = np.array([[CAM_F, 0, 820.0], [0, CAM_F, 616.0], [0, 0, 1]], dtype=np.float64)
cam.arucoMarkerLength = MARKER_SIZE
particles = []
obstacles = {}                      # id -> (x, y) world: obstacles seen earlier (camera is blind < ~90 cm)
moved = True                        # moved since the last filter update? (else nothing new to learn)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---- motion -----------------------------------------------------------------
def turn(angle):
    """Turn in place [rad], positive = left. Moves the particles too."""
    global moved
    deg = np.degrees(angle)
    if abs(deg) < 2:                # too small to turn -> particles must not turn either
        return
    moved = True
    arlo.go_diff(SPEED, SPEED, int(deg < 0), int(deg > 0))
    time.sleep(abs(deg) / DEG_PER_SEC)
    arlo.stop(); time.sleep(0.3)
    for p in particles:
        particle.move_particle(p, 0.0, angle)
    particle.add_uncertainty(particles, 2.0, np.radians(3) + 0.1 * abs(angle))  # big turns are worse


def drive(dist, sonar_limit=SONAR_STOP):
    """Drive `dist` cm (negative = backwards). Stops if the sonar in driving direction
    sees something closer than `sonar_limit` mm. Returns (driven, aborted)."""
    global moved
    if abs(dist) < 1:
        return 0.0, False
    moved = True
    fwd = dist > 0
    ping = arlo.read_front_ping_sensor if fwd else arlo.read_back_ping_sensor
    arlo.go_diff(SPEED, SPEED + 2, int(fwd), int(fwd))
    t0, aborted = time.time(), False
    while time.time() - t0 < abs(dist) / CM_PER_SEC:
        if 0 < ping() < sonar_limit:
            aborted = True; break
        time.sleep(0.05)
    driven = np.sign(dist) * min(abs(dist), (time.time() - t0) * CM_PER_SEC)  # before the settle pause
    arlo.stop(); time.sleep(0.3)
    for p in particles:
        particle.move_particle(p, driven, 0.0)
    particle.add_uncertainty(particles, 2.0 + 0.05 * abs(driven), np.radians(3))
    return driven, aborted


def forward(dist):
    """Drive forward; if the sonar stops us: turn 60 deg to the freer side and move on a bit."""
    if drive(dist)[1]:
        l, r = [s if s > 0 else 9999 for s in (arlo.read_left_ping_sensor(), arlo.read_right_ping_sensor())]
        turn(np.radians(60 if l >= r else -60))
        drive(35.0)


# ---- particle filter (Exercise 5, 4 landmarks) ------------------------------
def as_arrays():
    return (np.array([p.getX() for p in particles]), np.array([p.getY() for p in particles]),
            np.array([p.getTheta() for p in particles]))


def set_particles(xs, ys, ths):
    global particles
    particles = [particle.Particle(x, y, t, 1.0 / len(xs)) for x, y, t in zip(xs, ys, ths)]


def ring(lm, d, a, n=N):
    """Particles that would see landmark `lm` at distance d (camera -> box side) and bearing a."""
    lx, ly = LANDMARKS[lm]
    phi = np.random.uniform(-np.pi, np.pi, n)               # direction landmark -> camera
    r = d + BOX_HALF + np.random.normal(0, 10, n)
    ths = wrap(phi + np.pi - a + np.random.normal(0, np.radians(5), n))
    cx, cy = lx + r * np.cos(phi), ly + r * np.sin(phi)     # camera on a ring around the box
    return cx - CAM_FRONT * np.cos(ths), cy - CAM_FRONT * np.sin(ths), ths   # axle behind the camera


def log_likelihood(xs, ys, ths, meas):
    """log p(measurements | particle). Predicted from the CAMERA, because the camera measures."""
    cx, cy = xs + CAM_FRONT * np.cos(ths), ys + CAM_FRONT * np.sin(ths)
    logw = np.zeros(len(xs))
    for i, (d, a) in meas.items():
        lx, ly = LANDMARKS[i]
        ed = np.hypot(lx - cx, ly - cy) - BOX_HALF          # expected distance to the box side
        ea = wrap(np.arctan2(ly - cy, lx - cx) - ths)       # expected bearing
        sd = 10.0 + 0.08 * d                                # far markers are noisier
        logw -= 0.5 * ((d - ed) / sd) ** 2 + 0.5 * (wrap(a - ea) / SIGMA_A) ** 2
    return logw


def filter_update(meas):
    """Weight + systematic resampling. If even the best particle is ~4 sigma off -> re-seed."""
    global moved
    if not meas or not particles or not moved:      # same view as last time -> nothing new
        return
    xs, ys, ths = as_arrays()
    logw = log_likelihood(xs, ys, ths, meas)
    if logw.max() < -8.0 * len(meas):
        lm = min(meas, key=lambda i: meas[i][0])    # closest landmark = most accurate
        print("LOST -> new particles around", NAME[lm])
        xs, ys, ths = ring(lm, *meas[lm])
        logw = log_likelihood(xs, ys, ths, meas)    # other landmarks pick the right part of the ring
    w = np.exp(logw - logw.max()); w /= w.sum()
    cum = np.cumsum(w); cum[-1] = 1.0
    idx = np.searchsorted(cum, (np.random.uniform() + np.arange(len(w))) / len(w))
    set_particles(xs[idx], ys[idx], ths[idx])
    moved = False


def pose():
    e = particle.estimate_pose(particles)
    return e.getX(), e.getY(), e.getTheta()


def localised():
    """True if the particles agree on one pose (a ring around one landmark does not:
    its headings point everywhere)."""
    if not particles:
        return False
    xs, ys, ths = as_arrays()
    return np.std(xs) < 50 and np.std(ys) < 50 and np.hypot(np.mean(np.cos(ths)), np.mean(np.sin(ths))) > 0.9


def snap_to(lm):
    """We stand in front of `lm`, facing it: put every particle where the sonar says."""
    global moved
    s = arlo.read_front_ping_sensor()
    gap = s / 10.0 if 0 < s < 1500 else VISIT_DIST
    if not particles:                               # first visit: ring, a later scan decides the side
        set_particles(*ring(lm, gap + SONAR_FRONT - CAM_FRONT, 0.0))
    else:
        ths = wrap(as_arrays()[2] + np.random.normal(0, np.radians(3), len(particles)))
        r = gap + BOX_HALF + SONAR_FRONT
        lx, ly = LANDMARKS[lm]
        set_particles(lx - r * np.cos(ths), ly - r * np.sin(ths), ths)
    moved = True                                    # new particles -> the next sighting must be used
    return gap


# ---- sensing ------------------------------------------------------------------
def sense():
    """3 frames, averaged per ID; updates the filter. Returns {id: (dist, angle)} of all markers."""
    raw = {}
    for _ in range(3):
        ids, ds, angs = cam.detect_aruco_objects(cam.get_next_frame())
        if ids is not None:
            for i, d, a in zip(ids, ds, angs):
                if 0 < d < 500:
                    raw.setdefault(int(i), []).append((float(d), float(a)))
    seen = {i: tuple(np.mean(v, axis=0)) for i, v in raw.items()}   # angles are within +-35 deg
    print("seen:", {i: (round(d), round(np.degrees(a))) for i, (d, a) in seen.items()})
    filter_update({i: m for i, m in seen.items() if i in LANDMARKS})
    if localised():                                     # remember obstacles in world coordinates
        x, y, th = pose()
        cx, cy = x + CAM_FRONT * np.cos(th), y + CAM_FRONT * np.sin(th)
        for i, (d, a) in seen.items():
            if i not in LANDMARKS and d < 350:          # far sightings are too inaccurate
                obstacles[i] = (cx + (d + 12) * np.cos(th + a), cy + (d + 12) * np.sin(th + a))
    return seen


def scan(target, steps=FULL_SCAN):
    """Turn by `steps` [deg] and look after each turn. Stops early if `target` shows up."""
    seen = {}
    for s in steps:
        turn(np.radians(s))
        seen = sense()
        if target in seen:
            break
    return seen


def where(target, seen):
    """(angle to turn, camera -> box side distance after the turn). Camera if possible, else pose."""
    if target in seen:
        d, a = seen[target]
        x, y = CAM_FRONT + (d + BOX_HALF) * np.cos(a), (d + BOX_HALF) * np.sin(a)  # box seen from the axle
        return np.arctan2(y, x), np.hypot(x, y) - CAM_FRONT - BOX_HALF
    px, py, th = pose()
    gx, gy = LANDMARKS[target]
    return wrap(np.arctan2(gy - py, gx - px) - th), np.hypot(gx - px, gy - py) - CAM_FRONT - BOX_HALF


def things_ahead(seen, target, dist):
    """Seen markers, remembered obstacles and the other landmarks (map) up to `dist` cm ahead,
    as (ahead, side, keep) [cm] seen from the camera. ahead = d*cos(a), side = d*sin(a), left = +."""
    things = [(d * np.cos(a), d * np.sin(a), KEEP_LM if i in LANDMARKS else KEEP_OBS)
              for i, (d, a) in seen.items() if i != target]
    if localised():                                     # known positions -> rotate into the robot frame
        x, y, th = pose()
        known = {i: (o, KEEP_OBS) for i, o in obstacles.items()}
        known.update({i: (o, KEEP_LM) for i, o in LANDMARKS.items() if i != target})
        for i, ((ox, oy), keep) in known.items():
            if i not in seen:
                dx, dy = ox - x - CAM_FRONT * np.cos(th), oy - y - CAM_FRONT * np.sin(th)
                things.append((np.cos(th) * dx + np.sin(th) * dy, -np.sin(th) * dx + np.cos(th) * dy, keep))
    return [t for t in things if 0 < t[0] < dist]


def blocked(things):
    return any(abs(side) < keep for _, side, keep in things)


# ---- race -------------------------------------------------------------------
def go_around(things, target):
    """Waypoint beside everything in the way: push it sideways past one box after the other
    (so we never squeeze between two). Try the shorter side first and look before driving."""
    first = min(a for a, side, keep in things if abs(side) < keep)
    near = [t for t in things if t[0] < first + 100]    # boxes further on are the next round's job
    x = first + CAM_FRONT                               # seen from the axle (we turn around it)
    sides = []
    for sign in (1, -1):                                # 1 = pass on the left, -1 = on the right
        y = 0.0
        for _, side, keep in sorted(near, key=lambda t: sign * t[1]):
            if abs(side - y) < keep + 15:
                y = side + sign * (keep + 15)
        sides.append(y)
    print("something in the way -> going around it (sideways %.0f or %.0f cm)" % tuple(sides))
    heading = 0.0
    for y in sorted(sides, key=abs):
        turn(np.arctan2(y, x) - heading)
        heading = np.arctan2(y, x)
        dist = min(np.hypot(x, y), 2 * STEP)
        if not blocked(things_ahead(sense(), target, dist)):
            forward(dist)
            return
    drive(-40.0)                                        # both ways blocked -> back off


def final_approach(target, dist):
    """Last stretch without camera, to BLIND_STOP. Then creep closer only if the sonar sees the
    box (head-on). On a diagonal the box corner sticks out ~15 cm -> still < 40 cm, no bump."""
    rest = max(dist - BLIND_STOP, 0.0)
    driven, aborted = drive(rest, sonar_limit=330)
    if aborted and rest - driven > 15:                       # sonar glitch far from the box -> once more
        drive(rest - driven, sonar_limit=330)
    gap = snap_to(target)
    if 35 < gap < 80:
        drive(gap - VISIT_DIST, sonar_limit=330)
        gap = snap_to(target)
    print("%s visited (sonar %.0f cm)" % (NAME[target], gap))


def go_to(target):
    """One leg. Every round: face the target, look, one action."""
    for _ in range(80):
        seen = sense()
        if target not in seen and not localised():
            seen = scan(target)                              # look around until we know where we are
            if target not in seen and not localised():
                print("still lost -> moving somewhere else")
                turn(np.radians(120)); forward(60.0)
                continue
        ang, dist = where(target, seen)
        if abs(ang) > np.radians(10):                        # not facing it -> turn, then look again
            turn(ang); continue
        turn(ang)                                            # small correction, view stays the same
        things = things_ahead(seen, target, dist)
        if blocked(things):
            go_around(things, target)
        elif target in seen and dist <= BLIND_DIST:
            final_approach(target, dist)
            return True
        elif target in seen:
            forward(min(dist - BLIND_DIST + 10, STEP))       # stop just inside BLIND_DIST
        elif dist < BLIND_DIST:                              # too close for the camera to see it:
            if drive(dist - BLIND_DIST - 20, sonar_limit=200)[1]:    # back off until it fits the image
                final_approach(target, dist)                # (something behind us -> go on the pose)
                return True
        elif dist < SEE_DIST:                                # should be visible from here, but is not
            if target not in scan(target, AHEAD_SCAN):
                print(NAME[target], "not visible -> estimate is off, relocalise")
                particle.add_uncertainty(particles, 80.0, np.radians(30))
        else:
            forward(min(dist - SEE_DIST + 20, STEP))         # drive until the camera can see it
    return False


def main():
    if L1 not in sense():                                    # placed too close? marker does not fit
        print("L1 not in the first image -> backing up a bit")
        drive(-50.0, sonar_limit=200)
    for target in ROUTE:
        if not go_to(target):                                # clock keeps running -> just go on
            print("could not reach %s - continuing" % NAME[target])
    print("done")


def calibrate():
    print("12 x 30 deg left -> robot must end up facing exactly the same way")
    for _ in range(12):
        turn(np.radians(30))
    input("Enter -> drive 100 cm (no sonar stop), then measure")
    drive(100.0, sonar_limit=0)


if __name__ == "__main__":
    try:
        calibrate() if "calib" in sys.argv[1:] else main()
    finally:
        arlo.stop()
        cam.terminateCaptureThread()