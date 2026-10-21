r"""
rally.py - DIKU RoboRally 2026 (REX exam assignment 6), short version.

Track (Ex6 figure 1, seen from above, x to the right, y up):

    L2 ----------- 400 ----------- L4
     |  \                       /  |
    300     \   obstacles   /     300
     |   /                     \   |
    L1 ----------------------------- L3

  Route L1 -> L2 (up) -> L3 (diagonal) -> L4 (up) -> L1 (diagonal).

Strategy (KISS):
  - Particle filter from Exercise 5 with the 4 known corner landmarks.
    Measurements are predicted from the CAMERA position (21 cm in front of the axle).
  - "Visited" = within 40 cm of the box side, so we stop at ~30 cm (front sonar)
    and never touch the box (3 min penalty).
  - When the target is in view, drive on the camera measurement directly - but only
    if no obstacle is seen on the line to it (else the sonar would stop at the
    obstacle and we would count it as the target).
    Otherwise drive on the pose estimate, around the other landmarks
    (wrong order = 14 min penalty) and around remembered obstacles (any ID that
    is not a landmark; their world position is stored when we see them).
  - After every visit we know how far we are from that box -> move all
    particles there ("snap"). If measurements fit no particle twice in a row, the
    filter is lost -> re-seed the particles from that measurement ("sensor resetting").

Race-day checklist:
  1. Landmark IDs: race day uses 1, 2, 3, 4 (obstacles have ID > 4)!
  2. python3 rally.py calib -> 12 x 30 deg must be exactly ONE full turn and the
     drive exactly 100 cm. If not, scale DEG_PER_SEC / CM_PER_SEC.
     (There is no learned correction any more - this calibration is all we have.)
  3. Measure ROBOT_FRONT.
  4. Run with a log: python3 rally.py | tee run.log
"""
import sys
import time
import numpy as np
import particle, camera, robot

# ---- constants (check on race day) -----------------------------------------
BOX_HALF = 20.0                         # half landmark box side [cm]
# 300 x 400 cm are measured between the INNER box sides -> add one full box width
# to get centre-to-centre distances (the filter works with box centres).
INNER_H, INNER_W = 300.0, 400.0         # L1-L2 (vertical) and L2-L4 (horizontal) [cm]
L1, L2, L3, L4 = 5, 3, 6, 10            # PRACTICE track IDs. Race day: 1, 2, 3, 4 !
ROBOT_FRONT = 21.0                      # wheel-axle centre -> camera (CAM_OFFSET from our Ex4 rrtPlan.py)
ROBOT_RADIUS = 28.0                     # robot footprint radius [cm] (Ex4)
# Camera calibration from our Ex4 (rrtPlan.py). camera.py uses f=1687 and 15 cm markers,
# which makes every distance 1.33x too long (matches the measured "drive ratio" 1.29-1.40).
CAM_F, MARKER_SIZE = 1315.0, 0.145

_W, _H = INNER_W + 2 * BOX_HALF, INNER_H + 2 * BOX_HALF
LANDMARKS = {L1: (0.0, 0.0), L2: (0.0, _H), L3: (_W, 0.0), L4: (_W, _H)}
ROUTE = [L1, L2, L3, L4, L1]            # required order
NAME = {L1: "L1", L2: "L2", L3: "L3", L4: "L4"}

VISIT_DIST = 30.0                       # stop distance front -> box side (< 40 cm rule)
BLIND_DIST = 130.0                      # below this the last stretch is driven without camera
SEE_DIST = 160.0                        # from this gap on the camera should see the target
KEEP_AWAY = 100.0                       # min distance of our path to a non-target landmark centre
OBST_HALF = 12.0                        # marker face -> obstacle box centre [cm] (Box_Shift, Ex4)
OBST_RADIUS = 17.5                      # obstacle box radius [cm] (Box_Radius, Ex4)
OBST_KEEP = OBST_RADIUS + ROBOT_RADIUS + 30.0   # path -> obstacle centre; 30 cm for position errors
STEP = 50.0                             # driving step when target not in view [cm]
BACK_DIST = 50.0                        # back up at the start if L1 is not in the first image [cm]
SONAR_STOP = 450                        # [mm] > 40 cm, so we never "visit" a wrong box by accident
SPEED, CM_PER_SEC, DEG_PER_SEC = 64, 46.0, 125.0
N = 1000                                # particles
SIGMA_A = np.radians(10.0)              # bearing noise (distance noise grows with distance)

FULL_SCAN = [0] + [30] * 11             # turn steps [deg]: look all around
AHEAD_SCAN = [0, 30, 30, -90, -30]      # look at 0, +30, +60, -30, -60 deg only (fast)

arlo = robot.Robot()
cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
cam.intrinsic_matrix = np.array([[CAM_F, 0, 820.0], [0, CAM_F, 616.0], [0, 0, 1]], dtype=np.float64)
cam.arucoMarkerLength = MARKER_SIZE
particles = []
obstacles = {}                          # obstacle id -> (x, y) world position, remembered
moved = True                            # moved since the last filter update? (else nothing new to learn)
lost_count = 0                          # filter updates in a row where no particle fit


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def cam_pos(x, y, th):
    """Camera position: ROBOT_FRONT in front of the wheel axle. Works on numbers and arrays."""
    return x + ROBOT_FRONT * np.cos(th), y + ROBOT_FRONT * np.sin(th)


# ---- motion -----------------------------------------------------------------
def turn(angle):
    """Turn in place [rad], positive = left. Also moves the particles."""
    global moved
    deg = np.degrees(angle)
    if abs(deg) < 2:                    # too small to turn -> particles must not turn either
        return
    moved = True
    arlo.go_diff(SPEED, SPEED, 0, 1) if deg > 0 else arlo.go_diff(SPEED, SPEED, 1, 0)
    time.sleep(abs(deg) / DEG_PER_SEC)
    arlo.stop(); time.sleep(0.3)
    for p in particles:
        particle.move_particle(p, 0.0, angle)
    # timed turns get worse the bigger they are
    particle.add_uncertainty(particles, 2.0, np.radians(3) + 0.1 * abs(angle))


def drive(dist, sonar_limit=SONAR_STOP):
    """Drive forward [cm], abort if the front sonar sees something close. Returns (driven, aborted)."""
    global moved
    if dist <= 0:
        return 0.0, False
    moved = True
    t0, aborted = time.time(), False
    arlo.go_diff(SPEED, SPEED + 2, 1, 1)
    while time.time() - t0 < dist / CM_PER_SEC:
        s = arlo.read_front_ping_sensor()
        if 0 < s < sonar_limit:
            aborted = True; break
        time.sleep(0.05)
    elapsed = time.time() - t0          # measure BEFORE the settle pause
    arlo.stop(); time.sleep(0.3)
    driven = min(dist, elapsed * CM_PER_SEC)
    for p in particles:
        particle.move_particle(p, driven, 0.0)
    particle.add_uncertainty(particles, 2.0 + 0.05 * driven, np.radians(3))
    return driven, aborted


def avoid():
    """Obstacle ahead: turn 60 deg towards the freer side and move a bit."""
    l, r = arlo.read_left_ping_sensor(), arlo.read_right_ping_sensor()
    turn(np.radians(60) if (l if l > 0 else 9999) >= (r if r > 0 else 9999) else np.radians(-60))
    drive(35.0)


def back_up(dist):
    """Drive backwards up to `dist` cm - only as far as the back sonar says is free
    (keeping 20 cm), and stop if something shows up behind us.
    Only used at the start, before there are particles, so no particle update."""
    s = arlo.read_back_ping_sensor()
    if s > 0:                           # valid reading -> leave 20 cm to whatever is behind us
        dist = min(dist, s / 10.0 - 20.0)
    if dist < 10:
        print("no room behind us (back sonar %d mm) -> not backing up" % s)
        return
    print("backing up %.0f cm (back sonar %d mm)" % (dist, s))
    t0 = time.time()
    arlo.go_diff(SPEED, SPEED, 0, 0)
    while time.time() - t0 < dist / CM_PER_SEC:
        b = arlo.read_back_ping_sensor()
        if 0 < b < 200:                 # something closer than 20 cm behind us -> stop
            break
        time.sleep(0.05)
    arlo.stop(); time.sleep(0.3)


def step_to(point, dist):
    """Turn towards a world point (x, y) and drive `dist` cm; evade if the sonar stops us."""
    x, y, th = pose()
    turn(wrap(np.arctan2(point[1] - y, point[0] - x) - th))
    if drive(dist)[1]:
        avoid()


# ---- particle filter (Exercise 5, 4 landmarks) ------------------------------
def as_arrays():
    return (np.array([p.getX() for p in particles]), np.array([p.getY() for p in particles]),
            np.array([p.getTheta() for p in particles]))


def set_particles(xs, ys, ths):
    global particles
    particles = [particle.Particle(x, y, t, 1.0 / len(xs)) for x, y, t in zip(xs, ys, ths)]


def ring(lm_id, d, a, n=N):
    """Particles that would see lm_id at distance d (camera -> box side) and bearing a."""
    lx, ly = LANDMARKS[lm_id]
    phi = np.random.uniform(-np.pi, np.pi, n)            # direction landmark -> camera
    r = d + BOX_HALF + np.random.normal(0, 10, n)
    cx, cy = lx + r * np.cos(phi), ly + r * np.sin(phi)  # camera on a ring around the box
    ths = wrap(phi + np.pi - a + np.random.normal(0, np.radians(5), n))
    return cx - ROBOT_FRONT * np.cos(ths), cy - ROBOT_FRONT * np.sin(ths), ths   # axle behind camera


def filter_update(meas, check_lost=True):
    """Gaussian likelihood + systematic resampling. Re-seeds if no particle fits twice in a row."""
    global moved, lost_count
    if not meas or not particles or not moved:
        # No particles yet (start phase), or same spot and same view as the last update:
        # resampling on the same information again would only make the filter overconfident.
        return
    xs, ys, ths = as_arrays()
    cx, cy = cam_pos(xs, ys, ths)       # the camera measures, so predict from the camera
    logw = np.zeros(len(xs))
    for lm_id, (d, a) in meas.items():
        lx, ly = LANDMARKS[lm_id]
        ed = np.hypot(lx - cx, ly - cy) - BOX_HALF
        ea = wrap(np.arctan2(ly - cy, lx - cx) - ths)
        sd = 10.0 + 0.08 * d            # far markers are noisy (409 vs 362 cm for the same box)
        logw += -0.5 * ((d - ed) / sd) ** 2 - 0.5 * (wrap(a - ea) / SIGMA_A) ** 2
    if check_lost and logw.max() < -8.0 * len(meas):
        # Even the best particle is ~4 sigma off. Once could be a bad frame -> ignore it.
        # Twice in a row: the estimate is wrong, and resampling cannot fix that (it only
        # picks among wrong particles) -> start over from this sighting.
        lost_count += 1
        if lost_count < 2:
            print("measurement fits no particle -> ignored once")
            return
        lost_count = 0
        lm_id = min(meas, key=lambda i: meas[i][0])     # closest = most accurate
        print("LOST -> re-seeding particles around %s" % NAME[lm_id])
        set_particles(*ring(lm_id, *meas[lm_id]))
        filter_update(meas, check_lost=False)           # use the other landmarks on the new ring
        return
    lost_count = 0
    w = np.exp(logw - logw.max()); w /= w.sum()
    cum = np.cumsum(w); cum[-1] = 1.0
    idx = np.searchsorted(cum, (np.random.uniform() + np.arange(len(w))) / len(w))
    set_particles(xs[idx], ys[idx], ths[idx])
    moved = False


def pose():
    e = particle.estimate_pose(particles)
    return e.getX(), e.getY(), e.getTheta()


def localised():
    """True if the particles agree on one pose. A ring around one landmark does not: its
    positions can be close together (small ring) but its headings point everywhere."""
    if not particles:
        return False
    xs, ys, ths = as_arrays()
    heading_agree = np.hypot(np.mean(np.cos(ths)), np.mean(np.sin(ths)))   # 1 = all equal
    return np.std(xs) < 50 and np.std(ys) < 50 and heading_agree > 0.9


def snap_to(lm_id):
    """We stand in front of lm_id facing it: put every particle at the matching spot."""
    global moved
    s = arlo.read_front_ping_sensor()
    gap = s / 10.0 if 0 < s < 1500 else VISIT_DIST
    if not particles:                   # very first visit: ring, scan decides the side
        set_particles(*ring(lm_id, gap, 0.0))
    else:
        xs, ys, ths = as_arrays()
        ths = wrap(ths + np.random.normal(0, np.radians(3), len(ths)))
        r = gap + BOX_HALF + ROBOT_FRONT
        lx, ly = LANDMARKS[lm_id]
        set_particles(lx - r * np.cos(ths), ly - r * np.sin(ths), ths)
    moved = True                        # new particles -> the next sighting must be used
    return gap


def remember(obs):
    """Store the world position of every obstacle we see (only when we know where we are)."""
    if not obs or not localised():
        return
    x, y, th = pose()
    cx, cy = cam_pos(x, y, th)
    for i, (d, a) in obs.items():
        if d > 350:
            continue                    # far sightings are too inaccurate to plan with
        r = d + OBST_HALF
        ox, oy = cx + r * np.cos(th + a), cy + r * np.sin(th + a)
        if not (-60 < ox < _W + 60 and -60 < oy < _H + 60):
            continue                    # outside the track square -> not a race obstacle
        if i in obstacles:              # average with earlier sightings
            ox, oy = 0.5 * (ox + obstacles[i][0]), 0.5 * (oy + obstacles[i][1])
        else:
            print("obstacle %d at (%.0f, %.0f)" % (i, ox, oy))
        obstacles[i] = (ox, oy)


def sense():
    """Grab 3 frames, average per ID; update the filter. Returns (landmarks, obstacles) as {id: (dist, angle)}."""
    raw = {}
    for _ in range(3):
        ids, ds, angs = cam.detect_aruco_objects(cam.get_next_frame())
        if ids is not None:
            for i, d, a in zip(ids, ds, angs):
                if 0 < d < 500:
                    raw.setdefault(int(i), []).append((float(d), float(a)))
    seen = {i: (np.mean([d for d, _ in v]), np.arctan2(np.mean([np.sin(a) for _, a in v]),
                                                         np.mean([np.cos(a) for _, a in v])))
            for i, v in raw.items()}
    print("seen:", {i: (round(d), round(np.degrees(a))) for i, (d, a) in seen.items()})
    lms = {i: m for i, m in seen.items() if i in LANDMARKS}
    obs = {i: m for i, m in seen.items() if i not in LANDMARKS}       # everything else = obstacle
    filter_update(lms)
    remember(obs)
    return lms, obs


def blocked(obs, d_t, a_t):
    """True if a seen obstacle lies on the line to a point at distance d_t, bearing a_t.
    Rotate the view so that point is straight ahead: d*cos = how far ahead, d*sin = how far to the side."""
    return any(d * np.cos(a - a_t) < d_t and abs(d * np.sin(a - a_t)) < ROBOT_RADIUS + OBST_RADIUS
               for d, a in obs.values())


# ---- race -------------------------------------------------------------------
def scan(target, steps=FULL_SCAN):
    """Turn by `steps` [deg] and look after each turn (updates the filter).
    Stops early, facing it, if `target` shows up."""
    for s in steps:
        turn(np.radians(s))
        lms, _ = sense()
        if target in lms:
            return lms
    return {}


def relocalise(target):
    """Scan until the particles agree on one pose. Returns the landmarks if the target showed up."""
    for _ in range(3):
        lms = scan(target)
        if target in lms or localised():
            return lms
        print("not localised yet -> move a bit and scan again")
        turn(np.radians(120)); drive(60.0)
    return {}


def detour(x, y, gx, gy, target):
    """First landmark/obstacle the straight line to the goal passes too closely -> waypoint beside it."""
    things = [(lx, ly, KEEP_AWAY) for i, (lx, ly) in LANDMARKS.items() if i != target]
    things += [(ox, oy, OBST_KEEP) for ox, oy in obstacles.values()]
    dx, dy = gx - x, gy - y
    length = max(np.hypot(dx, dy), 1e-9)
    best = None
    for lx, ly, keep in things:
        t = ((lx - x) * dx + (ly - y) * dy) / length ** 2
        # behind us, beyond the goal, or we are already at the closest point (just left it) -> no problem
        if t >= 1 or t * length < 40:
            continue
        cx, cy = x + t * dx, y + t * dy
        off = np.hypot(cx - lx, cy - ly)
        if off < keep and (best is None or t < best[0]):
            nx, ny = (cx - lx, cy - ly) if off > 1 else (-dy, dx)   # push out on the side we already pass
            n = np.hypot(nx, ny)
            best = (t, (lx + (keep + 20) * nx / n, ly + (keep + 20) * ny / n))
    return None if best is None else best[1]


def approach(target, lms):
    """Target is in view (and nothing in the way): turn to it and drive until VISIT_DIST.
    Returns True once visited."""
    d, a = lms[target]
    # d, a are seen from the camera, but we turn around the axle 21 cm behind it.
    ax, ay = ROBOT_FRONT + d * np.cos(a), d * np.sin(a)    # marker as seen from the axle
    turn(np.arctan2(ay, ax))
    d = np.hypot(ax, ay) - ROBOT_FRONT                     # camera -> marker after the turn
    if d <= BLIND_DIST:
        # Final stretch: closer than ~90 cm the camera loses the marker, so drive
        # without re-checking it. The front sonar should do the stopping (330 mm),
        # but it misses boxes approached at an angle (sound reflects away), so the
        # time-based distance must be right as well (-> calib).
        remaining = max(d - VISIT_DIST, 0.0)
        driven, aborted = drive(remaining, sonar_limit=330)
        if aborted and remaining - driven > 15.0:      # sonar glitch far from the box -> try the rest once
            drive(remaining - driven, sonar_limit=330)
        gap = snap_to(target)
        if gap > 38:                                   # stopped short -> creep closer
            drive(gap - VISIT_DIST, sonar_limit=330)
            gap = snap_to(target)
        print("%s visited (camera said %.0f cm, sonar now %.0f cm)" % (NAME[target], d, gap))
        return True
    # Further away: stop ~10 cm inside BLIND_DIST so the next sighting is guaranteed.
    # The stop point is >= 120 cm from the target, so the normal sonar limit only reacts to obstacles.
    if drive(min(d - BLIND_DIST + 10.0, 80.0))[1]:
        avoid()
    return False


def go_to(target):
    """One leg: reach `target` using the pose estimate until the camera sees it."""
    gx, gy = LANDMARKS[target]
    for _ in range(60):
        lms, obs = sense()
        if blocked(obs, 80.0, 0.0):             # obstacle box right in front of us -> evade first
            avoid(); continue
        if target not in lms and not localised():
            relocalise(target)
            continue                            # look again from where the scan stopped
        if target in lms:
            d_t, a_t = lms[target]
            in_way = blocked(obs, d_t, a_t)     # obstacle between us and the target?
            # Go around a seen or remembered obstacle via a waypoint, if we know where we are.
            if localised() and (in_way or d_t > BLIND_DIST):
                x, y, _ = pose()
                wp = detour(x, y, gx, gy, target)
                if wp is not None:
                    print("target in view, but blocked -> detour via (%.0f, %.0f)" % wp)
                    step_to(wp, min(STEP, np.hypot(wp[0] - x, wp[1] - y)))
                    continue
            if in_way:                          # never approach through it: the sonar would
                print("obstacle in front of %s -> evade" % NAME[target])   # take it for the target
                avoid(); continue
            if approach(target, lms):
                return True
            continue
        x, y, th = pose()
        gap = np.hypot(gx - x, gy - y) - BOX_HALF - ROBOT_FRONT     # our front -> target box side
        print("pose x=%.0f y=%.0f th=%.0f deg | %s gap %.0f cm"
              % (x, y, np.degrees(th), NAME[target], gap))
        if gap < SEE_DIST:
            # The camera should see it from here: face it, then look left and right.
            turn(wrap(np.arctan2(gy - y, gx - x) - th))
            if target in scan(target, AHEAD_SCAN):
                continue
            # Not found but the other landmarks agree with the estimate (else LOST would have
            # fired): probably hidden behind an obstacle -> creep closer towards the estimate.
            print("%s not visible -> moving closer" % NAME[target])
            if gap > BLIND_DIST:
                step_to((gx, gy), min(STEP, gap - BLIND_DIST + 10))
            else:
                particle.add_uncertainty(particles, 80.0, np.radians(30))   # really lost -> relocalise
            continue
        wp = detour(x, y, gx, gy, target)
        if wp is not None:
            print("detour via (%.0f, %.0f)" % wp)
            step_to(wp, min(STEP, np.hypot(wp[0] - x, wp[1] - y)))
        else:
            step_to((gx, gy), min(STEP, gap - SEE_DIST + 20))       # stop where the camera can see it
    return False


def main():
    print("map:", {NAME[i]: p for i, p in LANDMARKS.items()})
    # Start: turn until L1 is in view, approach it on the camera (= first visit).
    # The visit seeds the particles on a ring around L1 at the sonar distance.
    # Placed too close to L1, the marker may not fit into the image (closer than ~90 cm
    # the camera loses it) -> back up a bit first; the search below then sees it.
    if L1 not in sense()[0]:
        print("L1 not in the first image - maybe too close -> backing up")
        back_up(BACK_DIST)
    misses = 0
    for _ in range(60):
        lms, obs = sense()
        if L1 not in lms:
            misses += 1
            if misses % 12 == 0:        # a full turn without L1 (hidden?) -> move, then look again
                print("L1 not seen in a full turn -> moving a bit")
                if drive(60.0)[1]:
                    avoid()
            else:
                turn(np.radians(30))
            continue
        if blocked(obs, *lms[L1]):
            avoid(); continue
        if approach(L1, lms):
            break
    # A ring around L1 has no useful mean -> scan until a second landmark fixes our side.
    relocalise(L2)
    for target in ROUTE[1:]:
        # If a leg fails the clock keeps running, so do not stop: a missed landmark
        # costs 14 min, but the following landmarks can still count.
        if not go_to(target):
            print("could not reach %s - continuing" % NAME[target])
    print("done - all landmarks visited")


def calibrate():
    """python3 rally.py calib"""
    print("12 x 30 deg left -> robot must end up facing exactly the same way")
    for _ in range(12):
        turn(np.radians(30))
    input("Enter -> drive 100 cm (no sonar stop), then measure")
    drive(100.0, sonar_limit=0)


try:
    calibrate() if "calib" in sys.argv[1:] else main()
finally:
    arlo.stop()
    cam.terminateCaptureThread()