r"""
rally.py - DIKU RoboRally 2026 (REX Ex6).

    L2 ---- 400 ---- L4      Route: L1 -> L2 -> L3 -> L4 -> L1
     |                |
    300  obstacles   300
     |                |
    L1 -------------- L3

Particle filter (Ex5) on the 4 corner landmarks. Approach a seen target on the camera,
last cm closed loop on the front sonar to VISIT_DIST. Otherwise drive on the pose
estimate around other landmarks and remembered obstacles.

Race day: landmark IDs 1-4, measure ROBOT_FRONT and SONAR_FRONT,
then `python3 rally.py | tee run.log`.
"""
import time
import numpy as np
import particle, camera, robot

# ---- constants ---------------------------------------------------------------
BOX_HALF = 20.0                         # half landmark box side [cm]
INNER_H, INNER_W = 300.0, 400.0         # distances between INNER box sides [cm]
L1, L2, L3, L4 = 8, 1, 9, 11            # PRACTICE IDs. Race day: 1, 2, 3, 4 !
ROBOT_FRONT = 21.0                      # axle -> camera [cm]
ROBOT_RADIUS = 28.0                     # robot footprint radius [cm]
SONAR_FRONT = 28.0                      # axle -> front sonar [cm] - MEASURE
CAM_F, MARKER_SIZE = 1315.0, 0.145      # Ex4 calibration (camera.py defaults are 1.33x too long)

_W, _H = INNER_W + 2 * BOX_HALF, INNER_H + 2 * BOX_HALF
LANDMARKS = {L1: (0.0, 0.0), L2: (0.0, _H), L3: (_W, 0.0), L4: (_W, _H)}
ROUTE = [L1, L2, L3, L4, L1]
NAME = {L1: "L1", L2: "L2", L3: "L3", L4: "L4"}

VISIT_DIST = 15.0                       # final gap front sonar -> box side [cm] (< 40 rule)
VISIT_MAX = 35.0                        # front sonar gap that already counts as visited [cm]
SONAR_MATCH = 30.0                      # camera blind near target: |sonar - expected gap| below this [cm]
BRAKE = 9.0                             # roll-out after sonar stop [cm] (test run: 18 -> ~9 cm)
BLIND_DIST = 130.0                      # closer than this: no more camera checks
BLIND_STOP = 45.0                       # blind stretch ends at this camera -> marker distance
BLIND_GAP = BLIND_STOP - (SONAR_FRONT - ROBOT_FRONT)   # expected sonar gap after it
SEE_DIST = 160.0                        # camera should see the target from here
KEEP_AWAY = 100.0                       # path -> non-target landmark centre [cm]
OBST_HALF = 12.0                        # marker face -> obstacle centre [cm]
OBST_RADIUS = 17.5                      # obstacle radius [cm]
OBST_KEEP = OBST_RADIUS + ROBOT_RADIUS + 30.0
CLEAR = OBST_RADIUS + ROBOT_RADIUS + 15.0     # min path -> obstacle centre while driving [cm]
STEP = 80.0                             # step when target not in view [cm]
BACK_DIST = 50.0                        # back up at start if L1 not found in a full turn [cm]
SONAR_STOP = 450                        # [mm] normal sonar abort
SIDE_STOP = 120                         # [mm] side sonar emergency stop while driving
SIDE_FREE = 600                         # [mm] both sides freer than this -> evade towards the goal
SPEED, CM_PER_SEC, DEG_PER_SEC = 64, 46.0, 125.0
FAST, FAST_CM_PER_SEC = 90, 64.0        # long free drives only - MEASURE (64.0 is a guess: 46 * 90/64)
T_START = 0.15                          # start-up time of a drive [s]
SETTLE = 0.3                            # robot standing still before a camera frame [s]
STOP_PAUSE = 0.1                        # pause after every move; the rest of SETTLE only before sense
FLUSH_FRAMES = 2                        # old frames (camera buffer) thrown away before measuring
FRAMES = 2                              # frames averaged per look
N = 500                                 # particles
SIGMA_A = np.radians(10.0)              # bearing noise
LOC_STD = 50.0                          # localised: particle std in x and y below this [cm] - see run.log

FULL_SCAN = [0] + [-45] * 7             # look all around [deg]; camera sees 2*arctan(820/1315) = 64 deg
AHEAD_SCAN = [0, -45, 90]               # 0, -45, +45
SONAR_SWEEP = [-15, -15, 45, 15]        # sonar at -15, -30, +15, +30

arlo = robot.Robot()
cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
cam.intrinsic_matrix = np.array([[CAM_F, 0, 820.0], [0, CAM_F, 616.0], [0, 0, 1]], dtype=np.float64)
cam.arucoMarkerLength = MARKER_SIZE
particles = []
obstacles = {}                          # id -> (x, y) world position
obst_w = {}                             # id -> weight of that position (sum of 1/distance)
moved = True                            # moved since last filter update?
last_view = None                        # result of the last sense(); None once we move
lost_count = 0                          # updates in a row where no particle fit
T0 = time.time()                        # run start (log time stamps)
stats = {"sense": 0.0, "turn": 0.0, "drive": 0.0, "pause": 0.0, "n_sense": 0}

_print = print
def print(*args, **kw):
    """Every log line gets the seconds since start: [  12.3] ..."""
    kw.setdefault("flush", True)        # show live through `| tee run.log`
    _print("[%6.1f]" % (time.time() - T0), *args, **kw)


def settle(t):
    time.sleep(t)
    stats["pause"] += t


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def cam_pos(x, y, th):
    """Camera position (ROBOT_FRONT ahead of the axle)."""
    return x + ROBOT_FRONT * np.cos(th), y + ROBOT_FRONT * np.sin(th)


def front_sonar():
    """Median of 3 front sonar readings [mm]."""
    return sorted(arlo.read_front_ping_sensor() for _ in range(3))[1]


# ---- motion ------------------------------------------------------------------
def turn(angle):
    """Turn in place [rad], positive = left. Moves the particles too."""
    global moved, last_view
    deg = np.degrees(angle)
    if abs(deg) < 2:
        return
    moved, last_view = True, None
    arlo.go_diff(SPEED, SPEED, 0, 1) if deg > 0 else arlo.go_diff(SPEED, SPEED, 1, 0)
    time.sleep(abs(deg) / DEG_PER_SEC)
    stats["turn"] += abs(deg) / DEG_PER_SEC
    arlo.stop(); settle(STOP_PAUSE)
    for p in particles:
        particle.move_particle(p, 0.0, angle)
    particle.add_uncertainty(particles, 2.0, np.radians(3) + 0.1 * abs(angle))


def drive(dist, sonar_limit=SONAR_STOP, sides=True, fast=False):
    """Drive forward [cm], abort on front sonar < sonar_limit or (sides) a side sonar < SIDE_STOP.
    fast: FAST speed (long free drives). Returns (driven, aborted)."""
    global moved, last_view
    if dist <= 0:
        return 0.0, False
    moved, last_view = True, None
    speed, cps = (FAST, FAST_CM_PER_SEC) if fast else (SPEED, CM_PER_SEC)
    t0, aborted, k = time.time(), False, 0
    arlo.go_diff(speed, speed + 2, 1, 1)
    while time.time() - t0 < dist / cps + T_START:
        s = arlo.read_front_ping_sensor()
        if 0 < s < sonar_limit:
            aborted = True; break
        if sides:                       # one side per loop, so the front check stays fast
            s = arlo.read_left_ping_sensor() if k % 2 == 0 else arlo.read_right_ping_sensor()
            k += 1
            if 0 < s < SIDE_STOP:
                print("side sonar %d mm -> emergency stop" % s)
                aborted = True; break
        time.sleep(0.05)
    elapsed = time.time() - t0
    stats["drive"] += elapsed
    arlo.stop(); settle(STOP_PAUSE)
    driven = min(dist, max(0.0, elapsed - T_START) * cps)
    for p in particles:
        particle.move_particle(p, driven, 0.0)
    particle.add_uncertainty(particles, 2.0 + 0.05 * driven, np.radians(3))
    return driven, aborted


def avoid(goal=None):
    """Turn 60 deg to the freer side (both free and localised: towards `goal`) and move 35 cm.
    Not free -> try the other side, neither free -> back up. So we always move."""
    l, r = [s if s > 0 else 9999 for s in (arlo.read_left_ping_sensor(), arlo.read_right_ping_sensor())]
    a = 60 if l >= r else -60
    if goal and min(l, r) > SIDE_FREE and localised():
        x, y, th = pose(); a = 60 if wrap(np.arctan2(goal[1] - y, goal[0] - x) - th) > 0 else -60
    for ang in (a, -2 * a):
        turn(np.radians(ang))
        _, obs = sense()
        if not blocked(obs, 35.0 + ROBOT_RADIUS, 0.0) and free_dist(35.0) >= 35.0:
            drive(35.0); return
    turn(np.radians(a))                 # back to the original heading
    back_up(25.0)


def back_up(dist):
    """Reverse up to `dist` cm, keeping 20 cm to whatever is behind. Moves the particles too."""
    global moved, last_view
    s = arlo.read_back_ping_sensor()
    if s > 0:
        dist = min(dist, s / 10.0 - 20.0)
    if dist < 10:
        print("no room behind us (back sonar %d mm) -> not backing up" % s)
        return
    print("backing up %.0f cm (back sonar %d mm)" % (dist, s))
    t0 = time.time()
    arlo.go_diff(SPEED, SPEED, 0, 0)
    while time.time() - t0 < dist / CM_PER_SEC + T_START:
        b = arlo.read_back_ping_sensor()
        if 0 < b < 200:
            break
        time.sleep(0.05)
    elapsed = time.time() - t0
    stats["drive"] += elapsed
    arlo.stop(); settle(STOP_PAUSE)
    driven = min(dist, max(0.0, elapsed - T_START) * CM_PER_SEC)
    moved, last_view = True, None
    for p in particles:
        particle.move_particle(p, -driven, 0.0)
    particle.add_uncertainty(particles, 2.0 + 0.05 * driven, np.radians(3))


def free_dist(dist):
    """How far we can drive straight ahead before getting closer than CLEAR to a remembered obstacle."""
    if not obstacles or not localised():
        return dist
    x, y, th = pose()
    ux, uy = np.cos(th), np.sin(th)
    for ox, oy in obstacles.values():
        ahead = (ox - x) * ux + (oy - y) * uy           # along our heading
        side = abs((oy - y) * ux - (ox - x) * uy)       # across it
        # Already inside CLEAR (e.g. after an evasion): we may pass, but get at most 10 cm
        # closer and never closer than 5 cm to touching - else we could never drive away.
        keep = min(CLEAR, max(np.hypot(ox - x, oy - y) - 10.0, OBST_RADIUS + ROBOT_RADIUS + 5.0))
        if ahead > 0 and side < keep:
            dist = min(dist, ahead - np.sqrt(keep ** 2 - side ** 2))
    return dist


def step_to(point, dist):
    """Turn towards world point (x, y), LOOK, then drive `dist` cm - only as far as it is free."""
    x, y, th = pose()
    turn(wrap(np.arctan2(point[1] - y, point[0] - x) - th))
    _, obs = sense()                    # camera check in the new direction before driving
    dist = min(dist, free_dist(dist))
    if dist < 10 or blocked(obs, dist + ROBOT_RADIUS, 0.0):
        print("path not free -> evade")
        avoid(point); return
    if drive(dist, fast=True)[1]:
        avoid(point)


def find_box_echo(max_cm):
    """Sonar sweep +-30 deg. Faces the closest echo < max_cm and returns its gap [cm], else None."""
    best, best_s, cur = None, max_cm * 10.0, 0
    s = front_sonar()
    if 0 < s < best_s:
        best, best_s = 0, s
    for st in SONAR_SWEEP:
        turn(np.radians(st)); cur += st
        s = front_sonar()
        if 0 < s < best_s:
            best, best_s = cur, s
    turn(np.radians((best if best is not None else 0) - cur))
    return None if best is None else best_s / 10.0


def close_in():
    """Creep on the front sonar until VISIT_DIST. Returns final gap [cm], None if no box echo."""
    limit = BLIND_GAP + 25.0            # echo further away = not our box
    s = front_sonar()
    gap = s / 10.0 if 0 < s < limit * 10 else None
    if gap is None:
        print("front sonar sees no box (diagonal?) -> sonar sweep")
        gap = find_box_echo(limit)
    if gap is None:
        return None                     # caller decides: camera saw it -> blind stop counts, else not
    for _ in range(2):                  # 2nd try only after a sonar glitch
        if gap <= VISIT_DIST + BRAKE:
            break
        # at most 3 cm past VISIT_DIST if the echo drops out
        drive(gap - VISIT_DIST + 3.0, sonar_limit=int((VISIT_DIST + BRAKE) * 10), sides=False)
        s = front_sonar()
        if not (0 < s < limit * 10):
            return VISIT_DIST
        gap = s / 10.0
    return gap


# ---- particle filter -----------------------------------------------------------
def as_arrays():
    return (np.array([p.getX() for p in particles]), np.array([p.getY() for p in particles]),
            np.array([p.getTheta() for p in particles]))


def set_particles(xs, ys, ths):
    global particles
    particles = [particle.Particle(x, y, t, 1.0 / len(xs)) for x, y, t in zip(xs, ys, ths)]


def ring(lm_id, d, a, n=N):
    """Particles that see lm_id at distance d (camera -> box side) and bearing a."""
    lx, ly = LANDMARKS[lm_id]
    phi = np.random.uniform(-np.pi, np.pi, n)
    r = d + BOX_HALF + np.random.normal(0, 10, n)
    cx, cy = lx + r * np.cos(phi), ly + r * np.sin(phi)
    ths = wrap(phi + np.pi - a + np.random.normal(0, np.radians(5), n))
    return cx - ROBOT_FRONT * np.cos(ths), cy - ROBOT_FRONT * np.sin(ths), ths


def filter_update(meas, check_lost=True):
    """Gaussian likelihood + systematic resampling; re-seed if no particle fits twice in a row."""
    global moved, lost_count
    if not meas or not particles or not moved:
        return                          # nothing new -> don't get overconfident
    xs, ys, ths = as_arrays()
    cx, cy = cam_pos(xs, ys, ths)
    logw = np.zeros(len(xs))
    for lm_id, (d, a) in meas.items():
        lx, ly = LANDMARKS[lm_id]
        ed = np.hypot(lx - cx, ly - cy) - BOX_HALF
        ea = wrap(np.arctan2(ly - cy, lx - cx) - ths)
        sd = 10.0 + 0.08 * d            # far markers are noisier
        logw += -0.5 * ((d - ed) / sd) ** 2 - 0.5 * (wrap(a - ea) / SIGMA_A) ** 2
    if check_lost and logw.max() < -8.0 * len(meas):
        lost_count += 1
        if lost_count < 2:
            print("measurement fits no particle -> ignored once")
            return
        lost_count = 0
        lm_id = min(meas, key=lambda i: meas[i][0])
        print("LOST -> re-seeding particles around %s" % NAME[lm_id])
        set_particles(*ring(lm_id, *meas[lm_id]))
        filter_update(meas, check_lost=False)
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


def spread():
    """std x, std y [cm] and heading agreement (0..1) of the particles."""
    xs, ys, th = as_arrays()
    return np.std(xs), np.std(ys), np.hypot(np.mean(np.cos(th)), np.mean(np.sin(th)))


def localised():
    """Particles agree on position AND heading (a ring around one landmark does not)."""
    if not particles:
        return False
    sx, sy, heading_agree = spread()
    return sx < LOC_STD and sy < LOC_STD and heading_agree > 0.9


def snap_to(lm_id, gap):
    """We face lm_id at `gap` cm (front sonar -> box side): move all particles there."""
    global moved, last_view
    if not particles:                   # first visit: ring, a scan decides the side
        set_particles(*ring(lm_id, gap + SONAR_FRONT - ROBOT_FRONT, 0.0))
    else:
        xs, ys, ths = as_arrays()
        ths = wrap(ths + np.random.normal(0, np.radians(3), len(ths)))
        r = gap + BOX_HALF + SONAR_FRONT
        lx, ly = LANDMARKS[lm_id]
        set_particles(lx - r * np.cos(ths), ly - r * np.sin(ths), ths)
    moved, last_view = True, None


def remember(obs):
    """Store obstacle world positions (only when localised)."""
    if not obs or not localised():
        return
    x, y, th = pose()
    cx, cy = cam_pos(x, y, th)
    for i, (d, a) in obs.items():
        if d > 250:                     # far sightings are too inaccurate
            continue
        r = d + OBST_HALF
        ox, oy = cx + r * np.cos(th + a), cy + r * np.sin(th + a)
        if not (-60 < ox < _W + 60 and -60 < oy < _H + 60):
            continue
        w = 1.0 / d                     # close sightings count more
        if i in obstacles:
            (px, py), pw = obstacles[i], obst_w[i]
            ox, oy, w = (pw * px + w * ox) / (pw + w), (pw * py + w * oy) / (pw + w), pw + w
        else:
            print("obstacle %d at (%.0f, %.0f)" % (i, ox, oy))
        obstacles[i], obst_w[i] = (ox, oy), w


def sense():
    """FRAMES frames, averaged per ID; updates the filter. Returns (landmarks, obstacles)."""
    global last_view
    if last_view is not None:           # not moved since the last look -> same picture, reuse it
        return last_view
    if moved:
        settle(SETTLE - STOP_PAUSE)     # camera needs a still robot, the rest of SETTLE
    t0 = time.time()
    raw = {}
    for k in range(FLUSH_FRAMES + FRAMES):   # first frames may still show the view before the move
        try:
            frame = cam.get_next_frame()
            if k < FLUSH_FRAMES or frame is None:
                continue
            ids, ds, angs = cam.detect_aruco_objects(frame)
        except Exception as e:          # one broken frame must not end the run
            print("camera error: %s -> frame skipped" % e); continue
        if ids is None:
            if k == FLUSH_FRAMES:       # first real frame shows nothing -> stop looking
                break
            continue
        for i, d, a in zip(ids, ds, angs):
            if 0 < d < 500:
                raw.setdefault(int(i), []).append((float(d), float(a)))
    seen = {i: (np.mean([d for d, _ in v]), np.arctan2(np.mean([np.sin(a) for _, a in v]),
                                                         np.mean([np.cos(a) for _, a in v])))
            for i, v in raw.items()}
    print("seen:", {i: (round(d), round(np.degrees(a))) for i, (d, a) in seen.items()})
    lms = {i: m for i, m in seen.items() if i in LANDMARKS}
    obs = {i: m for i, m in seen.items() if i not in LANDMARKS}
    filter_update(lms)
    remember(obs)
    stats["sense"] += time.time() - t0
    stats["n_sense"] += 1
    last_view = lms, obs
    return lms, obs


def blocked(obs, d_t, a_t):
    """True if a seen obstacle (centre) is closer than CLEAR to the line to the point (d_t, a_t)."""
    return any((d + OBST_HALF) * np.cos(a - a_t) < d_t and abs((d + OBST_HALF) * np.sin(a - a_t)) < CLEAR
               for d, a in obs.values())


# ---- race ----------------------------------------------------------------------
def scan(target, steps=FULL_SCAN, until_localised=False):
    """Turn and look after each step; stop early facing `target` if seen (or once localised)."""
    for s in steps:
        turn(np.radians(s))
        lms, _ = sense()
        if target in lms or (until_localised and localised()):
            return lms
    return {}


def relocalise(target):
    """Scan until localised. Returns the landmarks if the target showed up."""
    for _ in range(3):
        lms = scan(target, until_localised=True)
        if target in lms or localised():
            return lms
        print("not localised yet -> move a bit and scan again")
        turn(np.radians(120)); drive(60.0)
    return {}


def detour(x, y, gx, gy, target):
    """Waypoint beside the first landmark/obstacle the straight line passes too closely.
    Takes the side with the shorter total path; a side inside another box's zone only if both are."""
    things = [(lx, ly, KEEP_AWAY) for i, (lx, ly) in LANDMARKS.items() if i != target]
    things += [(ox, oy, OBST_KEEP) for ox, oy in obstacles.values()]
    dx, dy = gx - x, gy - y
    length = max(np.hypot(dx, dy), 1e-9)
    best = None
    for j, (lx, ly, keep) in enumerate(things):
        t = ((lx - x) * dx + (ly - y) * dy) / length ** 2
        if t >= 1 or t * length < 40:   # beyond the goal or already passed
            continue
        cx, cy = x + t * dx, y + t * dy
        off = np.hypot(cx - lx, cy - ly)
        if off < keep and (best is None or t < best[0]):
            best = (t, j, (cx - lx, cy - ly) if off > 1 else (-dy, dx))
    if best is None:
        return None
    _, j, (nx, ny) = best
    lx, ly, keep = things[j]
    n = np.hypot(nx, ny)
    options = []
    for side in (1, -1):
        wx, wy = lx + side * (keep + 20) * nx / n, ly + side * (keep + 20) * ny / n
        free = all(np.hypot(wx - ox, wy - oy) >= k for i, (ox, oy, k) in enumerate(things) if i != j)
        path = np.hypot(wx - x, wy - y) + np.hypot(gx - wx, gy - wy)
        options.append((not free, path, (wx, wy)))   # free sides first, then the shorter one
    return min(options)[2]


def approach(target, lms):
    """Target in view (go_to checked the path): turn to it and drive. Far away: shortcut straight
    in one go into BLIND_DIST (one camera re-aim there), close: blind + sonar. True once visited."""
    d, a = lms[target]
    ax, ay = ROBOT_FRONT + d * np.cos(a), d * np.sin(a)    # marker seen from the axle
    turn(np.arctan2(ay, ax))
    d = np.hypot(ax, ay) - ROBOT_FRONT
    if d <= BLIND_DIST:                 # camera loses the marker below ~90 cm -> blind, then sonar
        remaining = max(d - BLIND_STOP, 0.0)
        if free_dist(remaining) < remaining - 5.0:         # remembered box beside the blind path
            print("remembered obstacle next to the blind path to %s -> evade" % NAME[target])
            avoid(LANDMARKS[target]); return False
        driven, aborted = drive(remaining, sonar_limit=330)
        if aborted and remaining - driven > 15.0:          # sonar glitch -> try the rest once
            drive(remaining - driven, sonar_limit=330)
        gap = close_in()
        if gap is None:                 # camera saw the marker, sonar misses the (diagonal) box
            print("no echo -> staying at the blind stop (~%.0f cm, still < 40)" % BLIND_GAP)
            gap = BLIND_GAP
        snap_to(target, gap)
        print("%s visited (camera said %.0f cm, sonar now %.0f cm)" % (NAME[target], d, gap))
        return True
    dist = free_dist(d - BLIND_DIST + 10.0)                # shortcut: straight into BLIND_DIST
    if dist < 10 or drive(dist, fast=True)[1]:
        avoid(LANDMARKS[target])
    return False


def sonar_near(target):
    """Pose says target is too close for the camera: face it; echo fits the expected gap -> visit.
    Echo much closer -> something is in the way: evade. Otherwise we are lost -> relocalise."""
    gx, gy = LANDMARKS[target]
    x, y, th = pose()
    turn(wrap(np.arctan2(gy - y, gx - x) - th))
    expected = np.hypot(gx - x, gy - y) - BOX_HALF - SONAR_FRONT     # sonar -> box side
    _, obs = sense()
    s = front_sonar() / 10.0
    if (blocked(obs, expected + 40.0, 0.0) or 0 < s < expected - SONAR_MATCH
            or free_dist(s - BLIND_GAP) < s - BLIND_GAP - 5.0):   # seen, echo or remembered box
        print("echo %.0f cm, %s expected at %.0f cm, obstacles in view %s -> something in the way, evade"
              % (s, NAME[target], expected, list(obs)))
        avoid((gx, gy)); return False
    if not (0 < s and abs(s - expected) < SONAR_MATCH):
        print("sonar %.0f cm, expected %.0f cm -> relocalise" % (s, expected))
        particle.add_uncertainty(particles, 80.0, np.radians(30)); return False
    drive(max(s - BLIND_GAP, 0.0), sonar_limit=330)    # close_in only looks ~60 cm far
    g = close_in()
    if g is None:                       # no camera proof here, so no echo = not visited
        print("no echo after driving -> %s is not here, relocalise" % NAME[target])
        particle.add_uncertainty(particles, 80.0, np.radians(30)); return False
    snap_to(target, g)
    print("%s visited on sonar (expected %.0f cm, sonar said %.0f cm)" % (NAME[target], expected, s))
    return True


def go_to(target):
    """One leg: pose estimate until the camera sees the target, then approach."""
    gx, gy = LANDMARKS[target]
    for _ in range(60):
        lms, obs = sense()
        if blocked(obs, 80.0, 0.0):
            avoid((gx, gy)); continue
        if target not in lms and not localised():
            relocalise(target)
            continue
        if target in lms:
            d_t, a_t = lms[target]
            in_way = blocked(obs, d_t, a_t)
            if localised() and (in_way or d_t > BLIND_DIST):
                x, y, _ = pose()
                wp = detour(x, y, gx, gy, target)
                if wp is not None:
                    print("target in view, but blocked -> detour via (%.0f, %.0f)" % wp)
                    step_to(wp, min(STEP, np.hypot(wp[0] - x, wp[1] - y)))
                    continue
            if in_way:                  # sonar would take the obstacle for the target
                print("obstacle in front of %s -> evade" % NAME[target])
                avoid((gx, gy)); continue
            if approach(target, lms):
                return True
            continue
        x, y, th = pose()
        gap = np.hypot(gx - x, gy - y) - BOX_HALF - ROBOT_FRONT
        print("pose x=%.0f y=%.0f th=%.0f deg | spread %.0f/%.0f cm, heading %.2f | %s gap %.0f cm"
              % ((x, y, np.degrees(th)) + spread() + (NAME[target], gap)))
        wp = detour(x, y, gx, gy, target)           # also near the target: a box may hide it
        if wp is None and gap < SEE_DIST:           # should be visible: face it and look around
            turn(wrap(np.arctan2(gy - y, gx - x) - th))
            if target in scan(target, AHEAD_SCAN):
                continue
            x, y, _ = pose()
            wp = detour(x, y, gx, gy, target)       # the scan may just have found the hiding box
            if wp is None:
                print("%s not visible -> moving closer" % NAME[target])
                if gap > BLIND_DIST:
                    step_to((gx, gy), min(STEP, gap - BLIND_DIST + 10))
                elif sonar_near(target):
                    return True
                continue
        if wp is not None:
            print("detour via (%.0f, %.0f)" % wp)
            step_to(wp, min(STEP, np.hypot(wp[0] - x, wp[1] - y)))
        else:
            step_to((gx, gy), min(STEP, gap - SEE_DIST + 20))
    return False


def sonar_visit(lm_id):
    """Camera can't see lm_id: front sonar < VISIT_MAX -> we already stand close enough."""
    s = front_sonar()
    if 0 < s < VISIT_MAX * 10:
        print("%s not in view, but sonar %.0f cm -> visited" % (NAME[lm_id], s / 10.0))
        snap_to(lm_id, s / 10.0)
        return True
    return False


def visit_l1():
    """Start: turn until L1 is seen (camera) or close in front (sonar), visit it."""
    misses = 0
    for _ in range(60):
        lms, obs = sense()
        if L1 in lms:
            if blocked(obs, *lms[L1]):
                avoid(); continue
            if approach(L1, lms):
                return
            continue
        if sonar_visit(L1):             # too close for the camera -> already visited
            return
        misses += 1
        if misses == 12:                # full turn, nothing: 40-90 cm away? -> back up once
            print("L1 not found in a full turn -> backing up")
            back_up(BACK_DIST)
        elif misses % 12 == 0:
            print("L1 not found in a full turn -> moving a bit")
            if drive(60.0)[1]:
                avoid()
        else:
            turn(np.radians(30))


def main():
    print("map:", {NAME[i]: p for i, p in LANDMARKS.items()})
    t = time.time()
    visit_l1()                          # seeds the particles on a ring around L1
    relocalise(L2)                      # a 2nd landmark fixes our side
    legs = [("start->L1", time.time() - t)]
    for prev, target in zip(ROUTE, ROUTE[1:]):
        t = time.time()
        if not go_to(target):           # keep going: later landmarks still count
            print("could not reach %s - continuing" % NAME[target])
        legs.append(("%s->%s" % (NAME[prev], NAME[target]), time.time() - t))
    print("done - all landmarks visited")
    print("legs:  " + ", ".join("%s %.0f s" % lt for lt in legs))
    total = time.time() - T0
    print("time:  total %.0f s | sense %.0f s (%d x) | turn %.0f s | drive %.0f s | pause %.0f s | rest %.0f s"
          % (total, stats["sense"], stats["n_sense"], stats["turn"], stats["drive"], stats["pause"],
             total - stats["sense"] - stats["turn"] - stats["drive"] - stats["pause"]))


try:
    main()
finally:
    arlo.stop()
    cam.terminateCaptureThread()