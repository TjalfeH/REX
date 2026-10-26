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

Race day: landmark IDs 1-4, run `python3 rally.py calib`, measure ROBOT_FRONT and
SONAR_FRONT, then `python3 rally.py | tee run.log`.
"""
import sys
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
STEP = 50.0                             # step when target not in view [cm]
BACK_DIST = 50.0                        # back up at start if L1 not found in a full turn [cm]
SONAR_STOP = 450                        # [mm] normal sonar abort
SIDE_STOP = 120                         # [mm] side sonar emergency stop while driving
SPEED, CM_PER_SEC, DEG_PER_SEC = 64, 46.0, 125.0
T_START = 0.15                          # start-up time of a drive [s]
SETTLE = 0.3                            # pause after every move [s]
N = 1000                                # particles
SIGMA_A = np.radians(10.0)              # bearing noise

FULL_SCAN = [0] + [-30] * 11            # look all around [deg]
AHEAD_SCAN = [0, -30, -30, 90, 30]      # 0, -30, -60, +30, +60
SONAR_SWEEP = [-15, -15, 45, 15]        # sonar at -15, -30, +15, +30

arlo = robot.Robot()
cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
cam.intrinsic_matrix = np.array([[CAM_F, 0, 820.0], [0, CAM_F, 616.0], [0, 0, 1]], dtype=np.float64)
cam.arucoMarkerLength = MARKER_SIZE
particles = []
obstacles = {}                          # id -> (x, y) world position
obst_w = {}                             # id -> weight of that position (sum of 1/distance)
moved = True                            # moved since last filter update?
lost_count = 0                          # updates in a row where no particle fit
T0 = time.time()                        # run start (log time stamps)
stats = {"sense": 0.0, "turn": 0.0, "drive": 0.0, "pause": 0.0, "n_sense": 0}

_print = print
def print(*args, **kw):
    """Every log line gets the seconds since start: [  12.3] ..."""
    kw.setdefault("flush", True)        # show live through `| tee run.log`
    _print("[%6.1f]" % (time.time() - T0), *args, **kw)


def settle():
    time.sleep(SETTLE)
    stats["pause"] += SETTLE


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
    global moved
    deg = np.degrees(angle)
    if abs(deg) < 2:
        return
    moved = True
    arlo.go_diff(SPEED, SPEED, 0, 1) if deg > 0 else arlo.go_diff(SPEED, SPEED, 1, 0)
    time.sleep(abs(deg) / DEG_PER_SEC)
    stats["turn"] += abs(deg) / DEG_PER_SEC
    arlo.stop(); settle()
    for p in particles:
        particle.move_particle(p, 0.0, angle)
    particle.add_uncertainty(particles, 2.0, np.radians(3) + 0.1 * abs(angle))


def drive(dist, sonar_limit=SONAR_STOP, sides=True):
    """Drive forward [cm], abort on front sonar < sonar_limit or (sides) a side sonar < SIDE_STOP.
    Returns (driven, aborted)."""
    global moved
    if dist <= 0:
        return 0.0, False
    moved = True
    t0, aborted, k = time.time(), False, 0
    arlo.go_diff(SPEED, SPEED + 2, 1, 1)
    while time.time() - t0 < dist / CM_PER_SEC + T_START:
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
    arlo.stop(); settle()
    driven = min(dist, max(0.0, elapsed - T_START) * CM_PER_SEC)
    for p in particles:
        particle.move_particle(p, driven, 0.0)
    particle.add_uncertainty(particles, 2.0 + 0.05 * driven, np.radians(3))
    return driven, aborted


def avoid():
    """Turn 60 deg to the freer side and move 35 cm. Not free -> try the other side,
    neither free -> back up. So we always move instead of turning back and forth."""
    l, r = arlo.read_left_ping_sensor(), arlo.read_right_ping_sensor()
    a = 60 if (l if l > 0 else 9999) >= (r if r > 0 else 9999) else -60
    for ang in (a, -2 * a):
        turn(np.radians(ang))
        _, obs = sense()
        if not blocked(obs, 35.0 + ROBOT_RADIUS, 0.0) and free_dist(35.0) >= 35.0:
            drive(35.0); return
    turn(np.radians(a))                 # back to the original heading
    back_up(25.0)


def back_up(dist):
    """Reverse up to `dist` cm, keeping 20 cm to whatever is behind. Moves the particles too."""
    global moved
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
    arlo.stop(); settle()
    driven = min(dist, max(0.0, elapsed - T_START) * CM_PER_SEC)
    moved = True
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
        avoid(); return
    if drive(dist)[1]:
        avoid()


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
    """Creep on the front sonar until VISIT_DIST. Returns final gap [cm]."""
    limit = BLIND_GAP + 25.0            # echo further away = not our box
    s = front_sonar()
    gap = s / 10.0 if 0 < s < limit * 10 else None
    if gap is None:
        print("front sonar sees no box (diagonal?) -> sonar sweep")
        gap = find_box_echo(limit)
    if gap is None:
        print("no echo -> staying at the blind stop (~%.0f cm, still < 40)" % BLIND_GAP)
        return BLIND_GAP
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


def localised():
    """Particles agree on position AND heading (a ring around one landmark does not)."""
    if not particles:
        return False
    xs, ys, th = as_arrays()
    heading_agree = np.hypot(np.mean(np.cos(th)), np.mean(np.sin(th)))
    return np.std(xs) < 50 and np.std(ys) < 50 and heading_agree > 0.9


def snap_to(lm_id, gap):
    """We face lm_id at `gap` cm (front sonar -> box side): move all particles there."""
    global moved
    if not particles:                   # first visit: ring, a scan decides the side
        set_particles(*ring(lm_id, gap + SONAR_FRONT - ROBOT_FRONT, 0.0))
    else:
        xs, ys, ths = as_arrays()
        ths = wrap(ths + np.random.normal(0, np.radians(3), len(ths)))
        r = gap + BOX_HALF + SONAR_FRONT
        lx, ly = LANDMARKS[lm_id]
        set_particles(lx - r * np.cos(ths), ly - r * np.sin(ths), ths)
    moved = True


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
    """3 frames, averaged per ID; updates the filter. Returns (landmarks, obstacles)."""
    t0 = time.time()
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
    obs = {i: m for i, m in seen.items() if i not in LANDMARKS}
    filter_update(lms)
    remember(obs)
    stats["sense"] += time.time() - t0
    stats["n_sense"] += 1
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
    """Target in view: turn to it and drive. Returns True once visited."""
    d, a = lms[target]
    ax, ay = ROBOT_FRONT + d * np.cos(a), d * np.sin(a)    # marker seen from the axle
    turn(np.arctan2(ay, ax))
    d = np.hypot(ax, ay) - ROBOT_FRONT
    if d <= BLIND_DIST:                 # camera loses the marker below ~90 cm -> blind, then sonar
        remaining = max(d - BLIND_STOP, 0.0)
        driven, aborted = drive(remaining, sonar_limit=330)
        if aborted and remaining - driven > 15.0:          # sonar glitch -> try the rest once
            drive(remaining - driven, sonar_limit=330)
        gap = close_in()
        snap_to(target, gap)
        print("%s visited (camera said %.0f cm, sonar now %.0f cm)" % (NAME[target], d, gap))
        return True
    if drive(min(d - BLIND_DIST + 10.0, 80.0))[1]:          # stop inside BLIND_DIST
        avoid()
    return False


def go_to(target):
    """One leg: pose estimate until the camera sees the target, then approach."""
    gx, gy = LANDMARKS[target]
    for _ in range(60):
        lms, obs = sense()
        if blocked(obs, 80.0, 0.0):
            avoid(); continue
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
                avoid(); continue
            if approach(target, lms):
                return True
            continue
        x, y, th = pose()
        gap = np.hypot(gx - x, gy - y) - BOX_HALF - ROBOT_FRONT
        print("pose x=%.0f y=%.0f th=%.0f deg | %s gap %.0f cm"
              % (x, y, np.degrees(th), NAME[target], gap))
        if gap < SEE_DIST:              # should be visible: face it and look around
            turn(wrap(np.arctan2(gy - y, gx - x) - th))
            if target in scan(target, AHEAD_SCAN):
                continue
            print("%s not visible -> moving closer" % NAME[target])
            if gap > BLIND_DIST:
                step_to((gx, gy), min(STEP, gap - BLIND_DIST + 10))
            else:
                particle.add_uncertainty(particles, 80.0, np.radians(30))   # lost -> relocalise
            continue
        wp = detour(x, y, gx, gy, target)
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


def calibrate():
    """12 x 30 deg = one full turn; 30 cm short by x -> T_START += x / CM_PER_SEC; 100 cm exact."""
    print("12 x 30 deg left -> robot must end up facing exactly the same way")
    for _ in range(12):
        turn(np.radians(30))
    input("Enter -> drive 30 cm (no sonar stop), then measure")
    drive(30.0, sonar_limit=0, sides=False)
    input("Enter -> drive 100 cm (no sonar stop), then measure")
    drive(100.0, sonar_limit=0, sides=False)


try:
    calibrate() if "calib" in sys.argv[1:] else main()
finally:
    arlo.stop()
    cam.terminateCaptureThread()