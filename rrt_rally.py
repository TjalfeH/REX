r"""
rally.py - DIKU RoboRally 2026 (REX Ex6).

    L2 ---- 400 ---- L4      Route: L1 -> L2 -> L3 -> L4 -> L1
     |                |
    300  obstacles   300
     |                |
    L1 -------------- L3

Particle filter (Ex5) on the 4 corner landmarks. Approach a seen target on the camera,
last cm closed loop on the front sonar to VISIT_DIST. Otherwise drive on the pose
estimate around other landmarks and remembered obstacles,
planned with a simple RRT (replanned every step).

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
BRAKE = 3.0                             # roll-out after sonar stop [cm]
BLIND_DIST = 130.0                      # closer than this: no more camera checks
BLIND_STOP = 45.0                       # blind stretch ends at this camera -> marker distance
BLIND_GAP = BLIND_STOP - (SONAR_FRONT - ROBOT_FRONT)   # expected sonar gap after it
SEE_DIST = 160.0                        # camera should see the target from here
KEEP_AWAY = 100.0                       # path -> non-target landmark centre [cm]
OBST_HALF = 12.0                        # marker face -> obstacle centre [cm]
OBST_RADIUS = 17.5                      # obstacle radius [cm]
OBST_KEEP = OBST_RADIUS + ROBOT_RADIUS + 30.0
STEP = 50.0                             # step when target not in view [cm]
CAREFUL_DIST = 150.0                    # this close to a box / obstacle centre: slow, short steps [cm]
CAREFUL_STEP = 30.0                     # step there, so the sonars can react in time [cm]
SEARCH_R = 170.0                        # search circle radius around the target's map position [cm]
SEARCH_PTS = 8                          # look-points on the circle (45 deg apart)
BACK_DIST = 50.0                        # back up at start if L1 not visible [cm]
SONAR_STOP = 450                        # [mm] normal sonar abort
SIDE_STOP = 120                         # [mm] side sonar emergency stop while driving
SPEED, CM_PER_SEC, DEG_PER_SEC = 64, 46.0, 125.0   # turns + slow drives (last stretch to a box)
FAST_SPEED, FAST_CM_PER_SEC = 90, 65.0  # normal drives. 65 is a GUESS (46 * 90/64) - run calib!
T_START = 0.15                          # start-up time of a drive [s]
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
moved = True                            # moved since last filter update?
lost_count = 0                          # updates in a row where no particle fit


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
    arlo.stop(); time.sleep(0.3)
    for p in particles:
        particle.move_particle(p, 0.0, angle)
    particle.add_uncertainty(particles, 2.0, np.radians(3) + 0.1 * abs(angle))


def drive(dist, sonar_limit=SONAR_STOP, sides=True, fast=True):
    """Drive forward [cm], abort on front sonar < sonar_limit or (sides) a side sonar < SIDE_STOP.
    fast=False for the last stretch to a box: less roll-out after a sonar stop.
    Returns (driven, aborted)."""
    global moved
    if dist <= 0:
        return 0.0, False
    moved = True
    speed, cm_per_sec = (FAST_SPEED, FAST_CM_PER_SEC) if fast else (SPEED, CM_PER_SEC)
    t0, aborted, k = time.time(), False, 0
    arlo.go_diff(speed, speed + 2, 1, 1)
    while time.time() - t0 < dist / cm_per_sec + T_START:
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
    arlo.stop(); time.sleep(0.3)
    driven = min(dist, max(0.0, elapsed - T_START) * cm_per_sec)
    for p in particles:
        particle.move_particle(p, driven, 0.0)
    particle.add_uncertainty(particles, 2.0 + 0.05 * driven, np.radians(3))
    return driven, aborted


def avoid():
    """Turn 60 deg towards the freer side and move a bit."""
    l, r = arlo.read_left_ping_sensor(), arlo.read_right_ping_sensor()
    turn(np.radians(60) if (l if l > 0 else 9999) >= (r if r > 0 else 9999) else np.radians(-60))
    drive(35.0, fast=False)             # something is close by -> slow


def back_up(dist):
    """Reverse up to `dist` cm, keeping 20 cm to whatever is behind. Start only (no particles)."""
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
    arlo.stop(); time.sleep(0.3)


def near_box(x, y):
    """Distance from (x, y) to the closest landmark or remembered obstacle centre [cm]."""
    return min(np.hypot(x - px, y - py) for px, py in list(LANDMARKS.values()) + list(obstacles.values()))


def step_to(point, dist):
    """Turn towards world point (x, y), drive `dist` cm, evade if blocked.
    Near a box (anywhere on the way) only a short, slow step, so the sonars can stop us in time."""
    x, y, th = pose()
    h = np.arctan2(point[1] - y, point[0] - x)
    turn(wrap(h - th))
    careful = min(near_box(x + t * dist * np.cos(h), y + t * dist * np.sin(h))
                  for t in np.linspace(0, 1, 5)) < CAREFUL_DIST
    if careful:
        dist = min(dist, CAREFUL_STEP)
    if drive(dist, fast=not careful)[1]:
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
        drive(gap - VISIT_DIST + 3.0, sonar_limit=int((VISIT_DIST + BRAKE) * 10), sides=False, fast=False)
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
        if d > 350:
            continue
        r = d + OBST_HALF
        ox, oy = cx + r * np.cos(th + a), cy + r * np.sin(th + a)
        if not (-60 < ox < _W + 60 and -60 < oy < _H + 60):
            continue
        if i in obstacles:
            ox, oy = 0.5 * (ox + obstacles[i][0]), 0.5 * (oy + obstacles[i][1])
        else:
            print("obstacle %d at (%.0f, %.0f)" % (i, ox, oy))
        obstacles[i] = (ox, oy)


def sense():
    """3 frames, averaged per ID; updates the filter. Returns (landmarks, obstacles)."""
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
    return lms, obs


def blocked(obs, d_t, a_t):
    """True if a seen obstacle lies on the line to the point (d_t, a_t)."""
    return any(d * np.cos(a - a_t) < d_t and abs(d * np.sin(a - a_t)) < ROBOT_RADIUS + OBST_RADIUS
               for d, a in obs.values())


# ---- race ----------------------------------------------------------------------
def scan(target, steps=FULL_SCAN):
    """Turn and look after each step; stop early facing `target` if seen."""
    for s in steps:
        turn(np.radians(s))
        lms, _ = sense()
        if target in lms:
            return lms
    return {}


def relocalise(target):
    """Scan until localised. Returns the landmarks if the target showed up."""
    for _ in range(3):
        lms = scan(target)
        if target in lms or localised():
            return lms
        print("not localised yet -> move a bit and scan again")
        turn(np.radians(120)); drive(60.0)
    return {}


RRT_STEP = 30.0                         # tree extend step [cm]
RRT_ITER = 2000                         # max tree nodes tried


def zones(x, y, gx, gy, target):
    """Circles (cx, cy, keep) the path must stay out of: other landmarks and remembered obstacles.
    Right after a visit we stand ~70 cm from that box, inside its KEEP_AWAY circle. A circle that
    already contains us (or the goal) shrinks to just inside that point: we may drive out of it,
    but not deeper in. Otherwise the start is always 'in collision' and every plan fails."""
    return [(cx, cy, min(keep, np.hypot(x - cx, y - cy) - 5, np.hypot(gx - cx, gy - cy) - 5))
            for cx, cy, keep in keep_out(target)]


def keep_out(target):
    """Circles (cx, cy, keep) to stay out of: other landmarks and remembered obstacles."""
    things = [(lx, ly, KEEP_AWAY) for i, (lx, ly) in LANDMARKS.items() if i != target]
    return things + [(ox, oy, OBST_KEEP) for ox, oy in obstacles.values()]


def free(p, q, things):
    """True if the straight line p -> q stays out of every circle (checked every 5 cm)."""
    for t in np.linspace(0, 1, int(np.hypot(q[0] - p[0], q[1] - p[1]) / 5.0) + 2):
        px, py = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
        if any(np.hypot(px - cx, py - cy) < keep for cx, cy, keep in things):
            return False
    return True


def rrt(start, goal, things):
    """Plain RRT: grow a tree from start with random samples (10% the goal itself) until a node
    sees the goal. Returns the path start -> goal as a list of points, or None."""
    nodes, parent = [start], [None]
    for _ in range(RRT_ITER):
        if np.random.rand() < 0.1:
            sample = goal
        else:
            sample = (np.random.uniform(-60, _W + 60), np.random.uniform(-60, _H + 60))
        i = int(np.argmin([np.hypot(sample[0] - nx, sample[1] - ny) for nx, ny in nodes]))
        nx, ny = nodes[i]
        d = np.hypot(sample[0] - nx, sample[1] - ny)
        if d < 1e-9:
            continue
        s = min(RRT_STEP, d)
        new = (nx + (sample[0] - nx) / d * s, ny + (sample[1] - ny) / d * s)
        if not free(nodes[i], new, things):
            continue
        nodes.append(new); parent.append(i)
        if free(new, goal, things):     # goal in sight -> walk back to the start
            path, j = [goal], len(nodes) - 1
            while j is not None:
                path.append(nodes[j]); j = parent[j]
            return path[::-1]
    return None


def detour(x, y, gx, gy, target):
    """RRT around other landmarks and remembered obstacles. Returns the next waypoint,
    or None if the straight line to the goal is free (or no path was found)."""
    things = zones(x, y, gx, gy, target)
    if free((x, y), (gx, gy), things):
        return None
    path = rrt((x, y), (gx, gy), things)
    if path is None:
        print("RRT found no path -> driving straight")
        return None
    # RRT paths zigzag: take the furthest path point we can reach in a straight line
    return next(p for p in reversed(path[1:]) if free((x, y), p, things))


def approach(target, lms):
    """Target in view: turn to it and drive. Returns True once visited."""
    d, a = lms[target]
    ax, ay = ROBOT_FRONT + d * np.cos(a), d * np.sin(a)    # marker seen from the axle
    turn(np.arctan2(ay, ax))
    d = np.hypot(ax, ay) - ROBOT_FRONT
    if d <= BLIND_DIST:                 # camera loses the marker below ~90 cm -> blind, then sonar
        remaining = max(d - BLIND_STOP, 0.0)
        driven, aborted = drive(remaining, sonar_limit=330, fast=False)
        if aborted and remaining - driven > 15.0:          # sonar glitch -> try the rest once
            drive(remaining - driven, sonar_limit=330, fast=False)
        gap = close_in()
        snap_to(target, gap)
        print("%s visited (camera said %.0f cm, sonar now %.0f cm)" % (NAME[target], d, gap))
        return True
    if drive(min(d - BLIND_DIST + 10.0, 80.0), fast=d > CAREFUL_DIST)[1]:   # stop inside BLIND_DIST
        avoid()
    return False


def circle_search(target):
    """Target should be in view but is not (hidden / estimate off): drive around its map position
    on a circle. At each look-point face the centre and look left and right.
    Returns the landmarks as soon as the target is seen, else {}."""
    gx, gy = LANDMARKS[target]
    x, y, _ = pose()
    a0 = np.arctan2(y - gy, x - gx)     # start at the circle point closest to us
    for k in range(SEARCH_PTS):
        a = a0 + k * 2 * np.pi / SEARCH_PTS
        px, py = gx + SEARCH_R * np.cos(a), gy + SEARCH_R * np.sin(a)
        if not (-60 < px < _W + 60 and -60 < py < _H + 60):
            continue                    # outside the track
        if any(np.hypot(px - cx, py - cy) < keep for cx, cy, keep in keep_out(target)):
            continue                    # too close to another box
        print("search %s: look-point %d/%d at (%.0f, %.0f)" % (NAME[target], k + 1, SEARCH_PTS, px, py))
        for _ in range(8):              # move there (around obstacles), looking after every step
            x, y, _ = pose()
            if np.hypot(px - x, py - y) < 15:
                break
            wp = detour(x, y, px, py, target) or (px, py)
            step_to(wp, min(STEP, np.hypot(wp[0] - x, wp[1] - y)))
            lms, _ = sense()
            if target in lms:
                return lms
        x, y, th = pose()
        turn(wrap(np.arctan2(gy - y, gx - x) - th))    # face where we think it is
        lms = scan(target, AHEAD_SCAN)
        if target in lms:
            return lms
    return {}


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
            print("%s not visible -> circle search around its map position" % NAME[target])
            if not circle_search(target):
                particle.add_uncertainty(particles, 80.0, np.radians(30))   # lost -> relocalise
            continue
        wp = detour(x, y, gx, gy, target)
        if wp is not None:
            print("detour via (%.0f, %.0f)" % wp)
            step_to(wp, min(STEP, np.hypot(wp[0] - x, wp[1] - y)))
        else:
            step_to((gx, gy), min(STEP, gap - SEE_DIST + 20))
    return False


def main():
    print("map:", {NAME[i]: p for i, p in LANDMARKS.items()})
    if L1 not in sense()[0]:            # maybe too close to L1 -> back up
        print("L1 not in the first image - maybe too close -> backing up")
        back_up(BACK_DIST)
    misses = 0
    for _ in range(60):                 # find L1 and visit it (seeds the particles)
        lms, obs = sense()
        if L1 not in lms:
            misses += 1
            if misses % 12 == 0:
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
    relocalise(L2)                      # ring around L1 -> a 2nd landmark fixes our side
    for target in ROUTE[1:]:
        if not go_to(target):           # keep going: later landmarks still count
            print("could not reach %s - continuing" % NAME[target])
    print("done - all landmarks visited")


def calibrate():
    """12 x 30 deg = one full turn; 30 cm short by x -> T_START += x / CM_PER_SEC; 100 cm exact.
    Fast 100 cm drove y cm -> FAST_CM_PER_SEC *= y / 100."""
    print("12 x 30 deg left -> robot must end up facing exactly the same way")
    for _ in range(12):
        turn(np.radians(30))
    input("Enter -> drive 30 cm slow (no sonar stop), then measure")
    drive(30.0, sonar_limit=0, sides=False, fast=False)
    input("Enter -> drive 100 cm slow (no sonar stop), then measure")
    drive(100.0, sonar_limit=0, sides=False, fast=False)
    input("Enter -> drive 100 cm FAST (no sonar stop), then measure")
    drive(100.0, sonar_limit=0, sides=False)


try:
    calibrate() if "calib" in sys.argv[1:] else main()
finally:
    arlo.stop()
    cam.terminateCaptureThread()