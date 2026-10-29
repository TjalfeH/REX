"""
rally.py - DIKU RoboRally 2026 (REX Ex6). Route L1 -> L2 -> L3 -> L4 -> L1.
Particle filter (Ex5) on the 4 corner landmarks. Target in the camera: drive to it, the last
bit on the front sonar. Otherwise drive on the pose estimate with an RRT detour around the other
landmarks and the obstacles we have seen. Units: cm and rad. Race day: landmark IDs 1-4.
Run: python3 rally.py | tee run.log
"""
import time
import numpy as np
import camera, robot

BOX_HALF = 20.0                                     # half side of a landmark box
L1, L2, L3, L4 = 1, 2, 3, 4                         # practice IDs - race day: 1, 2, 3, 4 !
ROBOT_FRONT, ROBOT_RADIUS, SONAR_FRONT = 21.0, 28.0, 28.0   # axle -> camera, radius, axle -> sonar
CAM_F, MARKER_SIZE = 1315.0, 0.145                  # calibration from Ex4
W, H = 400.0 + 2 * BOX_HALF, 300.0 + 2 * BOX_HALF   # box centre to box centre
LANDMARKS = {L1: (0.0, 0.0), L2: (0.0, H), L3: (W, 0.0), L4: (W, H)}
ROUTE = [L1, L2, L3, L4, L1]
NAME = {L1: "L1", L2: "L2", L3: "L3", L4: "L4"}

# final approach: sonar gap at the end, sonar/pose agreement, roll-out after a sonar stop,
# no camera checks closer than BLIND_DIST, camera should see the target from SEE_DIST,
# no detours closer than NEAR
VISIT_DIST, SONAR_MATCH, BRAKE, BLIND_DIST, SEE_DIST, NEAR = 25.0, 30.0, 9.0, 130.0, 160.0, 60.0
BLIND_GAP = 35.0 - (SONAR_FRONT - ROBOT_FRONT)      # sonar gap after the blind drive
BLIND_SONAR_STOP, ECHO_RANGE = 33.0, BLIND_GAP + 25.0

# obstacles: path -> other landmark centre, marker -> obstacle centre, obstacle radius,
# min distance to an obstacle centre (15 cm air between robot and box)
KEEP_AWAY, OBST_HALF, OBST_RADIUS = 100.0, 12.0, 17.5
CLEAR = OBST_RADIUS + ROBOT_RADIUS + 15.0
OBST_KEEP = CLEAR                                   # RRT circle, never smaller than CLEAR
LM_EXTRA = KEEP_AWAY - CLEAR                        # landmarks get this much more room
OBST_NEAR, OBST_MAX_SEEN, STEP = 80.0, 250.0, 140.0
EVADE_TURN, EVADE_DIST, EVADE_BACK, BACK_KEEP = 60.0, 35.0, 25.0, 20.0
BACK_FIRST, BACK_AFTER = 15.0, 30.0                 # back up after a visit: first landmark, the others

# RRT: step, max nodes, tries, point reached, sample box pad, arena margin, detour range
RRT_STEP, RRT_ITER, RRT_TRIES, RRT_REACHED, RRT_PAD = 50.0, 3000, 8, 15.0, 200.0
ARENA_OUT, DETOUR_RANGE = 120.0, 160.0

SONAR_STOP, SIDE_STOP, SIDE_FREE = 45.0, 12.0, 60.0
SPEED, CM_PER_SEC, DEG_PER_SEC, TRIM = 64, 46.0, 125.0, 2
T_START, SETTLE, STOP_PAUSE, FLUSH_FRAMES = 0.15, 0.3, 0.1, 2
N, SIGMA_A, LOC_STD = 500, np.radians(10.0), 50.0
FULL_SCAN = [0, -55, -110, -165, -220, -275, -330]  # deg, camera sees 64 deg
AHEAD_SCAN, SONAR_SWEEP = [0, -45, 45], [0, -15, -30, 15, 30]

arlo = robot.Robot()
cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
cam.intrinsic_matrix = np.array([[CAM_F, 0, 820.0], [0, CAM_F, 616.0], [0, 0, 1]], dtype=np.float64)
cam.arucoMarkerLength = MARKER_SIZE

P = np.empty((0, 3))        # particles (x, y, theta)
obstacles = {}              # id -> (x, y, weight)
last_view = None            # last sense() result, None after a move
lost_count = 0
used = set()                # landmarks the particles are based on
target_now = None
plan = None                 # (target, path, i): RRT path we follow, path[i] is next
started = False             # L1 visited at the start?
T0 = time.time()

def log(*args):
    print("[%6.1f]" % (time.time() - T0), *args, flush=True)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def cam_pos(x, y, th):
    return x + ROBOT_FRONT * np.cos(th), y + ROBOT_FRONT * np.sin(th)

def along_across(x, y, th, px, py):
    """Point seen from (x, y) along th: distance ahead, |distance to the side|."""
    dx, dy = px - x, py - y
    return dx * np.cos(th) + dy * np.sin(th), abs(dy * np.cos(th) - dx * np.sin(th))

def sonar(side="front"):
    mm = getattr(arlo, "read_%s_ping_sensor" % side)()
    return mm / 10.0 if mm > 0 else np.inf

def front_sonar():
    return sorted([sonar(), sonar(), sonar()])[1]

# ---- motion --------------------------------------------------------------------------
def add_noise(s_xy, s_th):
    global P
    P = P + np.random.normal(0, [s_xy, s_xy, s_th], P.shape)
    P[:, 2] = wrap(P[:, 2])

def move_particles(d=0.0, a=0.0):
    global last_view
    last_view = None
    P[:, 0] += d * np.cos(P[:, 2])
    P[:, 1] += d * np.sin(P[:, 2])
    P[:, 2] += a
    add_noise(2.0 + 0.05 * abs(d), np.radians(3) + 0.1 * abs(a))

def turn(angle):                                    # positive = left
    deg = np.degrees(angle)
    if abs(deg) < 2:
        return
    arlo.go_diff(SPEED, SPEED, int(deg < 0), int(deg > 0))
    time.sleep(abs(deg) / DEG_PER_SEC)
    arlo.stop()
    time.sleep(STOP_PAUSE)
    move_particles(a=angle)

def straight(dist, stop):
    """Drive dist cm (negative = back) until done or stop(). Returns (driven, aborted)."""
    fwd = int(dist > 0)
    arlo.go_diff(SPEED, SPEED + TRIM * fwd, fwd, fwd)
    t0, aborted = time.time(), False
    while time.time() - t0 < abs(dist) / CM_PER_SEC + T_START:
        if stop():
            aborted = True
            break
        time.sleep(0.05)
    elapsed = time.time() - t0
    arlo.stop()
    time.sleep(STOP_PAUSE)
    driven = min(abs(dist), max(0.0, elapsed - T_START) * CM_PER_SEC)
    move_particles(driven if fwd else -driven)
    return driven, aborted

def drive(dist, front_stop=SONAR_STOP, sides=True):
    """Forward; stops on front sonar < front_stop or (sides) a side sonar < SIDE_STOP."""
    if dist <= 0:
        return 0.0, False
    check = ["left", "right"]

    def stop():
        if sonar() < front_stop:
            return True
        if sides:
            check.append(check.pop(0))              # one side per loop
            if sonar(check[-1]) < SIDE_STOP:
                log("%s sonar -> emergency stop" % check[-1])
                return True
        return False
    return straight(dist, stop)

def back_up(dist):
    dist = min(dist, sonar("back") - BACK_KEEP)
    if dist >= 10:
        log("backing up %.0f cm" % dist)
        straight(-dist, lambda: sonar("back") < BACK_KEEP)

def avoid(goal=None):
    """Turn to the freer side (localised: towards goal) and drive a bit; else back up."""
    l, r = sonar("left"), sonar("right")
    a = EVADE_TURN if l >= r else -EVADE_TURN
    if goal is not None and min(l, r) > SIDE_FREE and localised():
        x, y, th = pose()
        a = EVADE_TURN if wrap(np.arctan2(goal[1] - y, goal[0] - x) - th) > 0 else -EVADE_TURN
    for ang in (a, -2 * a):
        turn(np.radians(ang))
        _, obs = sense()
        if not blocked(obs, EVADE_DIST + ROBOT_RADIUS, 0.0) and free_dist(EVADE_DIST) >= EVADE_DIST:
            drive(EVADE_DIST)
            return
    turn(np.radians(a))
    back_up(EVADE_BACK)

def free_dist(dist):
    """How far we can drive ahead before getting closer than CLEAR to a remembered obstacle
    or CLEAR + LM_EXTRA to another landmark."""
    if len(P) == 0:
        return dist
    if localised():
        x, y, th = pose()
        things = [(ox, oy, 0.0) for ox, oy, _ in obstacles.values()]
        things += [(lx, ly, LM_EXTRA) for i, (lx, ly) in LANDMARKS.items() if i != target_now]
    elif len(used) == 1 and target_now not in used:   # ring around one landmark: keep away from it
        x, y, th = P[0]
        things = [LANDMARKS[list(used)[0]] + (LM_EXTRA,)]
    else:
        return dist
    for ox, oy, extra in things:
        ahead, side = along_across(x, y, th, ox, oy)
        # already inside: may get 10 cm closer, never closer than 5 cm to touching
        keep = min(CLEAR + extra, max(np.hypot(ox - x, oy - y) - 10.0, OBST_RADIUS + ROBOT_RADIUS + 5.0 + extra))
        if ahead > 0 and side < keep:
            dist = min(dist, ahead - np.sqrt(keep ** 2 - side ** 2))
    return dist

def blocked(obs, d_t, a_t):
    """A seen obstacle closer than CLEAR to the line to (d_t, a_t)?"""
    for d, a in obs.values():
        along, across = along_across(0.0, 0.0, a_t, (d + OBST_HALF) * np.cos(a), (d + OBST_HALF) * np.sin(a))
        if along < d_t and across < CLEAR:
            return True
    return False

def step_to(point, dist):
    """Turn to a world point, look, drive as far as it is free. Evade -> plan dropped."""
    global plan
    x, y, th = pose()
    turn(wrap(np.arctan2(point[1] - y, point[0] - x) - th))
    _, obs = sense()
    dist = free_dist(dist)
    if dist < 10 or blocked(obs, dist + ROBOT_RADIUS, 0.0):
        log("path not free -> evade")
        avoid(point)
        plan = None
    elif drive(dist)[1]:
        avoid(point)
        plan = None

def close_in():
    """Creep on the front sonar until VISIT_DIST. Returns the gap, None if no echo."""
    gap = front_sonar()
    if gap >= ECHO_RANGE:                           # no echo: sonar sweep for the box
        best, gap, cur = 0, ECHO_RANGE, 0
        for a in SONAR_SWEEP:
            turn(np.radians(a - cur))
            cur = a
            s = front_sonar()
            if s < gap:
                best, gap = a, s
        turn(np.radians(best - cur))
        if gap >= ECHO_RANGE:
            return None
    for _ in range(2):
        if gap <= VISIT_DIST + BRAKE:
            break
        drive(gap - VISIT_DIST + 3.0, front_stop=VISIT_DIST + BRAKE, sides=False)
        gap = front_sonar()
        if gap >= ECHO_RANGE:
            return VISIT_DIST
    return gap

# ---- particle filter -----------------------------------------------------------------
def ring(lm_id, d, a, n=N):
    """n particles that all see lm_id at distance d and bearing a."""
    lx, ly = LANDMARKS[lm_id]
    phi = np.random.uniform(-np.pi, np.pi, n)
    r = d + BOX_HALF + np.random.normal(0, 10, n)
    ths = wrap(phi + np.pi - a + np.random.normal(0, np.radians(5), n))
    cx, cy = lx + r * np.cos(phi), ly + r * np.sin(phi)
    return np.column_stack([cx - ROBOT_FRONT * np.cos(ths), cy - ROBOT_FRONT * np.sin(ths), ths])

def filter_update(meas, check_lost=True):
    """Weight the particles with the landmark measurements and resample."""
    global P, lost_count, used
    if not meas:
        return
    closest = min(meas, key=lambda i: meas[i][0])
    if len(P) == 0:
        log("first landmark %s -> ring" % NAME[closest])
        P, used, check_lost = ring(closest, *meas[closest]), {closest}, False
    cx, cy = cam_pos(P[:, 0], P[:, 1], P[:, 2])
    logw = np.zeros(len(P))
    for lm_id, (d, a) in meas.items():
        lx, ly = LANDMARKS[lm_id]
        ed = np.hypot(lx - cx, ly - cy) - BOX_HALF
        ea = wrap(np.arctan2(ly - cy, lx - cx) - P[:, 2])
        logw += -0.5 * ((d - ed) / (10.0 + 0.08 * d)) ** 2 - 0.5 * (wrap(a - ea) / SIGMA_A) ** 2
    if check_lost and logw.max() < -8.0 * len(meas):  # nothing fits
        lost_count += 1
        if lost_count < 2:
            log("measurement fits no particle -> ignored once")
            return
        lost_count = 0
        log("LOST -> new ring around %s" % NAME[closest])
        P, used = ring(closest, *meas[closest]), {closest}
        filter_update(meas, check_lost=False)
        return
    lost_count = 0
    used = used | set(meas)
    w = np.exp(logw - logw.max())
    cum = np.cumsum(w / w.sum())
    cum[-1] = 1.0
    P = P[np.searchsorted(cum, (np.random.uniform() + np.arange(len(w))) / len(w))]

def pose():
    return P[:, 0].mean(), P[:, 1].mean(), np.arctan2(np.sin(P[:, 2]).mean(), np.cos(P[:, 2]).mean())

def localised():
    if len(P) == 0 or len(used) < 2:                # one landmark only gives a ring
        return False
    agree = np.hypot(np.cos(P[:, 2]).mean(), np.sin(P[:, 2]).mean())
    return P[:, 0].std() < LOC_STD and P[:, 1].std() < LOC_STD and agree > 0.9

def snap_to(lm_id, gap):
    """We stand in front of lm_id with sonar reading gap: put all particles there."""
    global P, last_view, started
    started = True
    ths = wrap(P[:, 2] + np.random.normal(0, np.radians(3), len(P)))
    r = gap + BOX_HALF + SONAR_FRONT
    lx, ly = LANDMARKS[lm_id]
    P = np.column_stack([lx - r * np.cos(ths), ly - r * np.sin(ths), ths])
    last_view = None

def remember(obs):
    """Store seen obstacles in world coordinates (weighted mean, close sightings count more)."""
    if not obs or not localised():
        return
    x, y, th = pose()
    cx, cy = cam_pos(x, y, th)
    for i, (d, a) in obs.items():
        if d > OBST_MAX_SEEN:
            continue
        ox, oy, w = cx + (d + OBST_HALF) * np.cos(th + a), cy + (d + OBST_HALF) * np.sin(th + a), 1.0 / d
        far = ARENA_OUT + OBST_KEEP                 # further out can't touch a path: wrong reading
        if not (-far < ox < W + far and -far < oy < H + far):
            continue
        if i in obstacles:
            px, py, pw = obstacles[i]
            ox, oy, w = (pw * px + w * ox) / (pw + w), (pw * py + w * oy) / (pw + w), pw + w
        else:
            log("obstacle %d at (%.0f, %.0f)" % (i, ox, oy))
        obstacles[i] = (ox, oy, w)

def sense():
    """Take a picture, update filter and obstacle map. Returns (landmarks, obstacles) as
    {id: (distance, bearing)}. Same result again if we did not move."""
    global last_view
    if last_view is not None:
        return last_view
    time.sleep(SETTLE - STOP_PAUSE)
    raw = {}
    try:
        for _ in range(FLUSH_FRAMES):
            cam.get_next_frame()
        frame = cam.get_next_frame()
        ids, ds, angs = cam.detect_aruco_objects(frame) if frame is not None else (None, None, None)
    except Exception as e:
        log("camera error: %s" % e)
        ids = None
    if ids is not None:
        for i, d, a in zip(ids, ds, angs):
            if 0 < d < 500:
                raw.setdefault(int(i), []).append((float(d), float(a)))
    seen = {i: tuple(np.mean(v, axis=0)) for i, v in raw.items()}   # several sides of a box: mean
    log("seen:", {i: (round(d), round(np.degrees(a))) for i, (d, a) in seen.items()})
    lms = {i: m for i, m in seen.items() if i in LANDMARKS}
    obs = {i: m for i, m in seen.items() if i not in LANDMARKS}
    filter_update(lms)
    remember(obs)
    last_view = (lms, obs)
    return last_view

def scan(target, angles=FULL_SCAN, until_localised=False):
    """Look at the given angles; stop early when the target is seen (or we are localised)."""
    cur = 0
    for a in angles:
        turn(np.radians(a - cur))
        cur = a
        lms, _ = sense()
        if target in lms or (until_localised and localised()):
            return lms
    return {}

def relocalise(target):
    for _ in range(3):
        lms = scan(target, until_localised=True)
        if target in lms or localised():
            return lms
        log("not localised -> move and scan again")
        turn(np.radians(120))
        drive(60.0)
    return {}

# ---- RRT -----------------------------------------------------------------------------
def first_hit(p, q, things):
    """Distance along p -> q to the first point inside a circle (every 5 cm), None if free."""
    length = np.hypot(q[0] - p[0], q[1] - p[1])
    for t in np.linspace(0, 1, int(length / 5.0) + 2):
        px, py = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
        for cx, cy, r in things:
            if np.hypot(px - cx, py - cy) < r:
                return t * length
    return None

def free(p, q, things):
    return first_hit(p, q, things) is None

def rrt_once(start, goal, things, pad):
    """Plain RRT from start until a node sees the goal. Returns the path or None.
    Samples only within the bound (robot centre ARENA_OUT - ROBOT_RADIUS outside the arena). A start
    outside it still works: every step heads for a sample inside, so no node is further out than
    its parent - the path only leads back in."""
    nodes, parent = [start], [None]
    out = ARENA_OUT - ROBOT_RADIUS
    lo = np.maximum(np.minimum(start, goal) - pad, (-out, -out))
    hi = np.minimum(np.maximum(start, goal) + pad, (W + out, H + out))
    xy = np.empty((RRT_ITER + 1, 2))                # nodes as array for a fast nearest search
    xy[0] = start
    for _ in range(RRT_ITER):
        sample = goal if np.random.rand() < 0.1 else tuple(np.random.uniform(lo, hi))
        n = len(nodes)
        i = int(np.argmin(np.hypot(xy[:n, 0] - sample[0], xy[:n, 1] - sample[1])))
        nx, ny = nodes[i]
        d = np.hypot(sample[0] - nx, sample[1] - ny)
        if d < 1e-9:
            continue
        s = min(RRT_STEP, d)
        new = (nx + (sample[0] - nx) / d * s, ny + (sample[1] - ny) / d * s)
        if not free(nodes[i], new, things):
            continue
        xy[n] = new
        nodes.append(new)
        parent.append(i)
        if free(new, goal, things):                 # goal in sight: walk back to the start
            path, j = [goal], n
            while j is not None:
                path.append(nodes[j])
                j = parent[j]
            return path[::-1]
    return None

def rrt(start, goal, things):
    """Shortest of RRT_TRIES paths (zigzag removed), cut into RRT_STEP pieces. None if no path."""
    for pad in (RRT_PAD, 2 * RRT_PAD):              # bigger sample box if no path
        best, best_len = None, np.inf
        for _ in range(RRT_TRIES):
            path = rrt_once(start, goal, things, pad)
            if path is None:
                continue
            out, i = [path[0]], 0                   # shortcut: jump to the furthest free point
            while i < len(path) - 1:
                j = len(path) - 1
                while j > i + 1 and not free(path[i], path[j], things):
                    j -= 1
                out.append(path[j])
                i = j
            length = sum(np.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(out, out[1:]))
            if length < best_len:
                best, best_len = out, length
        if best is not None:
            steps = [best[0]]                       # split into drive steps
            for p, q in zip(best, best[1:]):
                n = int(np.ceil(np.hypot(q[0] - p[0], q[1] - p[1]) / RRT_STEP))
                steps += [(p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n) for k in range(1, n + 1)]
            return steps
    return None

def next_point(x, y, path, i, things):
    """Next path index to drive to, None if the rest of the path is not free any more."""
    while i < len(path) - 1 and np.hypot(path[i][0] - x, path[i][1] - y) < RRT_REACHED:
        i += 1
    if free((x, y), path[i], things) and all(free(path[k], path[k + 1], things) for k in range(i, len(path) - 1)):
        return i
    return None

def step_detour(target, why):
    """One RRT step around other landmarks and remembered obstacles if the straight line
    needs one. Follows an existing plan as long as it stays free. True if we stepped."""
    global plan
    x, y, _ = pose()
    gx, gy = LANDMARKS[target]
    # circles to stay out of; one we or the goal are inside of shrinks to that distance
    things = [(lx, ly, KEEP_AWAY) for i, (lx, ly) in LANDMARKS.items() if i != target]
    things += [(ox, oy, OBST_KEEP) for ox, oy, _ in obstacles.values()]
    things = [(cx, cy, min(r, np.hypot(x - cx, y - cy), np.hypot(gx - cx, gy - cy))) for cx, cy, r in things]
    hit = first_hit((x, y), (gx, gy), things)
    if hit is None or hit > DETOUR_RANGE:
        plan = None
        return False
    i = None
    if plan is not None and plan[0] == target:
        path = plan[1]
        i = next_point(x, y, path, plan[2], things)
    if i is None:
        path = rrt((x, y), (gx, gy), things)
        i = next_point(x, y, path, 1, things) if path else None
        if i is None:
            plan = None
            log("RRT found no path -> driving straight")
            return False
        log("RRT: new path to %s (%d steps), blocked %.0f cm ahead" % (NAME[target], len(path) - 1, hit))
    plan = (target, path, i)
    wp = path[i]
    dist = np.hypot(wp[0] - x, wp[1] - y)
    if dist < RRT_REACHED:
        return False
    log("%s -> detour via (%.0f, %.0f)" % (why, wp[0], wp[1]))
    step_to(wp, min(RRT_STEP, dist))
    return True

# ---- driving to a landmark -----------------------------------------------------------
def approach(target, lms):
    """Target in the picture: turn to it and drive. True once visited."""
    goal = LANDMARKS[target]
    d, a = lms[target]
    ax, ay = ROBOT_FRONT + d * np.cos(a), d * np.sin(a)
    turn(np.arctan2(ay, ax))                        # axle towards the marker
    d = np.hypot(ax, ay) - ROBOT_FRONT              # camera -> box side
    if d > BLIND_DIST:                              # far: drive to BLIND_DIST and look again
        dist = free_dist(d - BLIND_DIST + 10.0)
        if dist < 10 or drive(dist)[1]:
            avoid(goal)
        return False
    front_gap = d - (SONAR_FRONT - ROBOT_FRONT)     # robot front -> box side
    if front_gap > 70:                              # stop at ~55 cm and measure again
        dist = front_gap - 55
        if free_dist(dist) < dist - 5.0:
            log("remembered obstacle on the way to %s -> evade" % NAME[target])
            avoid(goal)
        else:
            drive(dist, front_stop=VISIT_DIST, sides=False)
        return False
    gap = front_gap
    if front_gap > VISIT_DIST + 5:
        dist = front_gap - VISIT_DIST
        if free_dist(dist) < dist - 5.0:
            log("remembered obstacle on the way to %s -> evade" % NAME[target])
            avoid(goal)
            return False
        driven, _ = drive(dist, front_stop=VISIT_DIST, sides=False)
        expect = front_gap - driven                 # where the box should be now
        s = front_sonar()
        if s < expect - 15:                         # something else in front
            log("%s: echo %.0f cm, box expected at %.0f cm -> evade" % (NAME[target], s, expect))
            avoid(goal)
            return False
        gap = s if abs(s - expect) < 15 else expect # echo only if it fits the camera
    if gap > 38:
        log("%s still %.0f cm away -> look again" % (NAME[target], gap))
        return False
    snap_to(target, gap)
    log("%s visited (camera %.0f cm, now ~%.0f cm)" % (NAME[target], d, gap))
    return True

def sonar_near(target):
    """Pose says the target is too close for the camera: face it and use the sonar."""
    gx, gy = LANDMARKS[target]
    x, y, th = pose()
    turn(wrap(np.arctan2(gy - y, gx - x) - th))
    expected = np.hypot(gx - x, gy - y) - BOX_HALF - SONAR_FRONT
    _, obs = sense()
    s = front_sonar()
    path = s - BLIND_GAP
    if (blocked(obs, expected + 40.0, 0.0) or s < expected - SONAR_MATCH
            or (s < np.inf and free_dist(path) < path - 5.0)):
        log("echo %.0f cm, %s expected at %.0f cm -> something in the way, evade" % (s, NAME[target], expected))
        avoid((gx, gy))
        return False
    if s < 40 and expected < 40 + SONAR_MATCH:      # sonar and pose agree: we are there
        snap_to(target, s)
        log("%s visited on sonar (%.0f cm)" % (NAME[target], s))
        return True
    if abs(s - expected) >= SONAR_MATCH:
        log("sonar %.0f cm, expected %.0f cm -> relocalise" % (s, expected))
        add_noise(80.0, np.radians(30))
        return False
    drive(max(path, 0.0), front_stop=BLIND_SONAR_STOP)
    g = close_in()
    if g is None:
        log("no echo -> %s is not here, relocalise" % NAME[target])
        add_noise(80.0, np.radians(30))
        return False
    snap_to(target, g)
    log("%s visited on sonar (%.0f cm)" % (NAME[target], g))
    return True

def toward_unseen(target):
    """One step on the pose estimate (localised, target not in the picture). True once visited."""
    global plan
    gx, gy = LANDMARKS[target]
    x, y, th = pose()
    gap = np.hypot(gx - x, gy - y) - BOX_HALF - ROBOT_FRONT
    log("pose (%.0f, %.0f, %.0f deg), %s gap %.0f cm" % (x, y, np.degrees(th), NAME[target], gap))
    if gap < NEAR:                                  # right next to it: no detours
        plan = None
    elif step_detour(target, "%s not in view" % NAME[target]):
        return False
    if gap >= SEE_DIST:
        step_to((gx, gy), min(STEP, gap - SEE_DIST + 20))
        return False
    turn(wrap(np.arctan2(gy - y, gx - x) - th))    # should be visible: look around
    if target in scan(target, AHEAD_SCAN):
        return False
    if gap >= NEAR and step_detour(target, "scan found a box"):
        return False
    log("%s not visible -> moving closer" % NAME[target])
    if gap > BLIND_DIST:
        step_to((gx, gy), min(STEP, gap - BLIND_DIST + 10))
        return False
    return sonar_near(target)

def go_to(target):
    """One leg of the route. True once the target is visited."""
    global target_now, plan
    target_now, plan = target, None
    for _ in range(60):
        lms, obs = sense()
        far = {i: m for i, m in lms.items() if m[0] > 150}
        if blocked(obs, OBST_NEAR, 0.0):
            avoid(LANDMARKS[target])
        elif target in lms:
            d_t, a_t = lms[target]
            in_way = blocked(obs, d_t, a_t)
            if localised() and (in_way or d_t > BLIND_DIST) and step_detour(target, "target in view, but blocked"):
                continue
            if in_way:                              # the sonar would take the obstacle for the target
                log("obstacle in front of %s -> evade" % NAME[target])
                avoid(LANDMARKS[target])
            elif approach(target, lms):
                return True
        elif not localised() and far and started:   # one far landmark: drive towards it
            d, a = min(far.values())
            turn(a)
            dist = free_dist(min(100.0, d - 100.0))
            if dist < 10 or drive(dist)[1]:
                avoid(LANDMARKS[target])
        elif not localised():
            relocalise(target)
        elif toward_unseen(target):
            return True
    return False

def start_visit():
    """We start facing L1. Too close for the camera: drive in on the sonar. True if visited."""
    global P, used, target_now
    target_now = L1
    lms, _ = sense()
    if L1 in lms:
        return False
    s, cur = front_sonar(), 0
    for a in (-20, 20):                             # no echo from a box face at an angle: look left/right
        if s <= 120:
            break
        turn(np.radians(a - cur))
        cur = a
        s = front_sonar()
    if s > 120:
        turn(np.radians(-cur))
        return False
    log("L1 too close for the camera, sonar %.0f cm" % s)
    drive(s - 50, front_stop=BLIND_SONAR_STOP)
    gap = close_in()
    if gap is None:
        return False
    if len(P) == 0:
        P, used = ring(L1, gap + SONAR_FRONT - ROBOT_FRONT, 0.0), {L1}
    snap_to(L1, gap)
    log("L1 visited on sonar (%.0f cm)" % gap)
    return True

def look_around():
    """Full turn after a visit: obstacles around us go into the map, landmarks into the filter."""
    log("360 deg look for obstacles")
    cur = 0
    for a in FULL_SCAN:
        turn(np.radians(a - cur))
        cur = a
        sense()

def main():
    missed = []
    for k, target in enumerate(ROUTE):
        if (k == 0 and start_visit()) or go_to(target):
            back_up(BACK_FIRST if k == 0 else BACK_AFTER)   # room to turn without scraping the box
            if k < len(ROUTE) - 1:                  # not after the last one: the run is over
                look_around()
        else:
            log("could not reach %s - continuing" % NAME[target])
            missed.append(NAME[target])
    log("done - " + ("MISSED: " + ", ".join(missed) if missed else "all landmarks visited"))
    log("total time %.0f s" % (time.time() - T0))

if __name__ == "__main__":
    try:
        main()
    finally:
        arlo.stop()
        cam.terminateCaptureThread()