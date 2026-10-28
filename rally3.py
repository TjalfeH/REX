"""
rally.py - DIKU RoboRally 2026 (REX Ex6)

    L2 ---- 400 ---- L4      Route: L1 -> L2 -> L3 -> L4 -> L1
     |                |
    300  obstacles   300
     |                |
    L1 -------------- L3

Particle filter (Ex5) with the 4 corner landmarks for the position.
Target in the camera: drive towards it, the last bit on the front sonar.
Target not in the camera: drive on the pose estimate, with an RRT detour
around the other landmarks and the obstacles we have seen.

Units: cm and radians. Degrees only where a constant says [deg].
Race day: landmark IDs 1-4, measure SONAR_FRONT.
Run with: python3 rally.py | tee run.log
"""
import time
import numpy as np
import camera, robot

# ---- arena [cm] ----------------------------------------------------------------------
BOX_HALF = 20.0                         # half side of a landmark box
INNER_H, INNER_W = 300.0, 400.0         # distance between the inner box sides
L1, L2, L3, L4 = 11, 10, 9, 8           # practice IDs - race day: 1, 2, 3, 4 !
ROBOT_FRONT = 21.0                      # axle -> camera
ROBOT_RADIUS = 28.0                     # robot radius
SONAR_FRONT = 28.0                      # axle -> front sonar (measure!)
CAM_F, MARKER_SIZE = 1315.0, 0.145      # calibration from Ex4

W = INNER_W + 2 * BOX_HALF              # box centre to box centre
H = INNER_H + 2 * BOX_HALF
LANDMARKS = {L1: (0.0, 0.0), L2: (0.0, H), L3: (W, 0.0), L4: (W, H)}
ROUTE = [L1, L2, L3, L4, L1]
NAME = {L1: "L1", L2: "L2", L3: "L3", L4: "L4"}

# ---- final approach ------------------------------------------------------------------
VISIT_DIST = 25.0                       # front sonar -> box side at the end (< 40 cm rule)
SONAR_MATCH = 30.0                      # sonar and pose must agree this well near the target
BRAKE = 9.0                             # the robot rolls this far after a sonar stop
BLIND_DIST = 130.0                      # closer than this we stop checking the camera
BLIND_STOP = 35.0                       # camera -> marker distance where the blind drive ends
BLIND_GAP = BLIND_STOP - (SONAR_FRONT - ROBOT_FRONT)   # sonar gap we expect after it
BLIND_SONAR_STOP = 33.0                 # sonar stop during the blind drive
ECHO_RANGE = BLIND_GAP + 25.0           # an echo further away than this is not our box
SEE_DIST = 160.0                        # from here the camera should see the target
NEAR = 60.0                             # closer than this: no detours any more, go for it

# ---- obstacles and paths -------------------------------------------------------------
KEEP_AWAY = 100.0                       # path -> other landmark centre (40 cm rule + robot + error)
OBST_HALF = 12.0                        # marker -> obstacle centre
OBST_RADIUS = 17.5                      # obstacle radius
CLEAR = OBST_RADIUS + ROBOT_RADIUS + 15.0       # driving: min distance to an obstacle centre
                                        # (= 15 cm between robot edge and box wall)
OBST_KEEP = CLEAR                       # RRT: path -> obstacle centre (never below CLEAR, else
                                        # free_dist refuses the paths the RRT plans)
LM_EXTRA = KEEP_AWAY - CLEAR            # landmarks get this much more room than obstacles
OBST_NEAR = 80.0                        # obstacle closer than this straight ahead -> evade
OBST_MAX_SEEN = 250.0                   # obstacles seen further away are not remembered
STEP = 140.0                            # one step when the target is not in view
EVADE_TURN, EVADE_DIST, EVADE_BACK = 60.0, 35.0, 25.0   # avoid(): turn [deg], forward, back
BACK_KEEP = 20.0                        # room we keep behind us when reversing

# ---- RRT -----------------------------------------------------------------------------
RRT_STEP = 50.0                         # tree step and drive step on a detour
RRT_ITER = 3000                         # max nodes per try
RRT_REACHED = 15.0                      # this close to a path point = reached
RRT_TRIES = 8                           # plan this often, take the shortest path
RRT_PAD = 200.0                         # sample box around start and goal (doubled if no path)
ARENA_OUT = 120.0                       # the robot may be this far outside the landmark rectangle
DETOUR_RANGE = 160.0                    # only detour if the line is blocked closer than this

# ---- sonar ---------------------------------------------------------------------------
SONAR_STOP = 45.0                       # normal front stop
SIDE_STOP = 12.0                        # side emergency stop
SIDE_FREE = 60.0                        # both sides this free -> evade towards the goal

# ---- motion and camera ---------------------------------------------------------------
SPEED, CM_PER_SEC, DEG_PER_SEC = 64, 46.0, 125.0
TRIM = 2                                # right wheel a bit faster -> drives straight
T_START = 0.15                          # start-up time of a drive [s]
SETTLE = 0.3                            # stand still this long before a camera picture [s]
STOP_PAUSE = 0.1                        # pause after every move [s]
FLUSH_FRAMES = 2                        # old frames thrown away before a picture

# ---- particle filter -----------------------------------------------------------------
N = 500                                 # particles
SIGMA_A = np.radians(10.0)              # bearing noise
LOC_STD = 50.0                          # localised if the particle std in x and y is below this

# scan angles [deg], relative to the heading when the scan starts
FULL_SCAN = [0, -55, -110, -165, -220, -275, -330]   # camera sees 64 deg
AHEAD_SCAN = [0, -45, 45]
SONAR_SWEEP = [0, -15, -30, 15, 30]

# ---- hardware and state --------------------------------------------------------------
arlo = robot.Robot()
cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
cam.intrinsic_matrix = np.array([[CAM_F, 0, 820.0], [0, CAM_F, 616.0], [0, 0, 1]], dtype=np.float64)
cam.arucoMarkerLength = MARKER_SIZE

P = np.empty((0, 3))                    # particles: one row (x, y, theta) each
obstacles = {}                          # id -> (x, y, weight)
last_view = None                        # result of the last sense(), None after a move
lost_count = 0                          # updates in a row where no particle fit
used = set()                            # landmarks the particles are based on
target_now = None                       # landmark we are driving to
plan = None                             # (target, path, i): RRT path we follow, path[i] is next
started = False                         # L1 visited at the start?
T0 = time.time()


def log(*args):
    print("[%6.1f]" % (time.time() - T0), *args, flush=True)


def wrap(a):
    """Angle into -pi..pi."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def cam_pos(x, y, th):
    """Camera position for a robot at (x, y, th)."""
    return x + ROBOT_FRONT * np.cos(th), y + ROBOT_FRONT * np.sin(th)


def along_across(x, y, th, px, py):
    """Point (px, py) seen from (x, y) looking along th: distance ahead and |distance to the side|."""
    dx, dy = px - x, py - y
    c, s = np.cos(th), np.sin(th)
    return dx * c + dy * s, abs(dy * c - dx * s)


def sonar(side="front"):
    """One sonar reading in cm, inf if no echo."""
    mm = getattr(arlo, "read_%s_ping_sensor" % side)()
    if mm > 0:
        return mm / 10.0
    return np.inf


def front_sonar():
    """Median of 3 front readings."""
    readings = [sonar(), sonar(), sonar()]
    return sorted(readings)[1]


# ---- motion --------------------------------------------------------------------------
def add_noise(s_xy, s_th):
    global P
    P = P + np.random.normal(0, [s_xy, s_xy, s_th], P.shape)
    P[:, 2] = wrap(P[:, 2])


def move_particles(d=0.0, a=0.0):
    """Motion model: all particles drive d cm or turn a rad, plus some noise."""
    global last_view
    last_view = None                    # the last picture is outdated now
    P[:, 0] += d * np.cos(P[:, 2])
    P[:, 1] += d * np.sin(P[:, 2])
    P[:, 2] += a
    add_noise(2.0 + 0.05 * abs(d), np.radians(3) + 0.1 * abs(a))


def turn(angle):
    """Turn on the spot, angle in rad, positive = left."""
    deg = np.degrees(angle)
    if abs(deg) < 2:
        return
    if deg > 0:
        arlo.go_diff(SPEED, SPEED, 0, 1)
    else:
        arlo.go_diff(SPEED, SPEED, 1, 0)
    time.sleep(abs(deg) / DEG_PER_SEC)
    arlo.stop()
    time.sleep(STOP_PAUSE)
    move_particles(a=angle)


def straight(dist, stop):
    """Drive dist cm straight (negative = backwards) until done or stop() is True.
    Returns (driven cm, aborted)."""
    fwd = int(dist > 0)
    arlo.go_diff(SPEED, SPEED + TRIM * fwd, fwd, fwd)
    t0 = time.time()
    aborted = False
    while time.time() - t0 < abs(dist) / CM_PER_SEC + T_START:
        if stop():
            aborted = True
            break
        time.sleep(0.05)
    elapsed = time.time() - t0
    arlo.stop()
    time.sleep(STOP_PAUSE)
    driven = min(abs(dist), max(0.0, elapsed - T_START) * CM_PER_SEC)
    if fwd:
        move_particles(driven)
    else:
        move_particles(-driven)
    return driven, aborted


def drive(dist, front_stop=SONAR_STOP, sides=True):
    """Drive forward dist cm. Stops if the front sonar sees something closer than front_stop
    or (if sides) a side sonar closer than SIDE_STOP. Returns (driven cm, aborted)."""
    if dist <= 0:
        return 0.0, False
    check = ["left", "right"]           # one side per loop so the front check stays fast

    def stop():
        if sonar() < front_stop:
            return True
        if sides:
            name = check.pop(0)
            check.append(name)
            s = sonar(name)
            if s < SIDE_STOP:
                log("%s sonar %.0f cm -> emergency stop" % (name, s))
                return True
        return False
    return straight(dist, stop)


def back_up(dist):
    """Reverse up to dist cm, keeping BACK_KEEP to whatever is behind us."""
    b = sonar("back")
    dist = min(dist, b - BACK_KEEP)
    if dist < 10:
        log("no room behind us (back sonar %.0f cm) -> not backing up" % b)
        return
    log("backing up %.0f cm (back sonar %.0f cm)" % (dist, b))
    straight(-dist, lambda: sonar("back") < BACK_KEEP)


def avoid(goal=None):
    """Turn EVADE_TURN to the freer side and drive EVADE_DIST. Both sides free and we know where
    we are: turn towards the goal. Not free: try the other side. Neither free: back up."""
    l, r = sonar("left"), sonar("right")
    if l >= r:
        a = EVADE_TURN
    else:
        a = -EVADE_TURN
    if goal is not None and min(l, r) > SIDE_FREE and localised():
        x, y, th = pose()
        if wrap(np.arctan2(goal[1] - y, goal[0] - x) - th) > 0:
            a = EVADE_TURN
        else:
            a = -EVADE_TURN
    for ang in (a, -2 * a):
        turn(np.radians(ang))
        _, obs = sense()
        if not blocked(obs, EVADE_DIST + ROBOT_RADIUS, 0.0) and free_dist(EVADE_DIST) >= EVADE_DIST:
            drive(EVADE_DIST)
            return
    turn(np.radians(a))                 # back to where we looked before
    back_up(EVADE_BACK)


def free_dist(dist):
    """How far we can drive straight ahead before we get closer than CLEAR to a remembered obstacle
    or closer than CLEAR + LM_EXTRA to a landmark that is not the target.
    Not localised, but all particles sit on a ring around one landmark: then we still know where
    that landmark is relative to us, so keep away from it."""
    if len(P) == 0:
        return dist
    if localised():
        x, y, th = pose()
        things = []
        for ox, oy, _ in obstacles.values():
            things.append((ox, oy, 0.0))
        for i, (lx, ly) in LANDMARKS.items():
            if i != target_now:
                things.append((lx, ly, LM_EXTRA))
    elif len(used) == 1 and target_now not in used:
        x, y, th = P[0]
        lx, ly = LANDMARKS[list(used)[0]]
        things = [(lx, ly, LM_EXTRA)]
    else:
        return dist
    for ox, oy, extra in things:
        ahead, side = along_across(x, y, th, ox, oy)
        # If we are already inside the circle (after an evasion for example) we may still drive,
        # but get at most 10 cm closer and never closer than 5 cm to touching.
        d_now = np.hypot(ox - x, oy - y)
        keep = min(CLEAR + extra, max(d_now - 10.0, OBST_RADIUS + ROBOT_RADIUS + 5.0 + extra))
        if ahead > 0 and side < keep:
            dist = min(dist, ahead - np.sqrt(keep ** 2 - side ** 2))
    return dist


def blocked(obs, d_t, a_t):
    """True if an obstacle in the picture is closer than CLEAR to the line to the point at
    distance d_t and bearing a_t."""
    for d, a in obs.values():
        r = d + OBST_HALF
        along, across = along_across(0.0, 0.0, a_t, r * np.cos(a), r * np.sin(a))
        if along < d_t and across < CLEAR:
            return True
    return False


def step_to(point, dist):
    """Turn towards a world point, take a picture, then drive dist cm (as far as it is free).
    If we have to evade, the RRT plan is thrown away: plan again from where we end up."""
    global plan
    x, y, th = pose()
    turn(wrap(np.arctan2(point[1] - y, point[0] - x) - th))
    _, obs = sense()
    dist = free_dist(dist)
    if dist < 10 or blocked(obs, dist + ROBOT_RADIUS, 0.0):
        log("path not free -> evade")
        avoid(point)
        plan = None
        return
    _, aborted = drive(dist)
    if aborted:
        avoid(point)
        plan = None


def face_marker(d, a):
    """Turn so the axle points at a marker seen at distance d and bearing a from the camera.
    Returns the distance axle -> marker."""
    ax = ROBOT_FRONT + d * np.cos(a)
    ay = d * np.sin(a)
    turn(np.arctan2(ay, ax))
    return np.hypot(ax, ay)


def find_box_echo(max_cm):
    """Sonar sweep +-30 deg. Turns to the closest echo below max_cm and returns it, else None."""
    best, best_s, cur = None, max_cm, 0
    for a in SONAR_SWEEP:
        turn(np.radians(a - cur))
        cur = a
        s = front_sonar()
        if s < best_s:
            best, best_s = a, s
    if best is None:
        turn(np.radians(-cur))
        return None
    turn(np.radians(best - cur))
    return best_s


def close_in():
    """Creep forward on the front sonar until VISIT_DIST. Returns the final gap, None if no echo."""
    s = front_sonar()
    if s < ECHO_RANGE:
        gap = s
    else:
        log("front sonar sees no box (diagonal?) -> sonar sweep")
        gap = find_box_echo(ECHO_RANGE)
    if gap is None:
        return None
    for _ in range(2):                  # second round only after a sonar glitch
        if gap <= VISIT_DIST + BRAKE:
            break
        drive(gap - VISIT_DIST + 3.0, front_stop=VISIT_DIST + BRAKE, sides=False)
        s = front_sonar()
        if s >= ECHO_RANGE:             # echo gone: we are there
            return VISIT_DIST
        gap = s
    return gap


# ---- particle filter -----------------------------------------------------------------
def ring(lm_id, d, a, n=N):
    """n particles that all see landmark lm_id at distance d (camera -> box side) and bearing a:
    a ring around the landmark."""
    lx, ly = LANDMARKS[lm_id]
    phi = np.random.uniform(-np.pi, np.pi, n)
    r = d + BOX_HALF + np.random.normal(0, 10, n)
    ths = wrap(phi + np.pi - a + np.random.normal(0, np.radians(5), n))
    cx = lx + r * np.cos(phi)
    cy = ly + r * np.sin(phi)
    return np.column_stack([cx - ROBOT_FRONT * np.cos(ths), cy - ROBOT_FRONT * np.sin(ths), ths])


def filter_update(meas, check_lost=True):
    """Weight the particles with the landmark measurements (gaussian), then resample.
    First landmark ever: start with a ring around it. No particle fits twice in a row: we are
    lost, start again with a ring around the closest landmark."""
    global P, lost_count, used
    if not meas:
        return
    closest = min(meas, key=lambda i: meas[i][0])
    if len(P) == 0:
        log("first landmark %s -> particles on a ring around it" % NAME[closest])
        P = ring(closest, *meas[closest])
        used = {closest}
        check_lost = False
    xs, ys, ths = P.T
    cx, cy = cam_pos(xs, ys, ths)
    logw = np.zeros(len(P))
    for lm_id, (d, a) in meas.items():
        lx, ly = LANDMARKS[lm_id]
        ed = np.hypot(lx - cx, ly - cy) - BOX_HALF          # expected distance
        ea = wrap(np.arctan2(ly - cy, lx - cx) - ths)       # expected bearing
        sd = 10.0 + 0.08 * d                                # far markers are less exact
        logw += -0.5 * ((d - ed) / sd) ** 2 - 0.5 * (wrap(a - ea) / SIGMA_A) ** 2
    if check_lost and logw.max() < -8.0 * len(meas):
        lost_count += 1
        if lost_count < 2:
            log("measurement fits no particle -> ignored once")
            return
        lost_count = 0
        log("LOST -> new ring around %s" % NAME[closest])
        P = ring(closest, *meas[closest])
        used = {closest}
        filter_update(meas, check_lost=False)
        return
    lost_count = 0
    used = used | set(meas)
    w = np.exp(logw - logw.max())
    w = w / w.sum()
    cum = np.cumsum(w)
    cum[-1] = 1.0
    picks = (np.random.uniform() + np.arange(len(w))) / len(w)   # systematic resampling
    P = P[np.searchsorted(cum, picks)]


def pose():
    """Mean of the particles: (x, y, theta)."""
    th = np.arctan2(np.sin(P[:, 2]).mean(), np.cos(P[:, 2]).mean())
    return P[:, 0].mean(), P[:, 1].mean(), th


def spread():
    """Std of x, std of y and how well the headings agree (0..1)."""
    agree = np.hypot(np.cos(P[:, 2]).mean(), np.sin(P[:, 2]).mean())
    return P[:, 0].std(), P[:, 1].std(), agree


def localised():
    """The particles agree on position and heading, based on at least two landmarks
    (one landmark only gives a ring)."""
    if len(P) == 0 or len(used) < 2:
        return False
    sx, sy, agree = spread()
    return sx < LOC_STD and sy < LOC_STD and agree > 0.9


def snap_to(lm_id, gap):
    """We stand in front of lm_id with the sonar reading gap: put all particles there."""
    global P, last_view, started
    started = True
    ths = wrap(P[:, 2] + np.random.normal(0, np.radians(3), len(P)))
    r = gap + BOX_HALF + SONAR_FRONT
    lx, ly = LANDMARKS[lm_id]
    P = np.column_stack([lx - r * np.cos(ths), ly - r * np.sin(ths), ths])
    last_view = None


def remember(obs):
    """Store the world position of seen obstacles (only when localised).
    Closer sightings count more (weight 1/distance)."""
    if not obs or not localised():
        return
    x, y, th = pose()
    cx, cy = cam_pos(x, y, th)
    for i, (d, a) in obs.items():
        if d > OBST_MAX_SEEN:
            continue
        r = d + OBST_HALF
        ox = cx + r * np.cos(th + a)
        oy = cy + r * np.sin(th + a)
        w = 1.0 / d
        if not (-60 < ox < W + 60 and -60 < oy < H + 60):    # outside the arena: wrong reading
            continue
        if i in obstacles:
            px, py, pw = obstacles[i]
            ox = (pw * px + w * ox) / (pw + w)
            oy = (pw * py + w * oy) / (pw + w)
            w = pw + w
        else:
            log("obstacle %d at (%.0f, %.0f)" % (i, ox, oy))
        obstacles[i] = (ox, oy, w)


def sense():
    """Take a picture, update the filter and the obstacle map.
    Returns (landmarks, obstacles), each {id: (distance camera -> marker, bearing)}.
    If we did not move since the last picture, the same result is returned again."""
    global last_view
    if last_view is not None:
        return last_view
    time.sleep(SETTLE - STOP_PAUSE)     # the camera needs a still robot
    raw = {}
    try:
        for _ in range(FLUSH_FRAMES):
            cam.get_next_frame()
        frame = cam.get_next_frame()
        if frame is None:
            ids = None
        else:
            ids, ds, angs = cam.detect_aruco_objects(frame)
    except Exception as e:              # one broken frame must not end the run
        log("camera error: %s -> nothing seen" % e)
        ids = None
    if ids is not None:
        for i, d, a in zip(ids, ds, angs):
            if 0 < d < 500:
                raw.setdefault(int(i), []).append((float(d), float(a)))
    seen = {}
    for i, v in raw.items():            # several sides of one box: take the mean
        seen[i] = tuple(np.mean(v, axis=0))
    log("seen:", {i: (round(d), round(np.degrees(a))) for i, (d, a) in seen.items()})
    lms = {i: m for i, m in seen.items() if i in LANDMARKS}
    obs = {i: m for i, m in seen.items() if i not in LANDMARKS}
    filter_update(lms)
    remember(obs)
    last_view = (lms, obs)
    return last_view


# ---- scanning ------------------------------------------------------------------------
def scan(target, angles=FULL_SCAN, until_localised=False):
    """Take pictures at the given angles. Stops early when the target is seen (or, with
    until_localised, once we are localised). Returns the landmarks of the last picture."""
    cur = 0
    for a in angles:
        turn(np.radians(a - cur))
        cur = a
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
        log("not localised yet -> move a bit and scan again")
        turn(np.radians(120))
        drive(60.0)
    return {}


# ---- RRT -----------------------------------------------------------------------------
def zones(x, y, gx, gy, target):
    """Circles (cx, cy, radius) the path must stay out of: the other landmarks and the
    remembered obstacles. A circle we (or the goal) are already inside of is shrunk to that
    distance, so we may drive out of it but not further in. Otherwise the start would always be
    in collision and no plan could be found."""
    things = []
    for i, (lx, ly) in LANDMARKS.items():
        if i != target:
            things.append((lx, ly, KEEP_AWAY))
    for ox, oy, _ in obstacles.values():
        things.append((ox, oy, OBST_KEEP))
    out = []
    for cx, cy, keep in things:
        keep = min(keep, np.hypot(x - cx, y - cy), np.hypot(gx - cx, gy - cy))
        out.append((cx, cy, keep))
    return out


def first_hit(p, q, things):
    """Distance along the line p -> q to the first point inside a circle (checked every 5 cm).
    None if the line is free."""
    length = np.hypot(q[0] - p[0], q[1] - p[1])
    for t in np.linspace(0, 1, int(length / 5.0) + 2):
        px = p[0] + t * (q[0] - p[0])
        py = p[1] + t * (q[1] - p[1])
        for cx, cy, keep in things:
            if np.hypot(px - cx, py - cy) < keep:
                return t * length
    return None


def free(p, q, things):
    """True if the line p -> q stays out of all circles."""
    return first_hit(p, q, things) is None


def rrt_once(start, goal, things, pad):
    """Plain RRT: grow a tree from start with random samples (10 % of them the goal itself)
    until a node sees the goal. Samples come from the box around start and goal plus pad,
    at most ARENA_OUT outside the arena. Returns the path start -> goal or None."""
    nodes = [start]
    parent = [None]
    lo = np.minimum(start, goal) - pad
    hi = np.maximum(start, goal) + pad
    out = ARENA_OUT - ROBOT_RADIUS
    lo = np.maximum(lo, (-out, -out))
    hi = np.minimum(hi, (W + out, H + out))
    lo = np.minimum(lo, start)          # we may already be outside: then we can drive back in
    hi = np.maximum(hi, start)
    xy = np.empty((RRT_ITER + 1, 2))    # the nodes as an array, for a fast nearest-node search
    xy[0] = start
    for _ in range(RRT_ITER):
        if np.random.rand() < 0.1:
            sample = goal
        else:
            sample = tuple(np.random.uniform(lo, hi))
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
        if free(new, goal, things):     # goal in sight: walk the tree back to the start
            path = [goal]
            j = len(nodes) - 1
            while j is not None:
                path.append(nodes[j])
                j = parent[j]
            path.reverse()
            return path
    return None


def shortcut(path, things):
    """Remove zigzag: from each point jump to the furthest later point we can reach in a
    straight free line."""
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not free(path[i], path[j], things):
            j -= 1
        out.append(path[j])
        i = j
    return out


def split(path):
    """Cut every segment into equal pieces of at most RRT_STEP (one piece = one drive step)."""
    out = [path[0]]
    for p, q in zip(path, path[1:]):
        n = int(np.ceil(np.hypot(q[0] - p[0], q[1] - p[1]) / RRT_STEP))
        for k in range(1, n + 1):
            out.append((p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n))
    return out


def path_length(path):
    total = 0.0
    for p, q in zip(path, path[1:]):
        total += np.hypot(q[0] - p[0], q[1] - p[1])
    return total


def rrt(start, goal, things):
    """Shortest of RRT_TRIES shortcut RRT paths, cut into RRT_STEP pieces. None if no path.
    No path at all: try again with a bigger sample box."""
    for pad in (RRT_PAD, 2 * RRT_PAD):
        paths = []
        for _ in range(RRT_TRIES):
            p = rrt_once(start, goal, things, pad)
            if p is not None:
                paths.append(shortcut(p, things))
        if paths:
            return split(min(paths, key=path_length))
    return None


def next_point(x, y, path, i, things):
    """Index of the path point to drive to next: i, or the one after it once i is reached.
    None if the way there or the rest of the path is not free any more."""
    while i < len(path) - 1 and np.hypot(path[i][0] - x, path[i][1] - y) < RRT_REACHED:
        i += 1
    if not free((x, y), path[i], things):
        return None
    for k in range(i, len(path) - 1):
        if not free(path[k], path[k + 1], things):
            return None
    return i


def detour(x, y, gx, gy, target):
    """Next waypoint of an RRT detour around the other landmarks and the remembered obstacles.
    None means: drive straight (line is free, or only blocked further than DETOUR_RANGE away,
    or no path found). An existing plan is followed point by point as long as it stays free,
    so we do not zigzag between random paths."""
    global plan
    things = zones(x, y, gx, gy, target)
    hit = first_hit((x, y), (gx, gy), things)
    if hit is None or hit > DETOUR_RANGE:
        plan = None
        return None
    i = None
    if plan is not None and plan[0] == target:
        path = plan[1]
        i = next_point(x, y, path, plan[2], things)
    if i is None:                       # no plan, or the old one is not free any more
        path = rrt((x, y), (gx, gy), things)
        if path is not None:
            i = next_point(x, y, path, 1, things)
        if i is None:
            plan = None
            log("RRT found no path -> driving straight")
            return None
        log("RRT: new path to %s (%d steps), blocked %.0f cm ahead" % (NAME[target], len(path) - 1, hit))
    plan = (target, path, i)
    return path[i]


def step_detour(target, why):
    """One step along a detour if the straight line to the target needs one. True if we stepped."""
    x, y, _ = pose()
    wp = detour(x, y, *LANDMARKS[target], target)
    if wp is None:
        return False
    dist = np.hypot(wp[0] - x, wp[1] - y)
    if dist < RRT_REACHED:              # only the goal is left and we are already there
        return False
    log("%s -> detour via (%.0f, %.0f)" % (why, wp[0], wp[1]))
    step_to(wp, min(RRT_STEP, dist))
    return True


# ---- driving to a landmark -----------------------------------------------------------
def approach(target, lms):
    """Target in the picture: turn to it and drive. Far away: drive straight to BLIND_DIST and
    look again. Close: drive the rest on the camera distance. True once visited."""
    goal = LANDMARKS[target]
    d = face_marker(*lms[target]) - ROBOT_FRONT         # camera -> box side, straight ahead
    if d > BLIND_DIST:
        dist = free_dist(d - BLIND_DIST + 10.0)
        if dist < 10:
            avoid(goal)
        else:
            _, aborted = drive(dist)
            if aborted:
                avoid(goal)
        return False
    # Close: drive on the camera distance. The sonar only stops us - at an angle it often gets no
    # echo from the box, so we never wait for one.
    front_gap = d - (SONAR_FRONT - ROBOT_FRONT)         # front of the robot -> box side
    if front_gap > 70:                                  # long way: stop at ~55 cm and measure again
        dist = front_gap - 55
        if free_dist(dist) < dist - 5.0:
            log("remembered obstacle next to the path to %s -> evade" % NAME[target])
            avoid(goal)
        else:
            drive(dist, front_stop=VISIT_DIST, sides=False)
        return False
    gap = front_gap
    if front_gap > VISIT_DIST + 5:
        dist = front_gap - VISIT_DIST
        if free_dist(dist) < dist - 5.0:
            log("remembered obstacle next to the path to %s -> evade" % NAME[target])
            avoid(goal)
            return False
        driven, _ = drive(dist, front_stop=VISIT_DIST, sides=False)
        expect = front_gap - driven                     # where the box should be now
        s = front_sonar()
        if s < expect - 15:                             # something else is in front of us
            log("%s: echo %.0f cm, box expected at %.0f cm -> obstacle in front, evade" % (NAME[target], s, expect))
            avoid(goal)
            return False
        if abs(s - expect) < 15:                        # use the echo only if it fits the camera
            gap = s
        else:
            gap = expect
    if gap > 38:                        # not within 40 cm yet: look again
        log("%s still %.0f cm away -> look again" % (NAME[target], gap))
        return False
    snap_to(target, gap)
    log("%s visited (camera said %.0f cm, now ~%.0f cm)" % (NAME[target], d, gap))
    back_up(30.0)                       # room to turn without scraping the box
    return True


def sonar_near(target):
    """The pose says the target is too close for the camera. Face it: if the echo fits the
    expected gap, drive in and visit. Echo much closer: something is in the way, evade.
    Otherwise our pose is wrong: relocalise."""
    gx, gy = LANDMARKS[target]
    x, y, th = pose()
    turn(wrap(np.arctan2(gy - y, gx - x) - th))
    expected = np.hypot(gx - x, gy - y) - BOX_HALF - SONAR_FRONT     # sonar -> box side
    _, obs = sense()
    s = front_sonar()
    path = s - BLIND_GAP
    in_way = blocked(obs, expected + 40.0, 0.0) or s < expected - SONAR_MATCH
    if s < np.inf and free_dist(path) < path - 5.0:
        in_way = True
    if in_way:
        log("echo %.0f cm, %s expected at %.0f cm, obstacles in view %s -> something in the way, evade"
            % (s, NAME[target], expected, list(obs)))
        avoid((gx, gy))
        return False
    if s < 40 and expected < 40 + SONAR_MATCH:          # sonar and pose agree: we are there
        snap_to(target, s)
        log("%s visited on sonar (%.0f cm, pose said %.0f cm)" % (NAME[target], s, expected))
        back_up(30.0)
        return True
    if abs(s - expected) >= SONAR_MATCH:
        log("sonar %.0f cm, expected %.0f cm -> relocalise" % (s, expected))
        add_noise(80.0, np.radians(30))
        return False
    drive(max(path, 0.0), front_stop=BLIND_SONAR_STOP)
    g = close_in()
    if g is None:                       # no echo and no camera proof: not visited
        log("no echo after driving -> %s is not here, relocalise" % NAME[target])
        add_noise(80.0, np.radians(30))
        return False
    snap_to(target, g)
    log("%s visited on sonar (expected %.0f cm, sonar said %.0f cm)" % (NAME[target], expected, s))
    back_up(30.0)
    return True


def toward_seen(target, lms, obs):
    """One step with the target in the picture. True once visited."""
    d_t, a_t = lms[target]
    in_way = blocked(obs, d_t, a_t)
    if localised() and (in_way or d_t > BLIND_DIST):
        if step_detour(target, "target in view, but blocked"):
            return False
    if in_way:                          # the sonar would take the obstacle for the target
        log("obstacle in front of %s -> evade" % NAME[target])
        avoid(LANDMARKS[target])
        return False
    return approach(target, lms)


def toward_unseen(target):
    """One step on the pose estimate (we are localised, target not in the picture).
    True once visited."""
    global plan
    gx, gy = LANDMARKS[target]
    x, y, th = pose()
    gap = np.hypot(gx - x, gy - y) - BOX_HALF - ROBOT_FRONT
    sx, sy, agree = spread()
    log("pose x=%.0f y=%.0f th=%.0f deg | spread %.0f/%.0f cm, heading %.2f | %s gap %.0f cm"
        % (x, y, np.degrees(th), sx, sy, agree, NAME[target], gap))
    if gap < NEAR:                      # right next to it: a detour would only lead around the box
        plan = None
    elif step_detour(target, "%s not in view" % NAME[target]):
        return False
    if gap >= SEE_DIST:
        step_to((gx, gy), min(STEP, gap - SEE_DIST + 20))
        return False
    turn(wrap(np.arctan2(gy - y, gx - x) - th))    # we should see it from here: look around
    if target in scan(target, AHEAD_SCAN):
        return False
    if gap >= NEAR and step_detour(target, "scan found a box"):   # the scan may have found the box hiding it
        return False
    log("%s not visible -> moving closer" % NAME[target])
    if gap > BLIND_DIST:
        step_to((gx, gy), min(STEP, gap - BLIND_DIST + 10))
        return False
    return sonar_near(target)


def go_to(target):
    """One leg of the route. True once the target is visited."""
    global target_now, plan
    target_now = target
    plan = None
    for _ in range(60):
        lms, obs = sense()
        far = {i: m for i, m in lms.items() if m[0] > 150}
        if blocked(obs, OBST_NEAR, 0.0):
            avoid(LANDMARKS[target])
        elif target in lms:
            if toward_seen(target, lms, obs):
                return True
        elif not localised() and far and started:
            # one far landmark is not enough to localise: drive towards it, near the corner
            # more landmarks come into view
            d, a = min(far.values())
            turn(a)
            dist = free_dist(min(100.0, d - 100.0))
            if dist < 10:
                avoid(LANDMARKS[target])
            else:
                _, aborted = drive(dist)
                if aborted:
                    avoid(LANDMARKS[target])
        elif not localised():
            relocalise(target)
        elif toward_unseen(target):
            return True
    return False


def start_visit():
    """We start facing L1. If it is too close for the camera, drive in on the front sonar.
    True if L1 was visited this way."""
    global P, used, target_now
    target_now = L1
    lms, _ = sense()
    if L1 in lms:
        return False                    # camera sees it: go_to does the rest
    s = front_sonar()
    cur = 0
    for a in (-20, 20):                 # a box face at an angle gives no echo: look left and right
        if s <= 120:
            break
        turn(np.radians(a - cur))
        cur = a
        s = front_sonar()
    if s > 120:
        turn(np.radians(-cur))
        return False
    log("L1 too close for the camera, sonar %.0f cm -> drive in on the sonar" % s)
    drive(s - 50, front_stop=BLIND_SONAR_STOP)
    gap = close_in()
    if gap is None:
        return False
    if len(P) == 0:                     # nothing seen yet: ring around L1, a scan decides the side
        P = ring(L1, gap + SONAR_FRONT - ROBOT_FRONT, 0.0)
        used = {L1}
    snap_to(L1, gap)
    log("L1 visited on sonar (%.0f cm)" % gap)
    back_up(30.0)
    return True


def main():
    log("map:", {NAME[i]: p for i, p in LANDMARKS.items()})
    missed = []
    for k, target in enumerate(ROUTE):
        if k == 0 and start_visit():
            continue
        if not go_to(target):           # keep going, the later landmarks still count
            log("could not reach %s - continuing" % NAME[target])
            missed.append(NAME[target])
    if missed:
        log("done - MISSED: " + ", ".join(missed))
    else:
        log("done - all landmarks visited")
    log("total time %.0f s" % (time.time() - T0))


if __name__ == "__main__":
    try:
        main()
    finally:
        arlo.stop()
        cam.terminateCaptureThread()