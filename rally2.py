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

Units: cm and rad everywhere in the code; deg only where a constant says [deg].

Race day: landmark IDs 1-4, measure ROBOT_FRONT, SONAR_FRONT and FAST_CM_PER_SEC,
then `python3 rally.py | tee run.log`.
"""
import itertools
import time
import numpy as np
import camera, robot

# ---- constants [cm] ----------------------------------------------------------------
BOX_HALF = 20.0                         # half landmark box side
INNER_H, INNER_W = 300.0, 400.0         # distances between INNER box sides
L1, L2, L3, L4 = 11, 10, 9, 8            # PRACTICE IDs. Race day: 1, 2, 3, 4 !
ROBOT_FRONT = 21.0                      # axle -> camera
ROBOT_RADIUS = 28.0                     # robot footprint radius
SONAR_FRONT = 28.0                      # axle -> front sonar - MEASURE
CAM_F, MARKER_SIZE = 1315.0, 0.145      # Ex4 calibration (camera.py defaults are 1.33x too long)

_W, _H = INNER_W + 2 * BOX_HALF, INNER_H + 2 * BOX_HALF
LANDMARKS = {L1: (0.0, 0.0), L2: (0.0, _H), L3: (_W, 0.0), L4: (_W, _H)}
ROUTE = [L1, L2, L3, L4, L1]
NAME = {L1: "L1", L2: "L2", L3: "L3", L4: "L4"}

# final approach
VISIT_DIST = 15.0                       # final gap front sonar -> box side (< 40 rule)
SONAR_MATCH = 30.0                      # camera blind near target: |sonar - expected gap| below this
BRAKE = 9.0                             # roll-out after sonar stop (test run: 18 -> ~9 cm)
BLIND_DIST = 130.0                      # closer than this: no more camera checks
BLIND_STOP = 35.0                       # blind stretch ends at this camera -> marker distance
BLIND_GAP = BLIND_STOP - (SONAR_FRONT - ROBOT_FRONT)   # expected sonar gap after it
BLIND_SONAR_STOP = 33.0                 # front sonar abort on the blind stretch
ECHO_RANGE = BLIND_GAP + 25.0           # close_in: an echo further away is not our box
SEE_DIST = 160.0                        # camera should see the target from here

# obstacles and paths
KEEP_AWAY = 100.0                       # path -> non-target landmark centre
OBST_HALF = 12.0                        # marker face -> obstacle centre
OBST_RADIUS = 17.5                      # obstacle radius
OBST_KEEP = OBST_RADIUS + ROBOT_RADIUS + 30.0   # detour: path -> obstacle centre
CLEAR = OBST_RADIUS + ROBOT_RADIUS + 15.0       # min path -> obstacle centre while driving
OBST_NEAR = 80.0                        # seen obstacle closer than this straight ahead -> evade
OBST_MAX_SEEN = 250.0                   # farther obstacle sightings are too inaccurate to remember
STEP = 140.0                            # step when target not in view
EVADE_TURN, EVADE_DIST, EVADE_BACK = 60.0, 35.0, 25.0   # avoid(): turn [deg], forward, backward
BACK_KEEP = 20.0                        # room kept behind us when reversing

# sonar
SONAR_STOP = 45.0                       # normal front abort
SIDE_STOP = 12.0                        # side emergency stop while driving
SIDE_FREE = 60.0                        # both sides freer than this -> evade towards the goal

# motion and camera timing
SPEED, CM_PER_SEC, DEG_PER_SEC = 64, 46.0, 125.0
TRIM = 2                                # right wheel faster by this -> drives straight (forward)
FAST, FAST_CM_PER_SEC = 90, 64.0        # long free drives only - MEASURE (64.0 is a guess: 46 * 90/64)
T_START = 0.15                          # start-up time of a drive [s]
SETTLE = 0.3                            # robot standing still before a camera frame [s]
STOP_PAUSE = 0.1                        # pause after every move; the rest of SETTLE only before a look [s]
FLUSH_FRAMES = 2                        # old frames (camera buffer) thrown away before measuring

# particle filter
N = 500                                 # particles
SIGMA_A = np.radians(10.0)              # bearing noise
LOC_STD = 50.0                          # localised: particle std in x and y below this - see run.log

# look / sweep angles [deg], absolute = relative to the heading when the scan starts
FULL_SCAN = [-55 * k for k in range(7)]  # camera sees 2*arctan(820/1315) = 64 deg -> 9 deg overlap
AHEAD_SCAN = [0, -45, 45]
SONAR_SWEEP = [0, -15, -30, 15, 30]

# ---- hardware and state ------------------------------------------------------------
arlo = robot.Robot()
cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
cam.intrinsic_matrix = np.array([[CAM_F, 0, 820.0], [0, CAM_F, 616.0], [0, 0, 1]], dtype=np.float64)
cam.arucoMarkerLength = MARKER_SIZE

P = np.empty((0, 3))                    # particles, one row (x, y, θ) each; empty until a landmark is seen
obstacles = {}                          # id -> (x, y, weight); weight = sum of 1/distance of the sightings
last_view = None                        # result of the last sense(); None once we moved
lost_count = 0                          # updates in a row where no particle fit
T0 = time.time()                        # run start (log time stamps)
stats = {"sense": 0.0, "turn": 0.0, "drive": 0.0, "pause": 0.0, "n_sense": 0}


def log(*args):
    """Log line with the seconds since start: [  12.3] ... (flushed, so `| tee` shows it live)."""
    print("[%6.1f]" % (time.time() - T0), *args, flush=True)


def settle(t):
    time.sleep(t)
    stats["pause"] += t


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def cam_pos(x, y, th):
    """Camera position (ROBOT_FRONT ahead of the axle)."""
    return x + ROBOT_FRONT * np.cos(th), y + ROBOT_FRONT * np.sin(th)


def along_across(x, y, th, px, py):
    """Point (px, py) seen from (x, y) looking along th: (distance along, |distance across|)."""
    dx, dy = px - x, py - y
    c, s = np.cos(th), np.sin(th)
    return dx * c + dy * s, abs(dy * c - dx * s)


def sonar(side="front"):
    """One sonar reading [cm]; no echo -> inf."""
    mm = getattr(arlo, "read_%s_ping_sensor" % side)()
    return mm / 10.0 if mm > 0 else np.inf


def front_sonar():
    """Median of 3 front sonar readings [cm]."""
    return sorted(sonar() for _ in range(3))[1]


# ---- motion ------------------------------------------------------------------------
def add_noise(s_xy, s_th):
    global P
    P = P + np.random.normal(0, [s_xy, s_xy, s_th], P.shape)
    P[:, 2] = wrap(P[:, 2])


def move_particles(d=0.0, a=0.0):
    """Motion model: drive d [cm] or turn a [rad], plus noise. The old picture is outdated now."""
    global last_view
    last_view = None
    P[:, 0] += d * np.cos(P[:, 2])
    P[:, 1] += d * np.sin(P[:, 2])
    P[:, 2] += a
    add_noise(2.0 + 0.05 * abs(d), np.radians(3) + 0.1 * abs(a))


def turn(angle):
    """Turn in place [rad], positive = left."""
    deg = np.degrees(angle)
    if abs(deg) < 2:
        return
    if deg > 0:
        arlo.go_diff(SPEED, SPEED, 0, 1)
    else:
        arlo.go_diff(SPEED, SPEED, 1, 0)
    t = abs(deg) / DEG_PER_SEC
    time.sleep(t)
    stats["turn"] += t
    arlo.stop(); settle(STOP_PAUSE)
    move_particles(a=angle)


def straight(dist, stop, fast=False):
    """Drive dist [cm] straight (negative = backwards) until done or stop() returns True.
    Returns (driven, aborted)."""
    speed, cps = (FAST, FAST_CM_PER_SEC) if fast else (SPEED, CM_PER_SEC)
    fwd = int(dist > 0)
    arlo.go_diff(speed, speed + TRIM * fwd, fwd, fwd)
    t0, aborted = time.time(), False
    while time.time() - t0 < abs(dist) / cps + T_START:
        if stop():
            aborted = True
            break
        time.sleep(0.05)
    elapsed = time.time() - t0
    stats["drive"] += elapsed
    arlo.stop(); settle(STOP_PAUSE)
    driven = min(abs(dist), max(0.0, elapsed - T_START) * cps)
    move_particles(driven if fwd else -driven)
    return driven, aborted


def drive(dist, front_stop=SONAR_STOP, sides=True, fast=False):
    """Forward [cm]; abort on front sonar < front_stop or (sides) a side sonar < SIDE_STOP.
    fast: FAST speed (long free drives). Returns (driven, aborted)."""
    if dist <= 0:
        return 0.0, False
    side = itertools.cycle(("left", "right"))   # one side per loop, so the front check stays fast

    def stop():
        if sonar() < front_stop:
            return True
        if sides:
            name = next(side)
            s = sonar(name)
            if s < SIDE_STOP:
                log("%s sonar %.0f cm -> emergency stop" % (name, s))
                return True
        return False
    return straight(dist, stop, fast)


def back_up(dist):
    """Reverse up to dist [cm], keeping BACK_KEEP to whatever is behind."""
    b = sonar("back")
    dist = min(dist, b - BACK_KEEP)
    if dist < 10:
        log("no room behind us (back sonar %.0f cm) -> not backing up" % b)
        return
    log("backing up %.0f cm (back sonar %.0f cm)" % (dist, b))
    straight(-dist, lambda: sonar("back") < BACK_KEEP)


def avoid(goal=None):
    """Turn EVADE_TURN to the freer side (both free and localised: towards goal) and move EVADE_DIST.
    Not free -> try the other side, neither free -> back up. So we always move."""
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
    turn(np.radians(a))                 # back to the original heading
    back_up(EVADE_BACK)


def free_dist(dist):
    """How far we can drive straight ahead before getting closer than CLEAR to a remembered obstacle."""
    if not obstacles or not localised():
        return dist
    x, y, th = pose()
    for ox, oy, _ in obstacles.values():
        ahead, side = along_across(x, y, th, ox, oy)
        # Already inside CLEAR (e.g. after an evasion): we may pass, but get at most 10 cm
        # closer and never closer than 5 cm to touching - else we could never drive away.
        keep = min(CLEAR, max(np.hypot(ox - x, oy - y) - 10.0, OBST_RADIUS + ROBOT_RADIUS + 5.0))
        if ahead > 0 and side < keep:
            dist = min(dist, ahead - np.sqrt(keep ** 2 - side ** 2))
    return dist


def blocked(obs, d_t, a_t):
    """True if a seen obstacle (centre) is closer than CLEAR to the line to the point (d_t, a_t)."""
    for d, a in obs.values():
        r = d + OBST_HALF
        along, across = along_across(0.0, 0.0, a_t, r * np.cos(a), r * np.sin(a))
        if along < d_t and across < CLEAR:
            return True
    return False


def step_to(point, dist):
    """Turn towards world point (x, y), look, then drive dist [cm] - only as far as it is free."""
    x, y, th = pose()
    turn(wrap(np.arctan2(point[1] - y, point[0] - x) - th))
    _, obs = sense()                    # camera check in the new direction before driving
    dist = free_dist(dist)
    if dist < 10 or blocked(obs, dist + ROBOT_RADIUS, 0.0):
        log("path not free -> evade")
        avoid(point)
        return
    if drive(dist, fast=True)[1]:
        avoid(point)


def face_marker(d, a):
    """Turn the axle towards a marker seen at (d, a) from the camera. Returns axle -> marker [cm]."""
    ax, ay = ROBOT_FRONT + d * np.cos(a), d * np.sin(a)
    turn(np.arctan2(ay, ax))
    return np.hypot(ax, ay)


def find_box_echo(max_cm):
    """Sonar sweep +-30 deg. Faces the closest echo < max_cm and returns its gap [cm], else None."""
    best, best_s, cur = None, max_cm, 0
    for a in SONAR_SWEEP:
        turn(np.radians(a - cur)); cur = a
        s = front_sonar()
        if s < best_s:
            best, best_s = a, s
    turn(np.radians((best if best is not None else 0) - cur))
    return None if best is None else best_s


def close_in():
    """Creep on the front sonar until VISIT_DIST. Returns final gap [cm], None if no box echo."""
    s = front_sonar()
    gap = s if s < ECHO_RANGE else None
    if gap is None:
        log("front sonar sees no box (diagonal?) -> sonar sweep")
        gap = find_box_echo(ECHO_RANGE)
    if gap is None:
        return None                     # caller decides: camera saw it -> blind stop counts, else not
    for _ in range(2):                  # 2nd try only after a sonar glitch
        if gap <= VISIT_DIST + BRAKE:
            break
        # at most 3 cm past VISIT_DIST if the echo drops out
        drive(gap - VISIT_DIST + 3.0, front_stop=VISIT_DIST + BRAKE, sides=False)
        s = front_sonar()
        if s >= ECHO_RANGE:
            return VISIT_DIST
        gap = s
    return gap


# ---- particle filter ---------------------------------------------------------------
def ring(lm_id, d, a, n=N):
    """n particles that see lm_id at distance d (camera -> box side) and bearing a."""
    lx, ly = LANDMARKS[lm_id]
    phi = np.random.uniform(-np.pi, np.pi, n)
    r = d + BOX_HALF + np.random.normal(0, 10, n)
    ths = wrap(phi + np.pi - a + np.random.normal(0, np.radians(5), n))
    cx, cy = lx + r * np.cos(phi), ly + r * np.sin(phi)
    return np.column_stack([cx - ROBOT_FRONT * np.cos(ths), cy - ROBOT_FRONT * np.sin(ths), ths])


def filter_update(meas, check_lost=True):
    """Gaussian likelihood + systematic resampling. First landmark ever seen: particles on a ring
    around it. No particle fits twice in a row: re-seed around the closest landmark."""
    global P, lost_count
    if not meas:
        return
    closest = min(meas, key=lambda i: meas[i][0])
    if not len(P):
        log("first landmark %s -> particles on a ring around it" % NAME[closest])
        P, check_lost = ring(closest, *meas[closest]), False
    xs, ys, ths = P.T
    cx, cy = cam_pos(xs, ys, ths)
    logw = np.zeros(len(P))
    for lm_id, (d, a) in meas.items():
        lx, ly = LANDMARKS[lm_id]
        ed = np.hypot(lx - cx, ly - cy) - BOX_HALF
        ea = wrap(np.arctan2(ly - cy, lx - cx) - ths)
        sd = 10.0 + 0.08 * d            # far markers are noisier
        logw += -0.5 * ((d - ed) / sd) ** 2 - 0.5 * (wrap(a - ea) / SIGMA_A) ** 2
    if check_lost and logw.max() < -8.0 * len(meas):
        lost_count += 1
        if lost_count < 2:
            log("measurement fits no particle -> ignored once")
            return
        lost_count = 0
        log("LOST -> re-seeding particles around %s" % NAME[closest])
        P = ring(closest, *meas[closest])
        filter_update(meas, check_lost=False)
        return
    lost_count = 0
    w = np.exp(logw - logw.max()); w /= w.sum()
    cum = np.cumsum(w); cum[-1] = 1.0
    P = P[np.searchsorted(cum, (np.random.uniform() + np.arange(len(w))) / len(w))]


def pose():
    """Mean particle (x, y, θ); all weights are equal after resampling."""
    return P[:, 0].mean(), P[:, 1].mean(), np.arctan2(np.sin(P[:, 2]).mean(), np.cos(P[:, 2]).mean())


def spread():
    """std x, std y [cm] and heading agreement (0..1) of the particles."""
    return P[:, 0].std(), P[:, 1].std(), np.hypot(np.cos(P[:, 2]).mean(), np.sin(P[:, 2]).mean())


def localised():
    """Particles agree on position AND heading (a ring around one landmark does not)."""
    if not len(P):
        return False
    sx, sy, heading_agree = spread()
    return sx < LOC_STD and sy < LOC_STD and heading_agree > 0.9


def snap_to(lm_id, gap):
    """We face lm_id at gap [cm] (front sonar -> box side): put every particle there, keeping its heading."""
    global P, last_view
    ths = wrap(P[:, 2] + np.random.normal(0, np.radians(3), len(P)))
    r = gap + BOX_HALF + SONAR_FRONT
    lx, ly = LANDMARKS[lm_id]
    P = np.column_stack([lx - r * np.cos(ths), ly - r * np.sin(ths), ths])
    last_view = None


def remember(obs):
    """Store obstacle world positions (only when localised); close sightings count more."""
    if not obs or not localised():
        return
    x, y, th = pose()
    cx, cy = cam_pos(x, y, th)
    for i, (d, a) in obs.items():
        if d > OBST_MAX_SEEN:
            continue
        r = d + OBST_HALF
        ox, oy, w = cx + r * np.cos(th + a), cy + r * np.sin(th + a), 1.0 / d
        if not (-60 < ox < _W + 60 and -60 < oy < _H + 60):   # outside the arena -> misreading
            continue
        if i in obstacles:
            px, py, pw = obstacles[i]
            ox, oy, w = (pw * px + w * ox) / (pw + w), (pw * py + w * oy) / (pw + w), pw + w
        else:
            log("obstacle %d at (%.0f, %.0f)" % (i, ox, oy))
        obstacles[i] = (ox, oy, w)


def sense():
    """One camera look; updates the filter and the obstacle map.
    Returns (landmarks, obstacles), each {id: (distance camera -> marker, bearing)}.
    Not moved since the last look -> the same picture again."""
    global last_view
    if last_view is not None:
        return last_view
    settle(SETTLE - STOP_PAUSE)         # camera needs a still robot: the rest of SETTLE
    t0 = time.time()
    raw = {}
    try:
        for _ in range(FLUSH_FRAMES):   # may still show the view from before the move
            cam.get_next_frame()
        frame = cam.get_next_frame()
        ids, ds, angs = cam.detect_aruco_objects(frame) if frame is not None else (None, None, None)
    except Exception as e:              # one broken frame must not end the run
        log("camera error: %s -> nothing seen" % e)
        ids = None
    if ids is not None:
        for i, d, a in zip(ids, ds, angs):
            if 0 < d < 500:             # several faces of one box -> averaged below
                raw.setdefault(int(i), []).append((float(d), float(a)))
    seen = {i: tuple(np.mean(v, axis=0)) for i, v in raw.items()}   # |bearing| < 32 deg: plain mean is fine
    log("seen:", {i: (round(d), round(np.degrees(a))) for i, (d, a) in seen.items()})
    lms = {i: m for i, m in seen.items() if i in LANDMARKS}
    obs = {i: m for i, m in seen.items() if i not in LANDMARKS}
    filter_update(lms)
    remember(obs)
    stats["sense"] += time.time() - t0
    stats["n_sense"] += 1
    last_view = lms, obs
    return last_view


# ---- race --------------------------------------------------------------------------
def scan(target, angles=FULL_SCAN, until_localised=False):
    """Look at each angle [deg]; stop early facing target if seen (or once localised)."""
    cur = 0
    for a in angles:
        turn(np.radians(a - cur)); cur = a
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
        turn(np.radians(120)); drive(60.0)
    return {}


def detour(x, y, gx, gy, target):
    """Waypoint beside the first landmark/obstacle the straight line passes too closely.
    Takes the side with the shorter total path; a side inside another box's zone only if both are."""
    things = [(lx, ly, KEEP_AWAY) for i, (lx, ly) in LANDMARKS.items() if i != target]
    things += [(ox, oy, OBST_KEEP) for ox, oy, _ in obstacles.values()]
    length, th = np.hypot(gx - x, gy - y), np.arctan2(gy - y, gx - x)
    best = None
    for j, (lx, ly, keep) in enumerate(things):
        along, off = along_across(x, y, th, lx, ly)
        if along >= length or along < 40:          # beyond the goal or already passed
            continue
        if off < keep and (best is None or along < best[0]):
            nx, ny = x + along * np.cos(th) - lx, y + along * np.sin(th) - ly   # thing -> line
            best = (along, j, (nx, ny) if off > 1 else (-np.sin(th), np.cos(th)))
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


def step_detour(target, why):
    """Step towards a detour waypoint if the straight line to target needs one. True if stepped."""
    x, y, _ = pose()
    wp = detour(x, y, *LANDMARKS[target], target)
    if wp is None:
        return False
    log("%s -> detour via (%.0f, %.0f)" % ((why,) + tuple(wp)))
    step_to(wp, min(STEP, np.hypot(wp[0] - x, wp[1] - y)))
    return True


def approach(target, lms):
    """Target in view (path checked): turn to it and drive. Far away: shortcut straight into
    BLIND_DIST (re-aim there). Close: blind + sonar. True once visited."""
    goal = LANDMARKS[target]
    d = face_marker(*lms[target]) - ROBOT_FRONT
    if d > BLIND_DIST:                  # shortcut: straight into BLIND_DIST
        dist = free_dist(d - BLIND_DIST + 10.0)
        if dist < 10 or drive(dist, fast=True)[1]:
            avoid(goal)
        return False
    # camera loses the marker below ~90 cm -> blind, then sonar
    remaining = max(d - BLIND_STOP, 0.0)
    if free_dist(remaining) < remaining - 5.0:     # remembered box beside the blind path
        log("remembered obstacle next to the blind path to %s -> evade" % NAME[target])
        avoid(goal)
        return False
    driven, aborted = drive(remaining, front_stop=BLIND_SONAR_STOP)
    if aborted and remaining - driven > 15.0:      # sonar glitch -> try the rest once
        drive(remaining - driven, front_stop=BLIND_SONAR_STOP)
    gap = close_in()
    if gap is None:                     # no echo: probably not facing the box -> re-aim on the camera
        gap = camera_close_in(target)
    if gap is None:                     # never count a visit we are not sure about
        log("%s: no echo and marker not in view -> back up and try again" % NAME[target])
        back_up(30.0)
        return False
    snap_to(target, gap)
    log("%s visited (camera said %.0f cm, sonar now %.0f cm)" % (NAME[target], d, gap))
    return True


def camera_close_in(target):
    """Sonar found no box: face the marker on the camera, then sonar again, or drive on the
    camera distance. Returns final sonar gap [cm], None if the marker is not in view."""
    lms, _ = sense()
    if target not in lms:
        return None
    d, a = lms[target]
    axle = face_marker(d, a)
    log("no echo -> re-aimed on the camera (%s at %.0f cm, %.0f deg)" % (NAME[target], d, np.degrees(a)))
    gap = close_in()
    if gap is None:                     # sonar still blind: trust the camera distance
        drive(axle - SONAR_FRONT - VISIT_DIST, front_stop=VISIT_DIST + BRAKE, sides=False)
        gap = VISIT_DIST
    return gap


def sonar_near(target):
    """Pose says target is too close for the camera: face it; echo fits the expected gap -> visit.
    Echo much closer -> something is in the way: evade. Otherwise we are lost -> relocalise."""
    gx, gy = LANDMARKS[target]
    x, y, th = pose()
    turn(wrap(np.arctan2(gy - y, gx - x) - th))
    expected = np.hypot(gx - x, gy - y) - BOX_HALF - SONAR_FRONT     # sonar -> box side
    _, obs = sense()
    s = front_sonar()
    path = s - BLIND_GAP
    if (blocked(obs, expected + 40.0, 0.0) or s < expected - SONAR_MATCH
            or (s < np.inf and free_dist(path) < path - 5.0)):     # seen, echo or remembered box
        log("echo %.0f cm, %s expected at %.0f cm, obstacles in view %s -> something in the way, evade"
            % (s, NAME[target], expected, list(obs)))
        avoid((gx, gy))
        return False
    if not abs(s - expected) < SONAR_MATCH:
        log("sonar %.0f cm, expected %.0f cm -> relocalise" % (s, expected))
        add_noise(80.0, np.radians(30))
        return False
    drive(max(path, 0.0), front_stop=BLIND_SONAR_STOP)   # close_in only looks ~60 cm far
    g = close_in()
    if g is None:                       # no camera proof here, so no echo = not visited
        log("no echo after driving -> %s is not here, relocalise" % NAME[target])
        add_noise(80.0, np.radians(30))
        return False
    snap_to(target, g)
    log("%s visited on sonar (expected %.0f cm, sonar said %.0f cm)" % (NAME[target], expected, s))
    return True


def toward_seen(target, lms, obs):
    """One step with the target in view: detour if needed, evade if blocked, else approach."""
    d_t, a_t = lms[target]
    in_way = blocked(obs, d_t, a_t)
    if localised() and (in_way or d_t > BLIND_DIST) and step_detour(target, "target in view, but blocked"):
        return False
    if in_way:                          # sonar would take the obstacle for the target
        log("obstacle in front of %s -> evade" % NAME[target])
        avoid(LANDMARKS[target])
        return False
    return approach(target, lms)


def toward_unseen(target):
    """One step on the pose estimate (localised, target not in view)."""
    gx, gy = LANDMARKS[target]
    x, y, th = pose()
    gap = np.hypot(gx - x, gy - y) - BOX_HALF - ROBOT_FRONT
    log("pose x=%.0f y=%.0f th=%.0f deg | spread %.0f/%.0f cm, heading %.2f | %s gap %.0f cm"
        % ((x, y, np.degrees(th)) + spread() + (NAME[target], gap)))
    if step_detour(target, "%s not in view" % NAME[target]):   # also near the target: a box may hide it
        return False
    if gap >= SEE_DIST:
        step_to((gx, gy), min(STEP, gap - SEE_DIST + 20))
        return False
    turn(wrap(np.arctan2(gy - y, gx - x) - th))    # should be visible: face it and look around
    if target in scan(target, AHEAD_SCAN):
        return False
    if step_detour(target, "scan found a box"):    # the scan may just have found the hiding box
        return False
    log("%s not visible -> moving closer" % NAME[target])
    if gap > BLIND_DIST:
        step_to((gx, gy), min(STEP, gap - BLIND_DIST + 10))
        return False
    return sonar_near(target)


def go_to(target):
    """One leg: pose estimate until the camera sees the target, then approach. True once visited."""
    for _ in range(60):
        lms, obs = sense()
        if blocked(obs, OBST_NEAR, 0.0):
            avoid(LANDMARKS[target])
        elif target in lms:
            if toward_seen(target, lms, obs):
                return True
        elif not localised():
            relocalise(target)
        elif toward_unseen(target):
            return True
    return False


def main():
    log("map:", {NAME[i]: p for i, p in LANDMARKS.items()})
    missed, legs, prev = [], [], "start"
    for target in ROUTE:                # L1 too: go_to localises first (relocalise) if needed
        t = time.time()
        if not go_to(target):           # keep going: later landmarks still count
            log("could not reach %s - continuing" % NAME[target])
            missed.append(NAME[target])
        legs.append(("%s->%s" % (prev, NAME[target]), time.time() - t))
        prev = NAME[target]
    log("done - " + ("all landmarks visited" if not missed else "MISSED: " + ", ".join(missed)))
    log("legs:  " + ", ".join("%s %.0f s" % lt for lt in legs))
    total = time.time() - T0
    log("time:  total %.0f s | sense %.0f s (%d x) | turn %.0f s | drive %.0f s | pause %.0f s | rest %.0f s"
        % (total, stats["sense"], stats["n_sense"], stats["turn"], stats["drive"], stats["pause"],
           total - stats["sense"] - stats["turn"] - stats["drive"] - stats["pause"]))


if __name__ == "__main__":
    try:
        main()
    finally:
        arlo.stop()
        cam.terminateCaptureThread()