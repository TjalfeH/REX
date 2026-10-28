r"""
rally.py - DIKU RoboRally 2026 (REX Ex6).

    L2 ---- 400 ---- L4      Route: L1 -> L2 -> L3 -> L4 -> L1
     |                |
    300  obstacles   300
     |                |
    L1 -------------- L3

Strategy (KISS):
  1. Particle filter (Ex5) on the 4 corner landmarks gives our pose.
  2. Target in view: turn to it and drive. The camera loses the marker below ~90 cm,
     so the last part is driven blind and finished on the front sonar.
  3. Target not in view: face it on the pose estimate and drive one step, look again.
  4. Obstacle seen on the path, or a sonar too close: turn away and drive a bit.

Units: cm and radians.
"""
import time
import numpy as np
import camera, robot

# ---- track and robot [cm] ------------------------------------------------------
L1, L2, L3, L4 = 11, 10, 9, 8           # practice IDs. Race day: 1, 2, 3, 4 !
BOX_HALF = 20.0                         # half landmark box side
W, H = 440.0, 340.0                     # landmark centres: 400 x 300 between the boxes + 2 * BOX_HALF
LANDMARKS = {L1: (0.0, 0.0), L2: (0.0, H), L3: (W, 0.0), L4: (W, H)}
ROUTE = [L1, L2, L3, L4, L1]
ROBOT_FRONT = 21.0                      # axle -> camera
SONAR_FRONT = 28.0                      # axle -> front sonar

# ---- driving -------------------------------------------------------------------
SPEED, CM_PER_SEC, DEG_PER_SEC = 64, 46.0, 125.0   # measured on our robot
STEP = 120.0                            # max drive between two looks when the target is not in view
SONAR_STOP = 45.0                       # front sonar closer than this -> stop
SIDE_STOP = 12.0                        # side sonar closer than this -> stop
CLEAR = 60.0                            # seen obstacle closer than this to our path -> blocked
BLIND_DIST = 130.0                      # camera -> marker; closer: drive the rest blind
KEEP = 75.0                             # detour: path -> remembered obstacle centre
VISIT_STOP = 35.0                       # creep until front sonar < this (rolls out to ~25 cm; rule: 40 cm)

# ---- particle filter -------------------------------------------------------------
N = 500                                 # particles
SIGMA_A = np.radians(10.0)              # bearing noise

arlo = robot.Robot()
cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
cam.intrinsic_matrix = np.array([[1315.0, 0, 820.0], [0, 1315.0, 616.0], [0, 0, 1]])   # Ex4 calibration
cam.arucoMarkerLength = 0.145
obstacles = {}                          # id -> (x, y) world position of seen obstacles
P = None                                # particles: N x 3 array (x, y, theta); None until a landmark is seen
T0 = time.time()


def log(*args):
    print("[%5.1f]" % (time.time() - T0), *args, flush=True)


def wrap(a):
    """Angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def sonar(side="front"):
    """Sonar distance [cm]; no echo -> inf."""
    mm = getattr(arlo, "read_%s_ping_sensor" % side)()
    return mm / 10.0 if mm > 0 else np.inf


# ---- motion ------------------------------------------------------------------------
def move_particles(d, a):
    """Motion model: drive d [cm] straight, then turn a [rad], plus noise."""
    if P is None:
        return
    P[:, 0] += d * np.cos(P[:, 2])
    P[:, 1] += d * np.sin(P[:, 2])
    P[:, 2] = wrap(P[:, 2] + a)
    P[:, :2] += np.random.normal(0, 2 + 0.05 * abs(d), (N, 2))
    P[:, 2] += np.random.normal(0, np.radians(3) + 0.1 * abs(a), N)


def turn(a):
    """Turn in place by a [rad], positive = left."""
    if abs(a) < np.radians(2):
        return
    if a > 0:
        arlo.go_diff(SPEED, SPEED, 0, 1)
    else:
        arlo.go_diff(SPEED, SPEED, 1, 0)
    time.sleep(np.degrees(abs(a)) / DEG_PER_SEC)
    arlo.stop()
    time.sleep(0.1)
    move_particles(0, a)


def drive(dist, stop_at=SONAR_STOP):
    """Drive forward dist [cm]. Stops early if the front sonar < stop_at or a side sonar < SIDE_STOP.
    Returns True if it stopped early."""
    if dist <= 0:
        return False
    arlo.go_diff(SPEED, SPEED + 2, 1, 1)                # +2: our right wheel is a bit slower
    t0, stopped = time.time(), False
    while time.time() - t0 < 0.15 + dist / CM_PER_SEC:  # 0.15 s to get going
        front, left, right = sonar(), sonar("left"), sonar("right")
        if front < stop_at or min(left, right) < SIDE_STOP:
            log("stop: sonar front %.0f, left %.0f, right %.0f cm" % (front, left, right))
            stopped = True
            break
        time.sleep(0.05)
    t = time.time() - t0
    arlo.stop()
    time.sleep(0.1)
    move_particles(min(dist, max(0, t - 0.15) * CM_PER_SEC), 0)
    return stopped


def back_up(dist):
    """Reverse dist [cm], unless something is close behind us."""
    if sonar("back") < dist + 20:
        return
    arlo.go_diff(SPEED, SPEED, 0, 0)
    time.sleep(0.15 + dist / CM_PER_SEC)
    arlo.stop()
    time.sleep(0.1)
    move_particles(-dist, 0)


# ---- particle filter ---------------------------------------------------------------
def ring(lm, d, a):
    """N particles that all see landmark lm at distance d and bearing a: a ring around it."""
    lx, ly = LANDMARKS[lm]
    phi = np.random.uniform(-np.pi, np.pi, N)           # where on the ring
    r = d + BOX_HALF + np.random.normal(0, 10, N)       # camera -> landmark centre
    th = wrap(phi + np.pi - a)                          # heading that gives bearing a
    cx, cy = lx + r * np.cos(phi), ly + r * np.sin(phi)  # camera position
    return np.column_stack([cx - ROBOT_FRONT * np.cos(th), cy - ROBOT_FRONT * np.sin(th), th])


def update(meas):
    """Weight the particles with the landmark measurements {id: (d, a)} and resample."""
    global P
    if not meas:
        return
    closest = min(meas, key=lambda i: meas[i][0])
    if P is None:                                       # first landmark ever seen
        P = ring(closest, *meas[closest])
    cx = P[:, 0] + ROBOT_FRONT * np.cos(P[:, 2])        # camera positions
    cy = P[:, 1] + ROBOT_FRONT * np.sin(P[:, 2])
    logw = np.zeros(N)
    for lm, (d, a) in meas.items():
        lx, ly = LANDMARKS[lm]
        exp_d = np.hypot(lx - cx, ly - cy) - BOX_HALF   # expected distance and bearing
        exp_a = wrap(np.arctan2(ly - cy, lx - cx) - P[:, 2])
        sd = 10 + 0.08 * d                              # far markers are noisier
        logw += -0.5 * ((d - exp_d) / sd) ** 2 - 0.5 * (wrap(a - exp_a) / SIGMA_A) ** 2
    if logw.max() < -8 * len(meas):                     # no particle fits: we are lost
        log("lost -> particles around landmark", closest)
        P = ring(closest, *meas[closest])
        update({closest: meas[closest]})
        return
    w = np.exp(logw - logw.max())                       # log weights: no underflow
    P = P[np.random.choice(N, N, p=w / w.sum())]


def pose():
    """Mean particle (x, y, theta)."""
    return P[:, 0].mean(), P[:, 1].mean(), np.arctan2(np.sin(P[:, 2]).mean(), np.cos(P[:, 2]).mean())


def localised():
    """Particles agree on position and heading (a ring around one landmark does not)."""
    if P is None:
        return False
    heading_agree = np.hypot(np.cos(P[:, 2]).mean(), np.sin(P[:, 2]).mean())
    return P[:, 0].std() < 50 and P[:, 1].std() < 50 and heading_agree > 0.9


def unsure():
    """The sensors don't fit the pose: spread the particles out, so the next round scans again."""
    P[:, :2] += np.random.normal(0, 80, (N, 2))
    P[:, 2] += np.random.normal(0, np.radians(30), N)


def at_landmark(lm, gap):
    """We face lm with gap [cm] between front sonar and box: move all particles there."""
    global P
    lx, ly = LANDMARKS[lm]
    r = gap + BOX_HALF + SONAR_FRONT                    # axle -> landmark centre
    th = P[:, 2]
    P = np.column_stack([lx - r * np.cos(th), ly - r * np.sin(th), th])


# ---- camera ------------------------------------------------------------------------
def look():
    """Take a picture and update the filter. Returns (landmarks, obstacles), each {id: (d, a)}."""
    time.sleep(0.2)                                     # robot must stand still for a sharp image
    seen = {}
    try:
        for _ in range(2):                              # old frames still in the buffer
            cam.get_next_frame()
        ids, dists, angles = cam.detect_aruco_objects(cam.get_next_frame())
        if ids is not None:
            seen = {int(i): (d, a) for i, d, a in zip(ids, dists, angles)}
    except Exception as e:                              # a broken frame must not end the run
        log("camera error:", e)
    log("seen:", {i: (round(d), round(np.degrees(a))) for i, (d, a) in seen.items()})
    lms = {i: m for i, m in seen.items() if i in LANDMARKS}
    obs = {i: m for i, m in seen.items() if i not in LANDMARKS}
    update(lms)
    remember(obs)
    return lms, obs


def remember(obs):
    """Store where seen obstacles are, so we still know about them when the camera can't see them."""
    if not localised():
        return
    x, y, th = pose()
    cx, cy = x + ROBOT_FRONT * np.cos(th), y + ROBOT_FRONT * np.sin(th)
    for i, (d, a) in obs.items():
        if d < 250:                                     # far sightings are too inaccurate
            r = d + 12                                  # marker -> obstacle centre
            new = (cx + r * np.cos(th + a), cy + r * np.sin(th + a))
            old = obstacles.get(i, new)
            obstacles[i] = ((old[0] + new[0]) / 2, (old[1] + new[1]) / 2)   # average out the noise


def boxes(target):
    """Everything to keep away from: remembered obstacles and the other landmarks (bigger: +10 cm).
    Returns [(x, y, extra margin)]."""
    return ([(ox, oy, 0.0) for ox, oy in obstacles.values()] +
            [(lx, ly, 10.0) for i, (lx, ly) in LANDMARKS.items() if i != target])


def free_dist(dist, target):
    """How far we can drive straight ahead before passing closer than CLEAR to a box (see boxes)."""
    if not localised():
        return dist
    x, y, th = pose()
    for ox, oy, extra in boxes(target):
        ahead = (ox - x) * np.cos(th) + (oy - y) * np.sin(th)
        side = abs((oy - y) * np.cos(th) - (ox - x) * np.sin(th))
        keep = min(CLEAR + extra, np.hypot(ox - x, oy - y) - 5)   # already closer: just don't get closer
        if ahead > 0 and side < keep:
            dist = min(dist, ahead - np.sqrt(keep ** 2 - side ** 2))
    return dist


def detour(x, y, gx, gy, target):
    """Waypoint beside the first box (see boxes) that is closer than KEEP to the line to the goal."""
    length = np.hypot(gx - x, gy - y)
    ux, uy = (gx - x) / length, (gy - y) / length
    best = None
    for ox, oy, extra in boxes(target):
        along = (ox - x) * ux + (oy - y) * uy
        across = (oy - y) * ux - (ox - x) * uy          # > 0: box left of the line
        if 0 < along < length and abs(across) < KEEP + extra and (best is None or along < best[0]):
            best = (along, ox, oy, across, extra)
    if best is None:
        return None
    _, ox, oy, across, extra = best
    side = -1 if across > 0 else 1                      # pass it on the side the line is already on
    r = KEEP + extra + 20
    for s in (side, -side):                             # that side inside another box? -> the other side
        wx, wy = ox - s * r * uy, oy + s * r * ux
        if all(np.hypot(wx - bx, wy - by) > KEEP for bx, by, _ in boxes(target) if (bx, by) != (ox, oy)):
            break
    return wx, wy


def blocked(obs, d_t, a_t):
    """Is a seen obstacle closer than CLEAR to the straight line to the point (d_t, a_t)?"""
    for d, a in obs.values():
        along, across = d * np.cos(a - a_t), abs(d * np.sin(a - a_t))
        if along < d_t and across < CLEAR:
            return True
    return False


# ---- race --------------------------------------------------------------------------
def avoid(target):
    """Something is in the way: turn 60 deg to the freer side and drive 40 cm.
    That side not free either -> try the other side, then back up."""
    a = np.radians(60) if sonar("left") > sonar("right") else np.radians(-60)
    for turn_by in (a, -2 * a):
        turn(turn_by)
        _, obs = look()
        if not blocked(obs, 70, 0) and sonar() > 60 and free_dist(40, target) >= 40:
            drive(40)
            return
    turn(a)                                             # back to where we started
    back_up(30)


def scan(target):
    """Turn around in 55 deg steps (the camera sees 64 deg) until the target is seen or we are localised."""
    for _ in range(6):
        turn(np.radians(-55))
        lms, _ = look()
        if target in lms or localised():
            return
    drive(60)                                           # nothing helped: try from another spot


def creep(target):
    """Close in on the front sonar. True (and particles moved to the landmark) once within 40 cm."""
    s = sonar()
    for a in (np.radians(-15), np.radians(30)):         # box at an angle -> echo may miss: try left/right
        if s < 70:
            break
        turn(a)
        s = sonar()
    if s >= 70:
        turn(np.radians(-15))
        log("no echo from landmark", target, "-> back up and look again")
        back_up(30)
        return False
    drive(s - 22, stop_at=VISIT_STOP)                   # never closer than 22 cm, even if the echo drops out
    gap = sonar()
    if gap == np.inf:                                   # echo lost right in front of the box
        gap = 22.0
    if gap > 40:
        return False
    at_landmark(target, gap)
    log("landmark", target, "visited, sonar %.0f cm" % gap)
    back_up(30)                                         # room to turn and leave without touching the box
    return True


def approach(target, d, a):
    """Target seen at (d, a): turn to it and drive. Far: drive close and look again (return False).
    Close: drive blind, then creep on the sonar. True once visited."""
    x, y = ROBOT_FRONT + d * np.cos(a), d * np.sin(a)  # marker seen from the axle
    turn(np.arctan2(y, x))
    d = np.hypot(x, y) - ROBOT_FRONT
    if d > BLIND_DIST:
        dist = free_dist(d - BLIND_DIST + 10, target)
        if dist < 10 or drive(dist):                    # something in the way
            avoid(target)
        return False
    if free_dist(d - 35, target) < d - 40:                      # remembered obstacle next to the blind path
        avoid(target)
        return False
    drive(d - 35, stop_at=33)                           # blind, until the camera would be ~35 cm away
    return creep(target)


def sonar_visit(target):
    """Pose says the target is close, but the camera can't see it: check it with the front sonar."""
    gx, gy = LANDMARKS[target]
    x, y, th = pose()
    turn(wrap(np.arctan2(gy - y, gx - x) - th))
    expected = np.hypot(gx - x, gy - y) - BOX_HALF - SONAR_FRONT
    _, obs = look()
    if blocked(obs, expected + 40, 0):                  # the echo would come from that obstacle
        avoid(target)
        return False
    s = sonar()
    if abs(s - expected) > 30:                          # something else in front, or the pose is wrong
        log("sonar %.0f cm, expected %.0f cm -> pose is wrong, scan again" % (s, expected))
        unsure()
        return False
    drive(s - 40, stop_at=33)
    return creep(target)


def go_to(target):
    """Drive to one landmark. True once visited."""
    gx, gy = LANDMARKS[target]
    wp = None                                           # detour waypoint we are heading for
    for _ in range(60):
        lms, obs = look()
        if blocked(obs, 80, 0):                         # obstacle right in front
            avoid(target)
        elif target in lms:
            d, a = lms[target]
            if blocked(obs, d, a):                      # obstacle between us and the target
                avoid(target)
            elif approach(target, d, a):
                return True
        elif not localised() and min(lms.values(), default=(0, 0))[0] > 150:
            d, a = min(lms.values())                    # one far landmark is not enough: head for it,
            turn(a)                                     # near the arena corner more come into view
            if drive(min(100, d - 100)):
                avoid(target)
        elif not localised():
            scan(target)
        else:                                           # drive on the pose estimate
            x, y, th = pose()
            if wp is not None and np.hypot(wp[0] - x, wp[1] - y) < 30:
                wp = None                               # reached it
            if wp is None:                              # keep it once chosen, else we swing back and forth
                wp = detour(x, y, gx, gy, target)
            tx, ty = wp if wp is not None else (gx, gy)
            err = wrap(np.arctan2(ty - y, tx - x) - th)
            gap = np.hypot(gx - x, gy - y) - BOX_HALF - ROBOT_FRONT
            log("pose (%.0f, %.0f, %.0f deg), landmark %d is %.0f cm away, heading for (%.0f, %.0f)"
                % (x, y, np.degrees(th), target, gap, tx, ty))
            if abs(err) > np.radians(15):
                turn(err)                               # face it; the next look checks the way
            elif wp is None and gap < BLIND_DIST:
                if sonar_visit(target):
                    return True
            else:
                dist = np.hypot(tx - x, ty - y) if wp is not None else gap - BLIND_DIST + 20
                dist = free_dist(min(STEP, dist), target)
                if dist < 10 or blocked(obs, dist, 0) or drive(dist):
                    avoid(target)
                    wp = None                           # plan again from where we ended up
    return False


def main():
    missed, legs = [], []
    for target in ROUTE:
        log("---- going to landmark", target)
        t = time.time()
        if not go_to(target):
            log("could not reach landmark", target, "- going on")
            missed.append(target)
        legs.append("%d: %.0f s" % (target, time.time() - t))
    log("done -", "MISSED %s" % missed if missed else "all landmarks visited", "| legs", ", ".join(legs))


if __name__ == "__main__":
    try:
        main()
    finally:
        arlo.stop()
        cam.terminateCaptureThread()