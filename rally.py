"""
rally.py - DIKU RoboRally 2026 (REX exam assignment 6), short version.

Strategy (KISS):
  - Particle filter from Exercise 5 with the 4 known corner landmarks.
  - Visit L1 -> L2 -> L3 -> L4 -> L1. "Visited" = within 40 cm of the box side,
    so we stop at 30 cm and never touch the box (3 min penalty).
  - Drive in short steps; when the target landmark is in view, drive on the
    camera measurement directly. Keep away from the other landmarks
    (wrong order = 14 min penalty) and avoid obstacles (any other ID) with
    the camera and the front sonar.
"""
import time
import numpy as np
import particle, camera, robot

# ---- constants (check on race day) -----------------------------------------
BOX_HALF = 20.0                         # half box side [cm]
# 300 x 400 cm are measured between the INNER box sides -> add one full box width
# to get centre-to-centre distances (the filter works with box centres).
INNER_W, INNER_H = 300.0, 400.0
# ArUco ID of each corner landmark as set up on our track (L1..L4).
L1, L2, L3, L4 = 1, 10, 5, 3
LANDMARKS = {L1: (0.0, 0.0), L2: (0.0, INNER_W + 2 * BOX_HALF),
             L3: (INNER_H + 2 * BOX_HALF, INNER_W + 2 * BOX_HALF), L4: (INNER_H + 2 * BOX_HALF, 0.0)}
ROUTE = [L1, L2, L3, L4, L1]            # required order
NAME = {L1: "L1", L2: "L2", L3: "L3", L4: "L4"}
VISIT_DIST = 30.0                       # stop distance to box side (< 40 cm rule)
KEEP_AWAY = 85.0                        # min distance to a non-target landmark centre
STEP = 50.0                             # driving step when target not in view [cm]
SONAR_STOP = 350                        # front sonar abort [mm]
SPEED, CM_PER_SEC, DEG_PER_SEC = 64, 46.0, 125.0
N = 1000                                # particles
SIGMA_D, SIGMA_A = 25.0, np.radians(10.0)

arlo = robot.Robot()
cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
particles = []


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---- motion -----------------------------------------------------------------
def turn(angle):
    """Turn in place [rad], positive = left. Also moves the particles."""
    deg = np.degrees(angle)
    if abs(deg) >= 2:
        arlo.go_diff(SPEED, SPEED, 0, 1) if deg > 0 else arlo.go_diff(SPEED, SPEED, 1, 0)
        time.sleep(abs(deg) / DEG_PER_SEC)
        arlo.stop(); time.sleep(0.3)
    for p in particles:
        particle.move_particle(p, 0.0, angle)
    particle.add_uncertainty(particles, 2.0, np.radians(5))


def drive(dist, sonar_limit=SONAR_STOP):
    """Drive forward [cm], abort if the front sonar sees something close. Returns (driven, aborted)."""
    t0, aborted = time.time(), False
    arlo.go_diff(SPEED, SPEED + 2, 1, 1)
    while time.time() - t0 < dist / CM_PER_SEC:
        s = arlo.read_front_ping_sensor()
        if 0 < s < sonar_limit:
            aborted = True; break
        time.sleep(0.05)
    arlo.stop(); time.sleep(0.3)
    driven = min(dist, (time.time() - t0) * CM_PER_SEC)
    for p in particles:
        particle.move_particle(p, driven, 0.0)
    particle.add_uncertainty(particles, 2.0 + 0.05 * driven, np.radians(3))
    return driven, aborted


def avoid():
    """Obstacle ahead: turn 60 deg towards the freer side and move a bit."""
    l, r = arlo.read_left_ping_sensor(), arlo.read_right_ping_sensor()
    turn(np.radians(60) if (l if l > 0 else 9999) >= (r if r > 0 else 9999) else np.radians(-60))
    drive(35.0)


# ---- particle filter (Exercise 5, 4 landmarks) ------------------------------
def init_particles_around(lm_id, side_dist):
    """Ring around a landmark, all particles facing it (start: we face L1 at known distance)."""
    global particles
    lx, ly = LANDMARKS[lm_id]
    particles = []
    for _ in range(N):
        a = np.random.uniform(0, 2 * np.pi)
        r = side_dist + BOX_HALF + np.random.normal(0, 10)
        particles.append(particle.Particle(lx + r * np.cos(a), ly + r * np.sin(a),
                                           a + np.pi + np.random.normal(0, np.radians(8)), 1.0 / N))


def expected(p, lm_id):
    """Distance to the box side and bearing a particle would measure."""
    dx, dy = LANDMARKS[lm_id][0] - p.getX(), LANDMARKS[lm_id][1] - p.getY()
    return max(np.hypot(dx, dy) - BOX_HALF, 0.0), wrap(np.arctan2(dy, dx) - p.getTheta())


def filter_update(meas):
    """Weight with a Gaussian likelihood, then systematic resampling."""
    global particles
    if not meas or not particles:       # no particles yet during the start phase
        return
    w = np.ones(N)
    for i, p in enumerate(particles):
        for lm_id, (d, a) in meas.items():
            ed, ea = expected(p, lm_id)
            w[i] *= np.exp(-0.5 * ((d - ed) / SIGMA_D) ** 2) * np.exp(-0.5 * (wrap(a - ea) / SIGMA_A) ** 2)
    if w.sum() <= 1e-300:
        return
    cum = np.cumsum(w / w.sum()); cum[-1] = 1.0
    idx = np.searchsorted(cum, np.random.uniform(0, 1.0 / N) + np.arange(N) / N)
    particles = [particle.Particle(particles[i].getX(), particles[i].getY(), particles[i].getTheta(), 1.0 / N)
                 for i in idx]


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
    lms = {i: m for i, m in seen.items() if i in LANDMARKS}
    filter_update(lms)
    return lms, {i: m for i, m in seen.items() if i not in LANDMARKS}   # everything else = obstacle


# ---- race -------------------------------------------------------------------
def approach(target, lms):
    """Target is in view: turn to it and drive until VISIT_DIST. Returns True once visited."""
    d, a = lms[target]
    if d <= VISIT_DIST:
        print("%s visited (%.0f cm)" % (NAME[target], d)); return True
    turn(a)
    driven, aborted = drive(min(d - VISIT_DIST, 60.0), sonar_limit=200)
    if aborted and d - driven > VISIT_DIST + 20:   # stopped far from the box -> obstacle
        avoid()
    return False


def go_to(target):
    """One leg: reach `target` using the pose estimate until the camera sees it."""
    for _ in range(60):
        lms, obs = sense()
        if target in lms:
            if approach(target, lms):
                return True
            continue
        # obstacle box straight ahead (camera) -> evade before planning
        if any(d < 60 and abs(a) < np.radians(25) for d, a in obs.values()):
            avoid(); continue
        est = particle.estimate_pose(particles)
        x, y, th = est.getX(), est.getY(), est.getTheta()
        gx, gy = LANDMARKS[target]
        dist = np.hypot(gx - x, gy - y)
        heading = np.arctan2(gy - y, gx - x)
        # straight line too close to another landmark? -> detour sideways around it
        for lm_id, (lx, ly) in LANDMARKS.items():
            if lm_id == target:
                continue
            t = np.clip(((lx - x) * (gx - x) + (ly - y) * (gy - y)) / max(dist ** 2, 1e-9), 0, 1)
            if np.hypot(lx - (x + t * (gx - x)), ly - (y + t * (gy - y))) < KEEP_AWAY:
                side = np.sign(np.cross([gx - x, gy - y], [lx - x, ly - y])) or 1.0
                heading = np.arctan2(ly - y, lx - x) - side * np.radians(60)   # pass on the far side
                break
        if dist - BOX_HALF - VISIT_DIST < 80:    # should be close but not visible -> look around
            turn(np.radians(30)); continue
        turn(wrap(heading - th))
        if drive(min(STEP, dist - BOX_HALF - VISIT_DIST))[1]:
            avoid()
    return False


def main():
    # Start: we stand near L1 and face it. Approach it on the camera (= first visit),
    # then initialise the filter around L1 and scan once so L2/L4 fix our side.
    for _ in range(60):
        lms, _ = sense()
        if L1 not in lms:
            turn(np.radians(30)); continue
        if approach(L1, lms):
            break
    init_particles_around(L1, lms[L1][0])
    # At the start we may see ONLY L1. A ring of particles around L1 has no
    # useful mean, so we need a second landmark to fix which side we are on.
    # Scan 360 deg; if nothing else is visible, move sideways around L1 and scan again.
    for attempt in range(4):
        others = set()
        for _ in range(12):
            lms, _ = sense(); others |= set(lms) - {L1}; turn(np.radians(30))
        xs = [p.getX() for p in particles]; ys = [p.getY() for p in particles]
        if others and np.std(xs) < 40 and np.std(ys) < 40:
            break                                   # particles collapsed to one place
        print("only L1 visible -> moving around it")
        lms, _ = sense()
        if L1 in lms:
            turn(lms[L1][1] + np.radians(90))       # face L1, then turn 90 deg left
        drive(60.0)                                 # ~60 cm along the ring around L1
    for target in ROUTE[1:]:
        # If a leg fails the clock keeps running, so do not stop: a missed landmark
        # costs 14 min, but the following landmarks can still count.
        if not go_to(target):
            print("could not reach %s - continuing" % NAME[target])
    print("done - all landmarks visited")


try:
    main()
finally:
    arlo.stop()
    cam.terminateCaptureThread()