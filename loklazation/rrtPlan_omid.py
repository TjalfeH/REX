import time
import argparse
import math

import cv2
import numpy as np
import picamera2

SPEED = 64
DEFAULT_SIZE = 0.145
SCALE = 150
# The earlier landmark code used 1797.39 at 1640x1232. Check this
# against a measured distance before trusting the map.
K = np.array([[1797.39, 0, 820], [0, 1797.39, 616], [0, 0, 1]], dtype=np.float64)
DIST = np.zeros(5)

parser = argparse.ArgumentParser()
parser.add_argument("--drive", action="store_true", help="Enable motors after checking map.png")
parser.add_argument("--expected-ids", type=int, nargs="*", default=[5, 6, 7, 8, 9, 10],
                    help="Boxes known to be in the driving area")
parser.add_argument("--box-width", type=float, default=0.25, help="Measured width in metres")
parser.add_argument("--box-depth", type=float, default=0.25, help="Measured depth in metres")
parser.add_argument("--robot-radius", type=float, default=0.20, help="Robot centre to outer edge in metres")
parser.add_argument("--margin", type=float, default=0.10, help="Extra clearance in metres")
args = parser.parse_args()
if min(args.box_width, args.box_depth, args.robot_radius) <= 0 or args.margin < 0:
    parser.error("Dimensions must be positive, and margin must be nonnegative")


cam = picamera2.Picamera2()
cam.configure(cam.create_video_configuration({"size": (1640, 1232), "format": "RGB888"}, queue=False))
cam.start()
time.sleep(1)
image = cam.capture_array()
cam.close()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict)
if ids is None:
    print("No Landmarks detected")
    exit()

world_map = np.full((800, 800, 3), 255, np.uint8)
for meter in range(1, 5):
    y_pixel = 750 - meter * SCALE
    cv2.line(world_map, (0, y_pixel), (800, y_pixel), (200, 200, 200))
    cv2.putText(world_map, str(meter) + " m", (5, y_pixel - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
cv2.circle(world_map, (400, 750), 12, (0, 0, 255), -1)

for meter in range(-2, 3):
    x_pixel = 400 + meter * SCALE
    cv2.line(world_map, (x_pixel, 0), (x_pixel, 800), (200, 200, 200))
    cv2.putText(world_map, str(meter) + " m", (x_pixel + 5, 790), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))

landmark = []
seen_ids = set()
# ArUco measures the marker on the SIDE of the box, not the box centre.
# Assume the marker is at the centre of a face and cover the farthest corner.
Box_Radius = max(math.hypot(args.box_width / 2, args.box_depth),
                 math.hypot(args.box_depth / 2, args.box_width))
Robot_Radius = args.robot_radius
SAFE_RADIUS = Box_Radius + Robot_Radius + args.margin
print(f"Safety distance from each marker: {SAFE_RADIUS:.2f} m")

for i in range(len(ids)):
    marker_id = int(ids[i][0])
    seen_ids.add(marker_id)

    _, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i:i + 1], DEFAULT_SIZE, K, DIST)
    x, y, z = tvec[0][0]
    landmark.append((x, z))
    distance = np.sqrt(x**2 + y**2 + z**2)
    print(f"ID {marker_id}: distance {distance:.3f} m  (x = {x:.3f}, z = {z:.3f}, size {DEFAULT_SIZE} m)")

    px = int(400 + x * SCALE)
    py = int(750 - z * SCALE)
    cv2.circle(world_map, (px, py), int(Box_Radius * SCALE), (255, 0, 0), -1)
    cv2.putText(world_map, "ID " + str(marker_id), (px + 15, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.circle(world_map, (px, py), int(SAFE_RADIUS * SCALE), (0, 255, 0), 2)

# Check the COMPLETE segment, including its middle, against each box.
def segment_is_clear(a, b):
    for (lx, lz) in landmark:
        dx, dz = b[0] - a[0], b[1] - a[1]
        segment_length_sq = dx * dx + dz * dz
        t = 0 if segment_length_sq == 0 else max(0, min(1,
            ((lx - a[0]) * dx + (lz - a[1]) * dz) / segment_length_sq))
        closest = (a[0] + t * dx, a[1] + t * dz)
        if math.dist(closest, (lx, lz)) <= SAFE_RADIUS:
            return False
    return True

nodes = [(0,0)]
parents = [None]
Goal = (0, 3.5)
STEP = 0.25

missing = sorted(set(args.expected_ids) - seen_ids)
if missing:
    cv2.imwrite("map.png", world_map)
    print(f"STOP: box IDs {missing} are missing. Map saved to map.png; robot did not move.")
    raise SystemExit(1)
if not segment_is_clear((0, 0), (0, 0)) or not segment_is_clear(Goal, Goal):
    cv2.imwrite("map.png", world_map)
    print("STOP: robot start or goal overlaps a box safety area. Map saved to map.png.")
    raise SystemExit(1)


for i in range(1000):
    

    rx = np.random.uniform(-2, 2)
    ry = np.random.uniform(0, 4)

    dists = []
    for (nx, ny) in nodes:
        dists.append(np.sqrt((nx-rx)**2 + (ny-ry)**2))
    nearest = dists.index(min(dists))
    nx, ny = nodes[nearest]
    dx = rx - nx
    dy = ry - ny
    if dists[nearest] < 1e-9:
        continue
    dx, dy = (dx / dists[nearest])*STEP, (dy / dists[nearest])*STEP
    new_node = (nx + dx, ny + dy)
    if -2 <= new_node[0] <= 2 and 0 <= new_node[1] <= 4 and segment_is_clear((nx, ny), new_node):
        nodes.append(new_node)
        parents.append(nearest)
        if np.sqrt((new_node[0]-Goal[0])**2 + (new_node[1]-Goal[1])**2) < STEP and segment_is_clear(new_node, Goal):
            parents.append(len(nodes)-1)
            nodes.append(Goal)
            print("Goal reached!")
            break
if nodes[-1] != Goal:   
    print("No path found")
    cv2.imwrite("map.png", world_map)
    exit()

path = []
i = len(nodes) - 1
while i is not None:
    path.append(nodes[i])
    i = parents[i]
path.reverse()  

for k in range(len(path) - 1):
    (x1, y1) = path[k]
    (x2, y2) = path[k + 1]
    px1 = int(400 + x1 * SCALE)
    py1 = int(750 - y1 * SCALE)
    px2 = int(400 + x2 * SCALE)
    py2 = int(750 - y2 * SCALE)
    cv2.line(world_map, (px1, py1), (px2, py2), (0, 0, 255), 2)
    cv2.circle(world_map, (px1, py1), 5, (0, 0, 255), -1)   


heading = 90
commands = []

for k in range(len(path) - 1):
    (x1, y1) = path[k]
    (x2, y2) = path[k + 1]
    dx = x2 - x1
    dy = y2 - y1

    target = np.degrees(np.arctan2(dy, dx))
    turn = target - heading
    if turn > 180:
        turn -= 360
    if turn < -180:
        turn += 360
    heading = target

    length = np.sqrt(dx**2 + dy**2)
    commands.append((turn, length))
    print(f"turn {turn:.1f} Degree, drive {length:.2f} m")

M_PER_SEC = 0.430  # Measure on your floor and motor power.
DEG_PER_SEC = 120.0  # Measure on your floor and motor power.

def drive(left, right, seconds):
    arlo.go_diff(SPEED, SPEED, left, right)
    time.sleep(seconds)
    arlo.stop()
    time.sleep(0.5)

cv2.imwrite("map.png", world_map)
print("Planned route saved to map.png. This is not the actual route driven.")
if args.drive:
    import robot
    arlo = robot.Robot()
    print("Open-loop drive: robot position is not checked again while moving.")
    try:
        for (turn, length) in commands:
            time.sleep(1)
            if turn > 0:
                drive(0, 1, turn / DEG_PER_SEC)
            elif turn < 0:
                drive(1, 0, -turn / DEG_PER_SEC)
            drive(1, 1, length / M_PER_SEC)
    finally:
        arlo.stop()
else:
    print("Preview only: motors were not started. Check map.png before --drive.")
