import time

import cv2
import numpy as np
import picamera2
import robot

SPEED = 64
DEFAULT_SIZE = 0.145
SCALE = 150
K = np.array([[1315, 0, 820], [0, 1315, 616], [0, 0, 1]], dtype=np.float64)
DIST = np.zeros(5)
arlo = robot.Robot()


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
Box_Radius = 0.175
Box_Shift = 0.12
Robot_Radius = 0.28
CAM_OFFSET = 0.21

for i in range(len(ids)):
    marker_id = int(ids[i][0])  

    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i:i + 1], DEFAULT_SIZE, K, DIST)
    x, y, z = tvec[0][0]
    R, _ = cv2.Rodrigues(rvec[0][0])
    normal = R[[0, 2], 2]
    x, z = np.array([x, z]) - Box_Shift * normal / np.linalg.norm(normal)
    z += CAM_OFFSET
    landmark.append((x, z))
    distance = np.sqrt(x**2 + y**2 + z**2)
    print(f"ID {marker_id}: distance {distance:.3f} m  (x = {x:.3f}, z = {z:.3f}, size {DEFAULT_SIZE} m)")

    px = int(400 + x * SCALE)
    py = int(750 - z * SCALE)
    cv2.circle(world_map, (px, py), int(Box_Radius * SCALE), (255, 0, 0), -1)
    cv2.putText(world_map, "ID " + str(marker_id), (px + 15, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.circle(world_map, (px, py), int((Box_Radius + Robot_Radius) * SCALE), (0, 255, 0), 2)

def in_collision(x, y):
    return any(np.hypot(x - lx, y - lz) <= Box_Radius + Robot_Radius for lx, lz in landmark)

def segment_free(p, q):
    n = int(np.hypot(q[0] - p[0], q[1] - p[1]) / 0.05) + 1
    for t in np.linspace(0, 1, n + 1):
        if in_collision(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])):
            return False
    return True



nodes = [(0,0)]
parents = [None]
Goal = (0, 3.5)
STEP = 0.5
if in_collision(*Goal):
    print("Goal is inside an obstacle")
    exit()

for i in range(1000):
    rx, ry = Goal if np.random.rand() < 0.1 else (np.random.uniform(-2, 2), np.random.uniform(0, 4))
    dists = [np.hypot(nx - rx, ny - ry) for nx, ny in nodes]
    nearest = int(np.argmin(dists))
    nx, ny = nodes[nearest]
    new_node = (nx + (rx - nx) / dists[nearest] * STEP, ny + (ry - ny) / dists[nearest] * STEP)
    if segment_free(nodes[nearest], new_node):
        nodes.append(new_node)
        parents.append(nearest)
        if np.hypot(new_node[0] - Goal[0], new_node[1] - Goal[1]) < STEP and segment_free(new_node, Goal):
            parents.append(len(nodes) - 1)
            nodes.append(Goal)
            print("Goal reached!")
            break
if nodes[-1] != Goal:   
    print("No path found")
    cv2.imwrite("map.png", world_map)
    exit()
    
def shortcut(path):
    smooth = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not segment_free(path[i], path[j]):
            j -= 1
        smooth.append(path[j])
        i = j
    return smooth

path = []
i = len(nodes) - 1
while i is not None:
    path.append(nodes[i])
    i = parents[i]
path.reverse()  
path = shortcut(path)

def px(p):
    return int(400 + p[0] * SCALE), int(750 - p[1] * SCALE)

heading = 90
commands = []
for p, q in zip(path, path[1:]):
    cv2.line(world_map, px(p), px(q), (0, 0, 255), 2)
    cv2.circle(world_map, px(p), 5, (0, 0, 255), -1)
    target = np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0]))
    turn = (target - heading + 180) % 360 - 180
    heading = target
    length = np.hypot(q[0] - p[0], q[1] - p[1])
    commands.append((turn, length))
    print(f"turn {turn:.1f} Degree, drive {length:.2f} m")

M_PER_SEC = 0.460
DEG_PER_SEC_LEFT = 125.0
DEG_PER_SEC_RIGHT = 125.0

def drive(left, right, seconds):
    if left == right:
        arlo.go_diff(SPEED, SPEED + 2, left, right)
    else:
        arlo.go_diff(SPEED, SPEED, left, right)
    time.sleep(seconds)
    arlo.stop()
    time.sleep(0.5)

cv2.imwrite("map.png", world_map)
for (turn, length) in commands:
    time.sleep(1)
    if turn > 0:
        drive(0, 1, turn / DEG_PER_SEC_LEFT)
    elif turn < 0:
        drive(1, 0, -turn / DEG_PER_SEC_RIGHT)
    drive(1, 1, length / M_PER_SEC)

arlo.stop()