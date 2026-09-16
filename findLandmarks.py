import math
import time

import cv2
import numpy as np
import picamera2
import robot

POWER = 64
DEG_PER_SEC = 90
M_PER_SEC = 0.35
MARKER_LENGTH = 0.145
STOP_DIST = 0.4

K = np.array([[1687, 0, 820], [0, 1687, 616], [0, 0, 1]], dtype=np.float64)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

arlo = robot.Robot()
cam = picamera2.Picamera2()
cam.configure(cam.create_video_configuration({"size": (1640, 1232), "format": "RGB888"}, queue=False))
cam.start()
time.sleep(1)


def look():
    time.sleep(0.3)
    corners, ids, _ = cv2.aruco.detectMarkers(cam.capture_array(), aruco_dict)
    if ids is None:
        return None
    _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, K, np.zeros(5))
    x, y, z = tvecs[0][0]
    return math.hypot(x, z), math.atan2(x, z)


def rotate(angle):
    if angle > 0:
        arlo.go_diff(POWER, POWER, 1, 0)
    else:
        arlo.go_diff(POWER, POWER, 0, 1)
    time.sleep(abs(math.degrees(angle)) / DEG_PER_SEC)
    arlo.stop()


def forward(dist):
    arlo.go_diff(POWER, POWER, 1, 1)
    time.sleep(dist / M_PER_SEC)
    arlo.stop()


while True:
    seen = look()
    if seen is None:
        rotate(math.radians(30))
        continue
    dist, angle = seen
    print(dist, math.degrees(angle))
    if dist < STOP_DIST + 0.05:
        break
    if abs(angle) > math.radians(6):
        rotate(angle)
    else:
        forward(min(dist - STOP_DIST, 0.5))

arlo.stop()
print("arrived")