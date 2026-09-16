import time

import cv2
import numpy as np
import picamera2

MARKER_LENGTH = 0.145
SCALE = 150
K = np.array([[1687, 0, 820], [0, 1687, 616], [0, 0, 1]], dtype=np.float64)

cam = picamera2.Picamera2()
cam.configure(cam.create_video_configuration({"size": (1640, 1232), "format": "RGB888"}))
cam.start()
time.sleep(1)
image = cam.capture_array()
cam.close()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict)
if ids is None:
    print("No Landmarks detected")
    exit()

_, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, K, np.zeros(5))
karte = np.full((800, 800, 3), 255, np.uint8)
cv2.line(karte, (400, 0), (400, 800), (200, 200, 200))
for meter in range(1, 5):
    y_pixel = 750 - meter * SCALE
    cv2.line(karte, (0, y_pixel), (800, y_pixel), (200, 200, 200))
    cv2.putText(karte, str(meter) + " m", (5, y_pixel - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
cv2.circle(karte, (400, 750), 12, (0, 0, 255), -1)

for i in range(len(ids)):
    x, y, z = tvecs[i][0]
    print("ID", ids[i][0], "x =", x, "z =", z)
    px = int(400 + x * SCALE)
    py = int(750 - z * SCALE)
    cv2.circle(karte, (px, py), 10, (255, 0, 0), -1)
    cv2.putText(karte, "ID " + str(ids[i][0]), (px + 15, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv2.imwrite("map.png", karte)