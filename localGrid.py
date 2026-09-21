import time

import cv2
import numpy as np
import picamera2

DEFAULT_SIZE = 0.145
SCALE = 150
K = np.array([[1315, 0, 820], [0, 1315, 616], [0, 0, 1]], dtype=np.float64)
DIST = np.zeros(5)

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
cv2.line(world_map, (400, 0), (400, 800), (200, 200, 200))
for meter in range(1, 5):
    y_pixel = 750 - meter * SCALE
    cv2.line(world_map, (0, y_pixel), (800, y_pixel), (200, 200, 200))
    cv2.putText(world_map, str(meter) + " m", (5, y_pixel - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
cv2.circle(world_map, (400, 750), 12, (0, 0, 255), -1)

landmark = []
Box_Radius = 0.07
Robot_Radius = 0.1

for i in range(len(ids)):
    marker_id = int(ids[i][0])

    _, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i:i + 1], DEFAULT_SIZE, K, DIST)
    x, y, z = tvec[0][0]
    landmark.append((x, z))
    distance = np.sqrt(x**2 + y**2 + z**2)
    print(f"ID {marker_id}: distance {distance:.3f} m  (x = {x:.3f}, z = {z:.3f}, size {DEFAULT_SIZE} m)")

    px = int(400 + x * SCALE)
    py = int(750 - z * SCALE)
    cv2.circle(world_map, (px, py), int(Box_Radius * SCALE), (255, 0, 0), -1)
    cv2.putText(world_map, "ID " + str(marker_id), (px + 15, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.circle(world_map, (px, py), int((Box_Radius + Robot_Radius) * SCALE), (0, 255, 0), 2)

def in_collision(x, y):
    for (lx, lz) in landmark:
        distance = np.sqrt((x-lx)**2 + (y-lz)**2)
        if distance <= Box_Radius + Robot_Radius:
            return True
    return False

cv2.imwrite("map.png", world_map)