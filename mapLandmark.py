import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import picamera2

MARKER_LENGTH = 0.145
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
    print("Keine Landmarken gefunden")
    exit()

_, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, K, np.zeros(5))

plt.plot(0, 0, "r^")
for i in range(len(ids)):
    x, y, z = tvecs[i][0]
    print("ID", ids[i][0], "x =", x, "z =", z)
    plt.plot(x, z, "bs")
    plt.text(x, z, str(ids[i][0]))

plt.xlabel("x [m]")
plt.ylabel("z [m]")
plt.axis("equal")
plt.grid()
plt.savefig("map.png")