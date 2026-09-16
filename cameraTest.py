import time

import cv2
import picamera2
import robot
import numpy

marker_length = 0.145
camera_matrix = numpy.array([[1800,  0, 820],
                            [ 0, 1800, 616],
                            [ 0,  0,  1]])
dist_coeffs = numpy.array([0,0,0,0,0])

arlo = robot.Robot()
cam = picamera2.Picamera2()
cam.configure(cam.create_video_configuration({"size": (1640, 1232), "format": "RGB888"}, queue=False))
cam.start()
time.sleep(1)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

corners, ids, _ = cv2.aruco.detectMarkers(cam.capture_array(), aruco_dict)
_, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

for tvec in tvecs:
    x, y, z = tvec[0]
    distance = numpy.sqrt(x**2 + y**2 + z**2)
    print(f"Distance to marker: {distance:.3f} meters")