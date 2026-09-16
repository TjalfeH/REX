import time

import cv2
import picamera2
import robot

POWER = 64
STOP_MM = 400

arlo = robot.Robot()
cam = picamera2.Picamera2()
cam.configure(cam.create_video_configuration({"size": (1640, 1232), "format": "RGB888"}, queue=False))
cam.start()
time.sleep(1)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)


def drive(left, right, seconds):
    arlo.go_diff(POWER, POWER, left, right)
    time.sleep(seconds)
    arlo.stop()
    time.sleep(0.5)


while True:
    distance = arlo.read_front_ping_sensor()
    if 0 < distance < STOP_MM:
        break

    corners, ids, _ = cv2.aruco.detectMarkers(cam.capture_array(), aruco_dict)

    if ids is None:
        drive(1, 0, 0.3)
        continue

    x = corners[0][0][:, 0].mean()

    if x < 520:
        drive(0, 1, 0.15)
    elif x > 1120:
        drive(1, 0, 0.15)
    else:
        drive(1, 1, 0.5)

arlo.stop()
cam.stop()
print("arrived")