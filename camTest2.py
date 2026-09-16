import time
import cv2
import numpy as np
import picamera2

MARKER_LENGTH = 0.145   # schwarzes Quadrat in Metern
FX = 1800               # Brennweite in Pixeln

cam = picamera2.Picamera2()
cam.configure(cam.create_video_configuration({"size": (1400, 900), "format": "RGB888"}))
cam.start()
time.sleep(1)
bild = cam.capture_array()
cam.close()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
corners, ids, _ = cv2.aruco.detectMarkers(bild, aruco_dict)

if ids is None:
    print("Kein Marker erkannt")
else:
    for i in range(len(ids)):
        c = corners[i][0]
        seite = np.linalg.norm(c[0] - c[1])          # Markerbreite in Pixeln
        entfernung = FX * MARKER_LENGTH / seite
        print("ID", ids[i][0], "Entfernung:", round(entfernung, 2), "m")