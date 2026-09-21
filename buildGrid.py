import time

import cv2
import numpy as np
import picamera2
import matplotlib.pyplot as plt

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

class GridOccupancyMap(object):
    """

    """
    def __init__(self, low=(-2.5, 0), high=(2.5, 5), res=0.05) -> None:
        self.map_area = [low, high]    #a rectangular area    
        self.map_size = np.array([high[0]-low[0], high[1]-low[1]])
        self.resolution = res

        self.n_grids = [ int(s//res) for s in self.map_size]

        self.grid = np.zeros((self.n_grids[0], self.n_grids[1]), dtype=np.uint8)

        self.extent = [self.map_area[0][0], self.map_area[1][0], self.map_area[0][1], self.map_area[1][1]]

    def in_collision(self, pos):
        """
        find if the position is occupied or not. return if the queried pos is outside the map
        """
        indices = [int((pos[i] - self.map_area[0][i]) // self.resolution) for i in range(2)]
        for i, ind in enumerate(indices):
            if ind < 0 or ind >= self.n_grids[i]:
                return 1
        
        return self.grid[indices[0], indices[1]] 

    def populate(self, id, x, y):
        for i in range(self.n_grids[0]):
            for j in range(self.n_grids[1]):
                centroid = np.array([
                    self.map_area[0][0] + self.resolution * (i + 0.5),
                    self.map_area[0][1] + self.resolution * (j + 0.5)
                ])

                if np.linalg.norm(centroid - np.array([x, y])) <= 0.1: #radius is 0.1
                    self.grid[i, j] = id
    
    def draw_map(self):
            #note the x-y axes difference between imshow and plot
            plt.imshow(self.grid.T, cmap="Greys", origin='lower', vmin=0, vmax=1, extent=self.extent, interpolation='none')

map = GridOccupancyMap()
for i in range(len(ids)):
    marker_id = int(ids[i][0])

    _, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i:i + 1], DEFAULT_SIZE, K, DIST)
    x, y, z = tvec[0][0]
    map.populate(marker_id, x, y, z)
    distance = np.sqrt(x**2 + y**2 + z**2)
    print(f"ID {marker_id}: distance {distance:.3f} m  (x = {x:.3f}, z = {z:.3f}, size {DEFAULT_SIZE} m)")

plt.clf()
map.draw_map()
plt.show()