import cv2
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer
import sys




# Flags
showGUI = False  # SSH on the robot cannot open OpenCV GUI windows
onRobot = True  # Whether or not we are running on the Arlo robot


def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
      You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot


if isRunningOnArlo():
    sys.path.append("/home/pi/REX-students/Arlo/python")


try:
    import robot
    onRobot = True
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
    onRobot = False




# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)

# Landmarks.
# The robot knows the position of 2 landmarks. Their coordinates are in the unit centimeters [cm].
landmarkIDs = [1, 3]
landmarks = {
    1: (0.0, 0.0),  # Coordinates for landmark 1
    3: (150.0, 0.0)  # Landmark 3 is 3 metres from landmark 1
}
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks





def jet(x):
    """Colour map for drawing particles. This function determines the colour of 
    a particle from its weight."""
    r = (x >= 3.0/8.0 and x < 5.0/8.0) * (4.0 * x - 3.0/2.0) + (x >= 5.0/8.0 and x < 7.0/8.0) + (x >= 7.0/8.0) * (-4.0 * x + 9.0/2.0)
    g = (x >= 1.0/8.0 and x < 3.0/8.0) * (4.0 * x - 1.0/2.0) + (x >= 3.0/8.0 and x < 5.0/8.0) + (x >= 5.0/8.0 and x < 7.0/8.0) * (-4.0 * x + 7.0/2.0)
    b = (x < 1.0/8.0) * (4.0 * x + 1.0/2.0) + (x >= 1.0/8.0 and x < 3.0/8.0) + (x >= 3.0/8.0 and x < 5.0/8.0) * (-4.0 * x + 5.0/2.0)

    return (255.0*r, 255.0*g, 255.0*b)

def draw_world(est_pose, particles, world):
    """Visualization.
    This functions draws robots position in the world coordinate system."""

    # Fix the origin of the coordinate system
    offsetX = 100
    offsetY = 250

    # Constant needed for transforming from world coordinates to screen coordinates (flip the y-axis)
    ymax = world.shape[0]

    world[:] = CWHITE # Clear background to white

    # Find largest weight
    max_weight = 0
    for particle in particles:
        max_weight = max(max_weight, particle.getWeight())

    # Draw particles
    for particle in particles:
        x = int(particle.getX() + offsetX)
        y = ymax - (int(particle.getY() + offsetY))
        colour = jet(particle.getWeight() / max_weight)
        cv2.circle(world, (x,y), 2, colour, 2)
        b = (int(particle.getX() + 15.0*np.cos(particle.getTheta()))+offsetX, 
                                     ymax - (int(particle.getY() + 15.0*np.sin(particle.getTheta()))+offsetY))
        cv2.line(world, (x,y), b, colour, 2)

    # Draw landmarks
    for i in range(len(landmarkIDs)):
        ID = landmarkIDs[i]
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        cv2.circle(world, lm, 5, landmark_colors[i], 2)

    # Draw estimated robot pose
    a = (int(est_pose.getX())+offsetX, ymax-(int(est_pose.getY())+offsetY))
    b = (int(est_pose.getX() + 15.0*np.cos(est_pose.getTheta()))+offsetX, 
         ymax-(int(est_pose.getY() + 15.0*np.sin(est_pose.getTheta()))+offsetY))
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)

# Driving calibration (from Exercise 4)
SPEED = 64
CM_PER_SEC = 46.0          # 0.46 m/s
DEG_PER_SEC = 125.0

def turn_robot(angle):
    deg = np.degrees(angle)
    if deg > 0:
        arlo.go_diff(SPEED, SPEED, 0, 1)  
    else:
        arlo.go_diff(SPEED, SPEED, 1, 0)   
    time.sleep(abs(deg) / DEG_PER_SEC)
    arlo.stop()
    time.sleep(0.3)                      

def drive_robot(distance):
    """Drive the robot forward by distance [cm]."""
    arlo.go_diff(SPEED, SPEED + 2, 1, 1)
    time.sleep(distance / CM_PER_SEC)
    arlo.stop()
    time.sleep(0.3)

def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points. 
        p = particle.Particle(600.0*np.random.ranf() - 100.0, 600.0*np.random.ranf() - 250.0, np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi), 1.0/num_particles)
        particles.append(p)

    return particles

def move_all_particles(particles, distance, delta_theta, sigma, sigma_theta):
    for p in particles:
        particle.move_particle(p, distance, delta_theta)
    particle.add_uncertainty(particles, sigma, sigma_theta)

def expected_measurement(p, lx, ly):
    dx = lx - p.getX()                             
    dy = ly - p.getY()
    dist = np.hypot(dx, dy)                        
    direction = np.arctan2(dy, dx)                 
    angle = direction - p.getTheta()               
    angle = np.mod(angle + np.pi, 2*np.pi) - np.pi 
    return dist, angle



# Collect the camera's measurements

def wrap_angle(angle):
    """Hold en vinkel mellem -pi og pi."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def collect_measurements(objectIDs, dists, angles):
    """
    Behold kun landmark 1 og 3.
    Hvis samme ID ses flere gange, beregnes gennemsnittet.
    """
    measurements = {}

    if objectIDs is None:
        return measurements

    for marker_id, distance, angle in zip(objectIDs, dists, angles):
        marker_id = int(marker_id)

        # Ignorér alle andre markører
        if marker_id not in landmarkIDs:
            continue

        distance = float(distance)
        angle = float(angle)

        # Ignorér ugyldige og urealistiske kameramålinger.
        if not np.isfinite(distance) or not np.isfinite(angle):
            continue
        if distance <= 0.0 or distance > 500.0:
            continue

        if marker_id not in measurements:
            measurements[marker_id] = {
                "distances": [],
                "angles": []
            }

        measurements[marker_id]["distances"].append(distance)
        measurements[marker_id]["angles"].append(angle)

    result = {}

    for marker_id, values in measurements.items():
        mean_distance = np.mean(values["distances"])

        # Cirkulært gennemsnit, fordi vinkler kan ligge omkring -pi og pi
        mean_sin = np.mean(np.sin(values["angles"]))
        mean_cos = np.mean(np.cos(values["angles"]))
        mean_angle = np.arctan2(mean_sin, mean_cos)

        result[marker_id] = (mean_distance, mean_angle)

    return result



def update_particle_weights(
        particles,
        measurements,
        sigma_distance=25.0,
        sigma_angle=np.radians(10.0)):

    for p in particles:
        weight = 1.0

        for marker_id, measurement in measurements.items():
            measured_distance, measured_angle = measurement

            lx, ly = landmarks[marker_id]

            expected_distance, expected_angle = expected_measurement(
                p, lx, ly
            )

            distance_error = measured_distance - expected_distance
            angle_error = wrap_angle(measured_angle - expected_angle)

            distance_probability = np.exp(
                -0.5 * (distance_error / sigma_distance) ** 2
            )

            angle_probability = np.exp(
                -0.5 * (angle_error / sigma_angle) ** 2
            )

            weight *= distance_probability * angle_probability

        p.setWeight(weight)



def normalize_weights(particles):
    total_weight = sum(p.getWeight() for p in particles)

    if total_weight <= 1e-300:
        # Hvis alle vægte praktisk talt er nul
        uniform_weight = 1.0 / len(particles)

        for p in particles:
            p.setWeight(uniform_weight)

        return False

    for p in particles:
        p.setWeight(p.getWeight() / total_weight)

    return True




def resample_particles(particles):
    number_of_particles = len(particles)

    weights = np.array(
        [p.getWeight() for p in particles],
        dtype=float
    )

    cumulative_weights = np.cumsum(weights)
    cumulative_weights[-1] = 1.0

    # Systematic resampling
    start = np.random.uniform(0.0, 1.0 / number_of_particles)
    positions = start + np.arange(number_of_particles) / number_of_particles

    new_particles = []
    index = 0

    for position in positions:
        while position > cumulative_weights[index]:
            index += 1

        selected = particles[index]

        copied_particle = particle.Particle(
            selected.getX(),
            selected.getY(),
            selected.getTheta(),
            1.0 / number_of_particles
        )

        new_particles.append(copied_particle)

    return new_particles


def calculate_movement_to_goal(est_pose):
    # Midtpunkt mellem landmark 1 og landmark 3
    goal_x = (landmarks[1][0] + landmarks[3][0]) / 2.0
    goal_y = (landmarks[1][1] + landmarks[3][1]) / 2.0

    # Vektor fra robotten til målet
    dx = goal_x - est_pose.getX()
    dy = goal_y - est_pose.getY()

    # Retningen og afstanden til målet
    target_angle = np.arctan2(dy, dx)
    turn_angle = wrap_angle(target_angle - est_pose.getTheta())
    distance_to_goal = np.hypot(dx, dy)

    return goal_x, goal_y, turn_angle, distance_to_goal


    
# Main program #
arlo = None
cam = None

try:
    if showGUI:
        # Open windows
        WIN_RF1 = "Robot view"
        cv2.namedWindow(WIN_RF1)
        cv2.moveWindow(WIN_RF1, 50, 50)

        WIN_World = "World view"
        cv2.namedWindow(WIN_World)
        cv2.moveWindow(WIN_World, 500, 50)


    # Initialize particles
    num_particles = 1000
    particles = initialize_particles(num_particles)

    est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

    # Driving parameters
    velocity = 0.0 # cm/sec
    angular_velocity = 0.0 # radians/sec


    if isRunningOnArlo():
        arlo = robot.Robot()
     
    # Allocate space for world map
    world = np.zeros((500,500,3), dtype=np.uint8)

    # Draw map
    draw_world(est_pose, particles, world)

    print("Opening and initializing camera")
    if isRunningOnArlo():
        #cam = camera.Camera(0, robottype='arlo', useCaptureThread=True)
        cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
    else:
        cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=True)
        #cam = camera.Camera(1, robottype='macbookpro', useCaptureThread=False)
    
    last_time = timer()

    # Robotten drejer 30 grader pr. måling.
    # 24 målinger svarer til to hele omgange.
    scan_steps = 0
    minimum_scan_steps = 12
    seen_ids = set()
    has_driven = False

    while True:

        # Move the robot according to user input (only for testing)
        action = cv2.waitKey(10)
        if action == ord('q'): # Quit
            break
    
        if not isRunningOnArlo():
            if action == ord('w'): # Forward
                velocity += 4.0
            elif action == ord('x'): # Backwards
                velocity -= 4.0
            elif action == ord('s'): # Stop
                velocity = 0.0
                angular_velocity = 0.0
            elif action == ord('a'): # Left
                angular_velocity += 0.2
            elif action == ord('d'): # Right
                angular_velocity -= 0.2



        
        # Use motor controls to update particles
        if isRunningOnArlo():
            step = np.radians(30)
            turn_robot(step)
            move_all_particles(particles, 0.0, step, 2.0, np.radians(5))
            scan_steps += 1
        else:
            now = timer()
            dt = now - last_time
            last_time = now
            if velocity != 0.0 or angular_velocity != 0.0:
                move_all_particles(particles, velocity * dt, angular_velocity * dt,
                                   1.0, np.radians(1))

        # Fetch next frame
        colour = cam.get_next_frame()
        
        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)


        
        if objectIDs is not None:
            measurements = collect_measurements(objectIDs, dists, angles)
            seen_ids.update(measurements.keys())

            for marker_id, measurement in measurements.items():
                measured_distance, measured_angle = measurement
                print(
                    "Landmark ID =", marker_id,
                    "distance =", measured_distance,
                    "angle =", measured_angle
                )

            if measurements:
                update_particle_weights(particles, measurements)

                if normalize_weights(particles):
                    particles = resample_particles(particles)
    
        est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose


        #--
        
        goal_x, goal_y, turn_angle, distance_to_goal = \
            calculate_movement_to_goal(est_pose)

        print(
            "Goal:", goal_x, goal_y,
            "turn:", np.degrees(turn_angle), "degrees",
            "drive:", distance_to_goal, "cm"
        )

        #----
        if showGUI:
            # Draw map
            draw_world(est_pose, particles, world)
    
            # Show frame
            cv2.imshow(WIN_RF1, colour)

            # Show world
            cv2.imshow(WIN_World, world)

        print(
            "Estimated pose:",
            est_pose.getX(),
            est_pose.getY(),
            est_pose.getTheta()
        )

        # Efter mindst to omgange og observation af begge landmarks:
        # drej mod midtpunktet og kør derhen præcis én gang.
        if (
            isRunningOnArlo()
            and not has_driven
            and scan_steps >= minimum_scan_steps
            and {1, 3}.issubset(seen_ids)
        ):
            print("Localization completed")
            print("Turning:", np.degrees(turn_angle), "degrees")
            print("Driving:", distance_to_goal, "cm")

            turn_robot(turn_angle)
            move_all_particles(
                particles, 0.0, turn_angle, 2.0, np.radians(5)
            )

            drive_robot(distance_to_goal)
            move_all_particles(
                particles, distance_to_goal, 0.0, 3.0, np.radians(2)
            )

            has_driven = True
            print("Goal reached")
            break
    
  
finally: 
    # Make sure to clean up even if an exception occurred
    if isRunningOnArlo() and arlo is not None:
        arlo.stop()
    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    if cam is not None:
        cam.terminateCaptureThread()
