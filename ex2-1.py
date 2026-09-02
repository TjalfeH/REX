import robot
from time import sleep
from time import perf_counter

arlo = robot.Robot()

print("Running ...")
def driveForward():
    leftSpeed = 40
    rightSpeed = 40
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))

def rightTurn():

    leftSpeed = 40
    rightSpeed = 40
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
    sleep(1.2)

def leftTurn():

    leftSpeed = 40
    rightSpeed = 40
    print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
    sleep(1.2)

def detectLeft():
    lowest = 0
    while lowest < 400: 
        driveForward()
        lowest = 9999
        for i in range(5):
            if arlo.read_left_ping_sensor() < lowest:
                lowest = arlo.read_left_ping_sensor()
    

def avoidMainSensor():
    rightTurn()
    start = perf_counter()
    detectLeft()
    driveForward()
    sleep(3)
    print(arlo.stop())
    end = perf_counter()
    leftTurn()
    detectLeft()
    driveForward()
    sleep(3)
    print(arlo.stop())
    leftTurn()
    driveForward()
    sleep(round(end-start, 2))
    rightTurn()
    driveForward()


for i in range(4):
    while arlo.read_front_ping_sensor() > 200:
        driveForward()
        print(arlo.read_front_ping_sensor())
        print(arlo.read_left_ping_sensor())
        print(arlo.read_right_ping_sensor())
    arlo.stop()
    avoidMainSensor()



