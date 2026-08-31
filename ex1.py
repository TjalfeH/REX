import robot
from time import sleep

arlo = robot.Robot()

print("Running ...")


def turnRight():
    leftSpeed = 85
    rightSpeed = 85
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))

    sleep(0.6)
    print(arlo.stop())

def driveForward():
    leftSpeed = 85
    rightSpeed = 85
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))

    sleep(0.6)
    print(arlo.stop())

def square():
    for i in range(4):
        driveForward()
        turnRight()
square()
arlo.stop()
