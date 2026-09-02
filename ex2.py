import robot
from time import sleep

arlo = robot.Robot()

print("Running ...")

def circle():

    leftSpeed = 100
    rightSpeed = 80
    for i in range(4):
        print(arlo.go_diff(leftSpeed, rightSpeed/2, 1, 1))
        sleep(0.5)
    
    print(arlo.stop())

circle()
arlo.stop()