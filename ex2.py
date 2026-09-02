import robot
from time import sleep

arlo = robot.Robot()

print("Running ...")

def rightCircle():

    leftSpeed = 80
    rightSpeed = 40
    for i in range(4):
        print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
        sleep(2.3)
    
    print(arlo.stop())

def leftCircle():

    leftSpeed = 40
    rightSpeed = 80
    for i in range(4):
        print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
        sleep(2.3)
    
    print(arlo.stop())

for i in range(4):
    leftCircle()
    sleep(0.5)
    rightCircle()
    sleep(0.5)
arlo.stop()