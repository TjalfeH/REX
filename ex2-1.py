import robot
from time import sleep

arlo = robot.Robot()

print("Running ...")
def driveForward():
    leftSpeed = 40
    rightSpeed = 40
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))

    sleep(0.5)
    print(arlo.stop())

for i in range(4):
    driveForward()
    print(arlo.read_front_ping_sensor())
    sleep(0.5)
arlo.stop()