import robot
from time import sleep

arlo = robot.Robot()

print("Running ...")

leftSpeed = 128
rightSpeed = 128
print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))

sleep(3);

print(arlo.stop())
