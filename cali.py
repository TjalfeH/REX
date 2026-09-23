import time

import robot

SPEED = 64
SECONDS = 1.0

arlo = robot.Robot()


def drive(left, right, seconds):
    arlo.go_diff(SPEED, SPEED, left, right)
    time.sleep(seconds)
    arlo.stop()
    time.sleep(0.5)


input("Test 1: straight ahead. Press Enter to start...")
drive(1, 1, SECONDS)
meters = float(input("How many meters did it drive? "))

input("Test 2: turn right. Press Enter to start...")
drive(1, 0, SECONDS)
degrees = float(input("How many degrees did it turn? "))

arlo.stop()
print(f"M_PER_SEC = {meters / SECONDS:.3f}")
print(f"DEG_PER_SEC = {degrees / SECONDS:.1f}")