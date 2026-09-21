import time

import robot

SPEED = 64
SECONDS = 2

arlo = robot.Robot()


def drive(left, right, seconds):
    arlo.go_diff(SPEED, SPEED, left, right)
    time.sleep(seconds)
    arlo.stop()
    time.sleep(0.5)


input("Test 1: geradeaus. Enter druecken zum Starten...")
drive(1, 1, SECONDS)
meters = float(input("Wie viele Meter ist er gefahren? "))

input("Test 2: rechts drehen. Enter druecken zum Starten...")
drive(1, 0, SECONDS)
degrees = float(input("Wie viele Grad hat er sich gedreht? "))

arlo.stop()
print(f"M_PER_SEC = {meters / SECONDS:.3f}")
print(f"DEG_PER_SEC = {degrees / SECONDS:.1f}")