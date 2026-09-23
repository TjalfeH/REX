import time
import robot

SPEED_L, SPEED_R = 64, 66
TURN_SPEED = 64
SECONDS = 1.0

arlo = robot.Robot()


def drive(power_l, power_r, left, right, seconds):
    try:
        arlo.go_diff(power_l, power_r, left, right)
        time.sleep(seconds)
    finally:
        arlo.stop()
    time.sleep(0.5)


input("Test 1: drive straight for 1 second. Press Enter...")
drive(SPEED_L, SPEED_R, 1, 1, SECONDS)
meters = float(input("How many meters did it drive? "))

input("Test 2: turn for 1 second. Press Enter...")
drive(TURN_SPEED, TURN_SPEED, 0, 1, SECONDS)
degrees = float(input("How many degrees did it turn? "))

print(f"M_PER_SEC = {meters / SECONDS:.3f}")
print(f"DEG_PER_SEC = {degrees / SECONDS:.1f}")