import time
import robot

SPEED_L, SPEED_R = 64, 66
TURN_SPEED = 50
STRAIGHT_COUNTS = 600
TURN_COUNTS = 100

arlo = robot.Robot()


def move(power_l, power_r, left, right, target):
    arlo.reset_encoder_counts()
    time.sleep(0.05)
    arlo.go_diff(power_l, power_r, left, right)
    while True:
        l = abs(int(arlo.read_left_wheel_encoder()))
        r = abs(int(arlo.read_right_wheel_encoder()))
        if (l + r) / 2 >= target:
            break
    arlo.stop()
    time.sleep(0.5)


input("Test 1: straight. Press Enter to start...")
move(SPEED_L, SPEED_R, 1, 1, STRAIGHT_COUNTS)
meters = float(input("How many meters did it drive? "))
drift = float(input("How many cm did it drift sideways (left negative, right positive)? "))

input("Test 2: turn left. Press Enter to start...")
move(TURN_SPEED, TURN_SPEED, 0, 1, TURN_COUNTS)
left_deg = float(input("How many degrees did it turn? "))

input("Test 3: turn right. Press Enter to start...")
move(TURN_SPEED, TURN_SPEED, 1, 0, TURN_COUNTS)
right_deg = float(input("How many degrees did it turn? "))

arlo.stop()
print(f"COUNTS_PER_M = {STRAIGHT_COUNTS / meters:.1f}")
print(f"COUNTS_PER_DEG = {2 * TURN_COUNTS / (left_deg + right_deg):.3f}")
print(f"left {left_deg:.0f} deg vs right {right_deg:.0f} deg, drift {drift:.0f} cm")