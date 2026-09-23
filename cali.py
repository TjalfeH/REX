import time
import robot

SPEED_L, SPEED_R = 64, 66
TURN_SPEED = 50
STRAIGHT_COUNTS = 600
TURN_COUNTS = 100

arlo = robot.Robot()


def counts():
    raw_l = arlo.read_left_wheel_encoder()
    raw_r = arlo.read_right_wheel_encoder()
    try:
        return abs(int(raw_l)), abs(int(raw_r))
    except ValueError:
        print("Bad encoder reply:", raw_l, raw_r)
        return 0, 0


def move(power_l, power_r, left, right, target):
    arlo.reset_encoder_counts()
    time.sleep(0.05)
    start = time.time()
    arlo.go_diff(power_l, power_r, left, right)
    try:
        while True:
            l, r = counts()
            print(f"\rleft {l}  right {r}   ", end="")
            if (l + r) / 2 >= target:
                break
            if time.time() - start > 1 + target / 40:
                print("\nTIMEOUT: encoders are not counting")
                break
    finally:
        arlo.stop()
    print()
    time.sleep(1)
    return counts()


try:
    input("Test 1: straight. Mark the start position, then press Enter...")
    l, r = move(SPEED_L, SPEED_R, 1, 1, STRAIGHT_COUNTS)
    straight = (l + r) / 2
    print(f"encoder left {l}, right {r}")
    if straight < 10:
        print("Encoders are not counting. Run encTest.py and send the output.")
        raise SystemExit
    meters = float(input("How many meters did it drive? "))
    drift = float(input("Sideways drift in cm (left negative, right positive)? "))

    input("Test 2: turn left. Press Enter...")
    l, r = move(TURN_SPEED, TURN_SPEED, 0, 1, TURN_COUNTS)
    left_turn = (l + r) / 2
    left_deg = float(input("How many degrees did it turn? "))

    input("Test 3: turn right. Press Enter...")
    l, r = move(TURN_SPEED, TURN_SPEED, 1, 0, TURN_COUNTS)
    right_turn = (l + r) / 2
    right_deg = float(input("How many degrees did it turn? "))

    COUNTS_PER_M = straight / meters
    COUNTS_PER_DEG_L = left_turn / left_deg
    COUNTS_PER_DEG_R = right_turn / right_deg
    OVERSHOOT_DRIVE = straight - STRAIGHT_COUNTS
    OVERSHOOT_TURN = (left_turn + right_turn) / 2 - TURN_COUNTS

    print("\nPaste into rrtPlan.py:\n")
    print(f"COUNTS_PER_M = {COUNTS_PER_M:.1f}")
    print(f"COUNTS_PER_DEG_L = {COUNTS_PER_DEG_L:.3f}")
    print(f"COUNTS_PER_DEG_R = {COUNTS_PER_DEG_R:.3f}")
    print(f"OVERSHOOT_DRIVE = {OVERSHOOT_DRIVE:.0f}")
    print(f"OVERSHOOT_TURN = {OVERSHOOT_TURN:.0f}\n")

    if drift > 2:
        print(f"Drifts right: set SPEED_R = {SPEED_R + 1} and run cali again")
    elif drift < -2:
        print(f"Drifts left: set SPEED_R = {SPEED_R - 1} and run cali again")
    else:
        print("Drives straight")

    input("\nCheck: robot drives 1 m, then turns 90 left, then 90 right. Press Enter...")
    move(SPEED_L, SPEED_R, 1, 1, 1.0 * COUNTS_PER_M - OVERSHOOT_DRIVE)
    move(TURN_SPEED, TURN_SPEED, 0, 1, 90 * COUNTS_PER_DEG_L - OVERSHOOT_TURN)
    move(TURN_SPEED, TURN_SPEED, 1, 0, 90 * COUNTS_PER_DEG_R - OVERSHOOT_TURN)
finally:
    arlo.stop()