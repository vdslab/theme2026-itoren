import numpy as np

print("=== Test 2: displaced midpoint pulls back toward line ===")

SPRING_CONSTANT = 0.6
natural_len = 5.0

# midpoint displaced upward from the line
left    = np.array([0.0, 0.0])
current = np.array([5.0, 3.0])  # displaced up
right   = np.array([10.0, 0.0])

vec_left  = left - current
dist_left = np.linalg.norm(vec_left)
dir_left  = vec_left / (dist_left + 1e-9)
F_left    = (dist_left - natural_len) * dir_left

vec_right  = right - current
dist_right = np.linalg.norm(vec_right)
dir_right  = vec_right / (dist_right + 1e-9)
F_right    = (dist_right - natural_len) * dir_right

F_spring = SPRING_CONSTANT * (F_left + F_right)

print(f"  dist_left  = {dist_left:.4f}  (natural: {natural_len}  -> stretched: {dist_left > natural_len})")
print(f"  dist_right = {dist_right:.4f}  (natural: {natural_len}  -> stretched: {dist_right > natural_len})")
print(f"  F_spring   = [{F_spring[0]:.4f}, {F_spring[1]:.4f}]")
print(f"  F_spring y = {F_spring[1]:.4f}  (expected: negative = downward)")

if F_spring[1] < 0:
    print("  -> PASS: spring pulls midpoint back down toward line")
else:
    print("  -> FAIL: spring pushes midpoint further up (inverted spring)")
