import numpy as np

print("=== テスト1: 自然長のとき F_spring がゼロになるか ===")

SPRING_CONSTANT = 0.6
natural_len = 5.0

# 3点が等間隔の直線上（セグメント長 = 自然長 ぴったり）
left    = np.array([0.0, 0.0])
current = np.array([5.0, 0.0])
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

print(f"  dist_left  = {dist_left}  (期待値: {natural_len})")
print(f"  dist_right = {dist_right}  (期待値: {natural_len})")
print(f"  F_left     = {F_left}")
print(f"  F_right    = {F_right}")
print(f"  F_spring   = {F_spring}  (期待値: [0. 0.])")

if np.allclose(F_spring, [0, 0], atol=1e-6):
    print("  → PASS ✓")
else:
    print("  → FAIL ✗ (ゼロにならない)")
