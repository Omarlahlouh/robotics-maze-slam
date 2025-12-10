#!/usr/bin/env python3
"""Verify whether the Y-coordinate of the path file is correct"""

import json

# Load the path file
with open("controllers/simple_robot_controller/navigation_path.json", "r") as f:
    data = json.load(f)

print("=" * 60)
print("Path file verification")
print("=" * 60)

# Verification start
start = data["start"]
print(f"\nStart coordinate:")
print(f"  X = {start['x']:.3f}")
print(f"  Y = {start['y']:.3f}")

if abs(start['y'] - 0.1) < 0.001:
    print("The Y-coordinate of the start is correct (0.1)")
else:
    print(f"The Y-coordinate of the starting point is incorrect! It should be 0.1, but actually it is{start['y']:.3f}")

# Verify the first 5 path points
print(f"\nThe first 5 path points:")
for i, point in enumerate(data["path"][:5]):
    print(f"Point{i+1}: ({point['x']:.3f}, {point['y']:.3f})")
    if i == 1:  # The second point should be around Y=0.74
        if abs(point['y'] - 0.74) < 0.05:
            print("The y-coordinate of the second point is correct (约0.74)")
        else:
            print(f"The y-coordinate of the second point is incorrect! It should be about 0.74. Actually, it is{point['y']:.3f}")

# Check if there are any points with y-coordinates close to 0.02 (this is incorrect).
wrong_points = [p for p in data["path"] if abs(p['y'] - 0.02) < 0.05]
if wrong_points:
    print(f"\n {len(wrong_points)} incorrect points (Y≈0.02) were found!")
    print("These points come from the old path file and need to be regenerated!")
else:
    print("\nNo incorrect Y-coordinate points were found")

print("\n" + "=" * 60)
print("Grid offset calculation:")
print("=" * 60)
print(f"Start Y (World coordinates): {start['y']:.3f} m")
print(f"Start Y (Grid coordinates): {(start['y'] - (-1.0)) / 0.02:.1f}")
print(f"Expected Grid Y: 55.0 (Corresponding world coordinates0.1)")
print(f"If Grid Y = 50.0，world Y = 0.0 (incorrect)")
print("=" * 60)
