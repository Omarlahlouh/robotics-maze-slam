#!/usr/bin/env python3
"""验证路径文件的Y坐标是否正确"""

import json

# 加载路径文件
with open("controllers/simple_robot_controller/navigation_path.json", "r") as f:
    data = json.load(f)

print("=" * 60)
print("路径文件验证")
print("=" * 60)

# 验证起点
start = data["start"]
print(f"\n起点坐标:")
print(f"  X = {start['x']:.3f}")
print(f"  Y = {start['y']:.3f}")

if abs(start['y'] - 0.1) < 0.001:
    print("  ✅ 起点Y坐标正确 (0.1)")
else:
    print(f"  ❌ 起点Y坐标错误！应该是0.1，实际是{start['y']:.3f}")

# 验证前5个路径点
print(f"\n前5个路径点:")
for i, point in enumerate(data["path"][:5]):
    print(f"  点{i+1}: ({point['x']:.3f}, {point['y']:.3f})")
    if i == 1:  # 第二个点应该在Y=0.74附近
        if abs(point['y'] - 0.74) < 0.05:
            print("    ✅ 第2个点Y坐标正确 (约0.74)")
        else:
            print(f"    ❌ 第2个点Y坐标错误！应该约0.74，实际是{point['y']:.3f}")

# 检查是否有Y坐标接近0.02的点（这是错误的）
wrong_points = [p for p in data["path"] if abs(p['y'] - 0.02) < 0.05]
if wrong_points:
    print(f"\n❌ 发现 {len(wrong_points)} 个错误的点（Y≈0.02）！")
    print("   这些点来自旧的路径文件，需要重新生成！")
else:
    print("\n✅ 没有发现错误的Y坐标点")

print("\n" + "=" * 60)
print("Grid偏移计算:")
print("=" * 60)
print(f"起点Y (世界坐标): {start['y']:.3f} m")
print(f"起点Y (Grid坐标): {(start['y'] - (-1.0)) / 0.02:.1f}")
print(f"预期Grid Y: 55.0 (对应世界坐标0.1)")
print(f"如果Grid Y = 50.0，则世界Y = 0.0 (错误)")
print("=" * 60)
