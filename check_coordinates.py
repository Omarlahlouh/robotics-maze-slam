"""
坐标检查工具 - 检查指定世界坐标在地图中的状态
"""

import numpy as np
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'controllers', 'simple_robot_controller'))
from occupancy_grid_map import OccupancyGridMap

def check_coordinate(map_file, world_x, world_y):
    """检查世界坐标在地图中的状态"""
    
    # 加载地图
    grid = np.load(map_file)
    print(f"✓ 地图加载成功: {map_file}")
    print(f"  网格尺寸: {grid.shape}")
    
    # 加载元数据
    metadata_file = map_file.replace('.npy', '_metadata.json')
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        resolution = metadata.get('resolution', 0.02)
        width = metadata.get('width', 4.0)
        height = metadata.get('height', 2.0)
    except:
        resolution = 0.02
        width = 4.0
        height = 2.0
    
    # 创建地图对象
    slam_map = OccupancyGridMap(width=width, height=height, resolution=resolution)
    slam_map.grid = grid
    
    # 转换为栅格坐标
    grid_x, grid_y = slam_map.world_to_grid(world_x, world_y)
    
    print(f"\n坐标检查:")
    print(f"  世界坐标: ({world_x:.3f}, {world_y:.3f}) m")
    print(f"  栅格坐标: ({grid_x}, {grid_y})")
    
    # 检查是否在地图范围内
    if not slam_map.is_valid_cell(grid_x, grid_y):
        print(f"  ❌ 坐标超出地图范围！")
        print(f"     地图范围: X[{slam_map.origin_x:.2f}, {slam_map.origin_x+slam_map.width:.2f}], "
              f"Y[{slam_map.origin_y:.2f}, {slam_map.origin_y+slam_map.height:.2f}]")
        return
    
    # 获取占据值
    occupancy = grid[grid_y, grid_x]
    
    print(f"  占据概率: {occupancy:.3f}")
    
    if occupancy < 0.4:
        print(f"  ✅ 自由空间（可通行）")
    elif occupancy <= 0.6:
        print(f"  ⚠️  未知区域")
    else:
        print(f"  ❌ 障碍物（不可通行）")
    
    # 检查周围区域
    print(f"\n周围区域检查（3x3）:")
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            nx, ny = grid_x + dx, grid_y + dy
            if slam_map.is_valid_cell(nx, ny):
                val = grid[ny, nx]
                if val < 0.4:
                    symbol = '🟢'
                elif val <= 0.6:
                    symbol = '🟡'
                else:
                    symbol = '🔴'
                print(f"  ({nx}, {ny}): {val:.2f} {symbol}", end='')
                if dx == 0 and dy == 0:
                    print(" ← 目标", end='')
                print()

def main():
    if len(sys.argv) < 4:
        print("使用方法: python check_coordinates.py <地图文件.npy> <x> <y>")
        print("\n示例:")
        print("  python check_coordinates.py slam_map.npy 1.75 0.20")
        sys.exit(1)
    
    map_file = sys.argv[1]
    world_x = float(sys.argv[2])
    world_y = float(sys.argv[3])
    
    check_coordinate(map_file, world_x, world_y)

if __name__ == "__main__":
    main()
