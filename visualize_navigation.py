"""
可视化导航规划 - 显示地图、起点、终点和规划路径（如果存在）
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.patches import Circle, Rectangle
import sys
import os
import platform

# 配置中文字体
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(os.path.join(os.path.dirname(__file__), 'controllers', 'simple_robot_controller'))

from occupancy_grid_map import OccupancyGridMap
from path_planner import PathPlanner

def visualize_with_navigation(map_file, start_pos, goal_pos):
    """
    可视化地图和导航点
    
    Args:
        map_file: 地图文件
        start_pos: 起点 (x, y)
        goal_pos: 终点 (x, y)
    """
    print("=" * 70)
    print("导航路径可视化")
    print("=" * 70)
    
    # 加载地图
    grid = np.load(map_file)
    print(f"\n✓ 地图加载: {os.path.basename(map_file)}")
    print(f"  尺寸: {grid.shape}")
    
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
    
    # 转换坐标
    start_grid = slam_map.world_to_grid(start_pos[0], start_pos[1])
    goal_grid = slam_map.world_to_grid(goal_pos[0], goal_pos[1])
    
    print(f"\n起点: 世界({start_pos[0]:.3f}, {start_pos[1]:.3f}) → 栅格({start_grid[0]}, {start_grid[1]})")
    print(f"终点: 世界({goal_pos[0]:.3f}, {goal_pos[1]:.3f}) → 栅格({goal_grid[0]}, {goal_grid[1]})")
    
    # 检查起点和终点状态
    start_occ = grid[start_grid[1], start_grid[0]] if slam_map.is_valid_cell(*start_grid) else -1
    goal_occ = grid[goal_grid[1], goal_grid[0]] if slam_map.is_valid_cell(*goal_grid) else -1
    
    print(f"\n起点状态: ", end='')
    if start_occ < 0:
        print("❌ 超出地图")
    elif start_occ < 0.4:
        print(f"✅ 自由空间 (占据率: {start_occ:.3f})")
    elif start_occ <= 0.6:
        print(f"⚠️ 未知区域 (占据率: {start_occ:.3f})")
    else:
        print(f"❌ 障碍物 (占据率: {start_occ:.3f})")
    
    print(f"终点状态: ", end='')
    if goal_occ < 0:
        print("❌ 超出地图")
    elif goal_occ < 0.4:
        print(f"✅ 自由空间 (占据率: {goal_occ:.3f})")
    elif goal_occ <= 0.6:
        print(f"⚠️ 未知区域 (占据率: {goal_occ:.3f})")
    else:
        print(f"❌ 障碍物 (占据率: {goal_occ:.3f})")
    
    # 尝试规划路径
    print(f"\n" + "=" * 70)
    print("尝试路径规划...")
    print("=" * 70)
    
    planner = PathPlanner(slam_map)
    path = planner.plan_path(start_pos[0], start_pos[1], goal_pos[0], goal_pos[1])
    
    if path:
        path_length = sum([
            np.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)
            for i in range(len(path) - 1)
        ])
        print(f"✅ 路径规划成功！")
        print(f"   路径点数: {len(path)}")
        print(f"   路径长度: {path_length:.2f} m")
        print(f"   直线距离: {np.sqrt((goal_pos[0]-start_pos[0])**2 + (goal_pos[1]-start_pos[1])**2):.2f} m")
    else:
        print(f"❌ 路径规划失败")
        print(f"   原因: 起点到终点之间被障碍物完全阻挡")
    
    # 创建可视化
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 显示地图
    extent = [slam_map.origin_x, slam_map.origin_x + slam_map.width,
              slam_map.origin_y, slam_map.origin_y + slam_map.height]
    im = ax.imshow(grid, cmap='gray_r', origin='lower', extent=extent, alpha=0.8)
    
    # 绘制栅格线
    for i in range(0, grid.shape[1], 10):
        x = slam_map.origin_x + i * resolution
        ax.axvline(x, color='gray', linewidth=0.3, alpha=0.3)
    for j in range(0, grid.shape[0], 10):
        y = slam_map.origin_y + j * resolution
        ax.axhline(y, color='gray', linewidth=0.3, alpha=0.3)
    
    # 绘制起点（大绿圈）
    start_circle = Circle(start_pos, 0.08, color='green', alpha=0.8, zorder=10, linewidth=3, fill=False)
    ax.add_patch(start_circle)
    ax.plot(start_pos[0], start_pos[1], 'g*', markersize=25, zorder=11)
    ax.text(start_pos[0], start_pos[1] + 0.15, '起点\nSTART', 
            ha='center', va='bottom', fontsize=13, fontweight='bold',
            color='green', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='green', linewidth=2))
    
    # 绘制终点（大红圈）
    goal_circle = Circle(goal_pos, 0.08, color='red', alpha=0.8, zorder=10, linewidth=3, fill=False)
    ax.add_patch(goal_circle)
    ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=25, zorder=11)
    ax.text(goal_pos[0], goal_pos[1] + 0.15, '出口\nEXIT', 
            ha='center', va='bottom', fontsize=13, fontweight='bold',
            color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='red', linewidth=2))
    
    # 如果有路径，绘制路径
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, 'b-', linewidth=4, label='规划路径', zorder=5, alpha=0.7)
        ax.plot(path_x, path_y, 'co', markersize=6, zorder=6, alpha=0.8)
        
        # 添加箭头
        for i in range(0, len(path) - 1, max(1, len(path) // 15)):
            dx = path_x[i+1] - path_x[i]
            dy = path_y[i+1] - path_y[i]
            ax.arrow(path_x[i], path_y[i], dx*0.8, dy*0.8,
                    head_width=0.05, head_length=0.03, fc='blue', ec='blue',
                    alpha=0.6, zorder=7, linewidth=1.5)
    else:
        # 绘制直线（表示无法直达）
        ax.plot([start_pos[0], goal_pos[0]], [start_pos[1], goal_pos[1]], 
               'r--', linewidth=2, label='直线路径（被阻挡）', alpha=0.5, zorder=3)
    
    # 设置
    ax.set_xlabel('X 坐标 (米)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y 坐标 (米)', fontsize=12, fontweight='bold')
    
    if path:
        title = f'✅ 导航路径规划成功\n路径长度: {path_length:.2f}m, 路径点数: {len(path)}'
        title_color = 'green'
    else:
        title = '❌ 路径被阻挡 - 需要清除障碍物'
        title_color = 'red'
    
    ax.set_title(title, fontsize=15, fontweight='bold', color=title_color, pad=20)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('占据概率\n(0=自由, 1=障碍)', rotation=270, labelpad=20, fontsize=10)
    
    # 添加说明文本
    info_text = (
        f"起点: ({start_pos[0]:.3f}, {start_pos[1]:.3f}) m\n"
        f"终点: ({goal_pos[0]:.3f}, {goal_pos[1]:.3f}) m\n"
        f"直线距离: {np.sqrt((goal_pos[0]-start_pos[0])**2 + (goal_pos[1]-start_pos[1])**2):.2f} m"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    plt.tight_layout()
    
    # 保存
    output_file = map_file.replace('.npy', '_navigation_visualization.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化已保存: {output_file}")
    
    plt.show()
    
    return path

def main():
    if len(sys.argv) < 2:
        print("使用方法: python visualize_navigation.py <地图文件.npy> [起点x] [起点y] [终点x] [终点y]")
        print("\n示例:")
        print("  python visualize_navigation.py slam_map.npy")
        print("  python visualize_navigation.py slam_map.npy 0 0 1.68 0.22")
        sys.exit(1)
    
    map_file = sys.argv[1]
    
    # 解析坐标
    if len(sys.argv) >= 6:
        start_x, start_y = float(sys.argv[2]), float(sys.argv[3])
        goal_x, goal_y = float(sys.argv[4]), float(sys.argv[5])
    else:
        start_x, start_y = 0.0, 0.0
        goal_x, goal_y = 1.68, 0.22
        print(f"使用默认坐标: 起点({start_x}, {start_y}), 终点({goal_x}, {goal_y})")
    
    path = visualize_with_navigation(map_file, (start_x, start_y), (goal_x, goal_y))
    
    print("\n" + "=" * 70)
    if path:
        print("✅ 路径规划成功！可以进行自动导航")
    else:
        print("❌ 路径被阻挡，需要：")
        print("   1. 使用 map_editor.py 清除阻挡的障碍物")
        print("   2. 或者选择其他可达的终点位置")
    print("=" * 70)

if __name__ == "__main__":
    main()
