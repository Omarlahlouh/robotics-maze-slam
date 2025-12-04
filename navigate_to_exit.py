"""
Navigate to Exit - 自动导航到出口
使用已保存的SLAM地图规划路径并导航到指定出口
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.patches import Circle, FancyArrowPatch
import sys
import os

# 添加控制器路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'controllers', 'simple_robot_controller'))

from occupancy_grid_map import OccupancyGridMap
from path_planner import PathPlanner

# 配置中文字体
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class NavigationVisualizer:
    def __init__(self, map_file, start_pos, goal_pos):
        """
        初始化导航可视化器
        
        Args:
            map_file: 地图文件路径
            start_pos: 起点坐标 (x, y)
            goal_pos: 目标坐标 (x, y)
        """
        print("=" * 70)
        print("自动导航到出口 - Navigate to Exit")
        print("=" * 70)
        
        # 加载地图
        self.grid = np.load(map_file)
        print(f"\n✓ 地图加载成功: {map_file}")
        print(f"  网格尺寸: {self.grid.shape}")
        
        # 加载元数据
        metadata_file = map_file.replace('.npy', '_metadata.json')
        try:
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            print(f"✓ 元数据加载成功")
            resolution = self.metadata.get('resolution', 0.02)
            width = self.metadata.get('width', 4.0)
            height = self.metadata.get('height', 2.0)
        except:
            print("⚠ 未找到元数据文件，使用默认参数")
            resolution = 0.02
            width = 4.0
            height = 2.0
        
        # 创建地图对象
        self.slam_map = OccupancyGridMap(width=width, height=height, resolution=resolution)
        self.slam_map.grid = self.grid
        
        # 设置起点和终点
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        
        print(f"\n起点位置: ({start_pos[0]:.3f}, {start_pos[1]:.3f}) m")
        print(f"出口位置: ({goal_pos[0]:.3f}, {goal_pos[1]:.3f}) m")
        
        # 路径规划
        self.planner = PathPlanner(self.slam_map)
        self.path = None
        
    def plan_path(self):
        """规划路径"""
        print("\n" + "=" * 70)
        print("开始路径规划...")
        print("=" * 70)
        
        # 使用A*算法规划路径
        self.path = self.planner.plan_path(
            self.start_pos[0], self.start_pos[1],
            self.goal_pos[0], self.goal_pos[1]
        )
        
        if self.path is None:
            print("\n❌ 路径规划失败！可能原因：")
            print("  1. 起点或终点在障碍物上")
            print("  2. 起点和终点之间被障碍物完全阻挡")
            print("  3. 地图质量问题")
            print("\n建议：")
            print("  - 使用 map_editor.py 手动清除阻挡路径的障碍物")
            print("  - 检查起点和终点坐标是否正确")
            return False
        
        # 计算路径长度
        path_length = 0.0
        for i in range(len(self.path) - 1):
            dx = self.path[i+1][0] - self.path[i][0]
            dy = self.path[i+1][1] - self.path[i][1]
            path_length += np.sqrt(dx**2 + dy**2)
        
        print(f"\n✓ 路径规划成功！")
        print(f"  路径点数: {len(self.path)}")
        print(f"  路径长度: {path_length:.2f} m")
        print(f"  直线距离: {np.sqrt((self.goal_pos[0]-self.start_pos[0])**2 + (self.goal_pos[1]-self.start_pos[1])**2):.2f} m")
        
        # 打印路径
        print(f"\n路径详情（前5个点）:")
        for i, (x, y) in enumerate(self.path[:5]):
            print(f"  点 {i+1}: ({x:.3f}, {y:.3f}) m")
        if len(self.path) > 5:
            print(f"  ... （共{len(self.path)}个点）")
            x, y = self.path[-1]
            print(f"  点 {len(self.path)}: ({x:.3f}, {y:.3f}) m [终点]")
        
        return True
    
    def visualize(self, save_file=None):
        """可视化地图和路径"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # 显示地图
        im = ax.imshow(self.grid, cmap='gray_r', origin='lower', extent=[
            self.slam_map.origin_x,
            self.slam_map.origin_x + self.slam_map.width,
            self.slam_map.origin_y,
            self.slam_map.origin_y + self.slam_map.height
        ])
        
        # 绘制起点（绿色圆圈）
        start_circle = Circle(self.start_pos, 0.05, color='green', alpha=0.7, zorder=10)
        ax.add_patch(start_circle)
        ax.text(self.start_pos[0], self.start_pos[1] + 0.1, '起点', 
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                color='green', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 绘制终点（红色星形）
        ax.plot(self.goal_pos[0], self.goal_pos[1], 'r*', markersize=20, zorder=10)
        ax.text(self.goal_pos[0], self.goal_pos[1] + 0.1, '出口', 
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 绘制路径
        if self.path:
            path_x = [p[0] for p in self.path]
            path_y = [p[1] for p in self.path]
            ax.plot(path_x, path_y, 'b-', linewidth=3, label='规划路径', zorder=5, alpha=0.7)
            
            # 绘制路径点
            ax.plot(path_x, path_y, 'co', markersize=4, zorder=6, alpha=0.6)
            
            # 绘制箭头指示方向
            for i in range(0, len(self.path) - 1, max(1, len(self.path) // 10)):
                arrow = FancyArrowPatch(
                    (path_x[i], path_y[i]), 
                    (path_x[i+1], path_y[i+1]),
                    arrowstyle='->', mutation_scale=20, linewidth=2,
                    color='blue', alpha=0.6, zorder=7
                )
                ax.add_patch(arrow)
        
        ax.set_xlabel('X (米)', fontsize=12)
        ax.set_ylabel('Y (米)', fontsize=12)
        ax.set_title('自动导航路径规划 - 从起点到出口', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('占据概率 (0=自由, 1=障碍)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"\n✓ 路径规划图已保存: {save_file}")
        
        plt.show()
        
    def export_path_for_robot(self, output_file='navigation_path.json'):
        """导出路径供机器人使用"""
        if not self.path:
            print("❌ 没有可导出的路径")
            return
        
        path_data = {
            'start': {
                'x': float(self.start_pos[0]),
                'y': float(self.start_pos[1])
            },
            'goal': {
                'x': float(self.goal_pos[0]),
                'y': float(self.goal_pos[1])
            },
            'path': [
                {'x': float(x), 'y': float(y), 'index': i}
                for i, (x, y) in enumerate(self.path)
            ],
            'path_length': sum([
                np.sqrt((self.path[i+1][0] - self.path[i][0])**2 + 
                       (self.path[i+1][1] - self.path[i][1])**2)
                for i in range(len(self.path) - 1)
            ]),
            'num_waypoints': len(self.path)
        }
        
        with open(output_file, 'w') as f:
            json.dump(path_data, f, indent=2)
        
        print(f"\n✓ 路径数据已导出: {output_file}")
        print(f"  可以在 simple_robot_controller.py 中加载此路径")

def main():
    print("=" * 70)
    print("自动导航到出口 - Navigate to Exit")
    print("=" * 70)
    
    if len(sys.argv) < 2:
        print("\n使用方法:")
        print("  python navigate_to_exit.py <地图文件.npy> [起点x] [起点y] [出口x] [出口y]")
        print("\n示例:")
        print("  python navigate_to_exit.py slam_map_edited.npy 0 0 1.75 0.20")
        print("  python navigate_to_exit.py slam_map_edited.npy")
        print("\n参数:")
        print("  地图文件: 已保存的SLAM地图(.npy)")
        print("  起点坐标: 起点的世界坐标(x, y)，默认(0, 0)")
        print("  出口坐标: 出口的世界坐标(x, y)，默认(1.75, 0.20)")
        print("=" * 70)
        sys.exit(1)
    
    map_file = sys.argv[1]
    
    # 解析坐标参数
    if len(sys.argv) >= 6:
        start_x = float(sys.argv[2])
        start_y = float(sys.argv[3])
        goal_x = float(sys.argv[4])
        goal_y = float(sys.argv[5])
    else:
        # 默认值：用户指定的坐标
        start_x, start_y = 0.0, 0.0  # 起点
        goal_x, goal_y = 1.75, 0.20  # 出口
        print(f"\n使用默认坐标:")
        print(f"  起点: ({start_x}, {start_y})")
        print(f"  出口: ({goal_x}, {goal_y})")
    
    # 创建导航可视化器
    nav_viz = NavigationVisualizer(
        map_file=map_file,
        start_pos=(start_x, start_y),
        goal_pos=(goal_x, goal_y)
    )
    
    # 规划路径
    success = nav_viz.plan_path()
    
    if not success:
        print("\n" + "=" * 70)
        print("❌ 导航规划失败")
        print("=" * 70)
        sys.exit(1)
    
    # 导出路径
    output_dir = os.path.dirname(map_file)
    path_file = os.path.join(output_dir, 'navigation_path.json')
    nav_viz.export_path_for_robot(path_file)
    
    # 可视化
    viz_file = map_file.replace('.npy', '_navigation_plan.png')
    nav_viz.visualize(save_file=viz_file)
    
    print("\n" + "=" * 70)
    print("✅ 导航规划完成！")
    print("=" * 70)
    print("\n下一步:")
    print("  1. 在Webots中启动仿真")
    print("  2. 按键'3'切换到自动导航模式")
    print("  3. 机器人将自动沿规划路径导航到出口")
    print("\n路径文件:")
    print(f"  {path_file}")
    print(f"  {viz_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
