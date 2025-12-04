"""
SLAM Map Optimizer
优化已建地图，清除误差障碍，恢复可行路线
"""

import numpy as np
import json
import sys
from scipy import ndimage
import matplotlib.pyplot as plt
from datetime import datetime

class MapOptimizer:
    def __init__(self, map_file):
        """
        初始化地图优化器
        
        Args:
            map_file: 地图文件路径 (.npy)
        """
        print("=" * 60)
        print("SLAM Map Optimizer - 地图优化工具")
        print("=" * 60)
        
        # 加载地图
        self.map_file = map_file
        self.grid = np.load(map_file)
        print(f"\n✓ 地图加载成功: {map_file}")
        print(f"  网格尺寸: {self.grid.shape}")
        
        # 加载元数据
        metadata_file = map_file.replace('.npy', '_metadata.json')
        try:
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            print(f"✓ 元数据加载成功")
            print(f"  分辨率: {self.metadata.get('resolution', 0.02)}m")
            print(f"  地图尺寸: {self.metadata.get('width', 4.0)}m × {self.metadata.get('height', 2.0)}m")
        except:
            print("⚠ 未找到元数据文件，使用默认参数")
            self.metadata = {
                'resolution': 0.02,
                'width': 4.0,
                'height': 2.0
            }
        
        # 保存原始地图
        self.original_grid = self.grid.copy()
        
        # 打印统计信息
        self._print_statistics("原始地图")
    
    def _print_statistics(self, label="地图"):
        """打印地图统计信息"""
        total = self.grid.size
        unknown = np.sum((self.grid >= 0.4) & (self.grid <= 0.6))
        free = np.sum(self.grid < 0.4)
        occupied = np.sum(self.grid > 0.6)
        
        print(f"\n{label}统计:")
        print(f"  总栅格数: {total}")
        print(f"  未知区域: {unknown} ({unknown/total*100:.1f}%)")
        print(f"  自由空间: {free} ({free/total*100:.1f}%)")
        print(f"  障碍物: {occupied} ({occupied/total*100:.1f}%)")
        print(f"  已探索率: {(total-unknown)/total*100:.1f}%")
    
    def optimize_pathways(self, erosion_size=1, remove_thin_walls=True, 
                         remove_isolated=True, min_obstacle_size=3):
        """
        优化可行路线，清除误差障碍
        
        Args:
            erosion_size: 腐蚀操作的结构元素大小（清除薄障碍）
            remove_thin_walls: 是否移除薄墙（单像素墙）
            remove_isolated: 是否移除孤立障碍点
            min_obstacle_size: 最小障碍物尺寸（小于此值的视为噪点）
        """
        print("\n" + "=" * 60)
        print("开始优化可行路线...")
        print("=" * 60)
        
        optimized = self.grid.copy()
        
        # 转换为二值图像
        binary_map = optimized > 0.6
        
        # 1. 移除孤立障碍点
        if remove_isolated:
            print(f"\n[步骤1] 移除孤立障碍点（连通域分析）...")
            labeled, num_features = ndimage.label(binary_map)
            
            removed_count = 0
            for i in range(1, num_features + 1):
                component = (labeled == i)
                size = np.sum(component)
                
                # 移除小于阈值的连通域
                if size < min_obstacle_size:
                    optimized[component] = 0.0  # 设为自由空间
                    removed_count += 1
            
            print(f"  ✓ 移除了 {removed_count} 个孤立障碍点")
            binary_map = optimized > 0.6
        
        # 2. 移除薄墙（单像素墙）
        if remove_thin_walls:
            print(f"\n[步骤2] 移除薄墙...")
            # 检测每个障碍物像素的8邻域
            thin_wall_mask = np.zeros_like(binary_map)
            
            for i in range(1, binary_map.shape[0] - 1):
                for j in range(1, binary_map.shape[1] - 1):
                    if binary_map[i, j]:
                        # 8邻域
                        neighbors = binary_map[i-1:i+2, j-1:j+2]
                        obstacle_neighbors = np.sum(neighbors) - 1  # 减去自己
                        
                        # 如果障碍物邻居很少（<=2），可能是薄墙
                        if obstacle_neighbors <= 2:
                            thin_wall_mask[i, j] = True
            
            removed = np.sum(thin_wall_mask)
            optimized[thin_wall_mask] = 0.0
            print(f"  ✓ 移除了 {removed} 个薄墙像素")
            binary_map = optimized > 0.6
        
        # 3. 腐蚀操作（清除薄障碍）
        if erosion_size > 0:
            print(f"\n[步骤3] 腐蚀操作（清除薄障碍，尺寸={erosion_size}）...")
            structure = np.ones((erosion_size*2+1, erosion_size*2+1))
            eroded = ndimage.binary_erosion(binary_map, structure=structure)
            
            removed = np.sum(binary_map) - np.sum(eroded)
            optimized[binary_map & ~eroded] = 0.0
            print(f"  ✓ 腐蚀了 {removed} 个边缘障碍像素")
            binary_map = optimized > 0.6
        
        # 4. 开运算（去除小的突出部分）
        print(f"\n[步骤4] 开运算（平滑障碍边缘）...")
        structure = np.ones((3, 3))
        opened = ndimage.binary_opening(binary_map, structure=structure)
        
        removed = np.sum(binary_map) - np.sum(opened)
        optimized[binary_map & ~opened] = 0.0
        print(f"  ✓ 平滑了 {removed} 个边缘突出像素")
        
        # 5. 膨胀主要墙壁（保持结构）
        print(f"\n[步骤5] 轻微膨胀主要墙壁（保持结构）...")
        binary_map = optimized > 0.6
        structure = np.ones((2, 2))
        dilated = ndimage.binary_dilation(binary_map, structure=structure)
        
        # 只在确定是自由空间的地方膨胀
        free_space = self.original_grid < 0.3
        safe_dilation = dilated & ~free_space
        
        added = np.sum(safe_dilation) - np.sum(binary_map)
        optimized[safe_dilation & ~binary_map] = 0.8
        print(f"  ✓ 膨胀了 {added} 个墙壁像素（仅在安全区域）")
        
        self.grid = optimized
        
        print("\n" + "=" * 60)
        print("✓ 优化完成！")
        print("=" * 60)
        
        self._print_statistics("优化后地图")
        
        # 统计改善
        original_occupied = np.sum(self.original_grid > 0.6)
        optimized_occupied = np.sum(self.grid > 0.6)
        cleared = original_occupied - optimized_occupied
        
        print(f"\n改善统计:")
        print(f"  原始障碍物: {original_occupied} 像素")
        print(f"  优化后障碍物: {optimized_occupied} 像素")
        print(f"  清除障碍物: {cleared} 像素 ({cleared/original_occupied*100:.1f}%)")
        
        if cleared > 0:
            print(f"  ✅ 成功清除了 {cleared} 个误差障碍，恢复了可行路线！")
        else:
            print(f"  ⚠ 未清除障碍物，可能地图质量已经很好")
    
    def widen_pathways(self, width=2):
        """
        拓宽通道，确保机器人可以通过
        
        Args:
            width: 腐蚀宽度（像素）
        """
        print(f"\n[额外优化] 拓宽通道（腐蚀 {width} 像素）...")
        
        binary_map = self.grid > 0.6
        structure = np.ones((width*2+1, width*2+1))
        eroded = ndimage.binary_erosion(binary_map, structure=structure)
        
        removed = np.sum(binary_map) - np.sum(eroded)
        self.grid[binary_map & ~eroded] = 0.0
        
        print(f"  ✓ 拓宽了通道，移除 {removed} 个边缘障碍")
    
    def save_optimized_map(self, suffix="_optimized"):
        """保存优化后的地图"""
        # 生成输出文件名
        output_file = self.map_file.replace('.npy', f'{suffix}.npy')
        output_metadata = output_file.replace('.npy', '_metadata.json')
        
        # 保存地图
        np.save(output_file, self.grid)
        print(f"\n✓ 优化后地图已保存: {output_file}")
        
        # 更新元数据
        self.metadata['optimized'] = True
        self.metadata['optimization_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(output_metadata, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"✓ 元数据已保存: {output_metadata}")
        
        return output_file
    
    def visualize_comparison(self, save_file=None):
        """可视化对比原始地图和优化后地图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 原始地图
        axes[0].imshow(self.original_grid, cmap='gray_r', origin='lower')
        axes[0].set_title('原始地图', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X (栅格)')
        axes[0].set_ylabel('Y (栅格)')
        axes[0].grid(True, alpha=0.3)
        
        # 优化后地图
        axes[1].imshow(self.grid, cmap='gray_r', origin='lower')
        axes[1].set_title('优化后地图', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X (栅格)')
        axes[1].set_ylabel('Y (栅格)')
        axes[1].grid(True, alpha=0.3)
        
        # 差异图（红色=被清除的障碍，绿色=新增的障碍）
        diff = np.zeros((*self.grid.shape, 3))
        diff[:, :, 1] = (self.original_grid > 0.6) & (self.grid <= 0.6)  # 绿色：清除的障碍
        diff[:, :, 0] = (self.original_grid <= 0.6) & (self.grid > 0.6)  # 红色：新增的障碍
        
        axes[2].imshow(diff, origin='lower')
        axes[2].set_title('差异图（绿色=清除障碍，红色=新增障碍）', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('X (栅格)')
        axes[2].set_ylabel('Y (栅格)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_file is None:
            save_file = self.map_file.replace('.npy', '_optimization_comparison.png')
        
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ 对比图已保存: {save_file}")
        
        plt.show()
        
        return save_file

def main():
    if len(sys.argv) < 2:
        print("使用方法: python map_optimizer.py <地图文件.npy> [选项]")
        print("\n选项:")
        print("  --erosion <尺寸>        腐蚀尺寸 (默认: 1)")
        print("  --min-size <尺寸>       最小障碍物尺寸 (默认: 3)")
        print("  --widen <宽度>          拓宽通道宽度 (默认: 0)")
        print("  --no-thin-walls         不移除薄墙")
        print("  --no-isolated           不移除孤立点")
        print("\n示例:")
        print("  python map_optimizer.py slam_map.npy")
        print("  python map_optimizer.py slam_map.npy --erosion 2 --widen 1")
        sys.exit(1)
    
    map_file = sys.argv[1]
    
    # 解析参数
    erosion_size = 1
    min_obstacle_size = 3
    widen_width = 0
    remove_thin_walls = True
    remove_isolated = True
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--erosion':
            erosion_size = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--min-size':
            min_obstacle_size = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--widen':
            widen_width = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--no-thin-walls':
            remove_thin_walls = False
            i += 1
        elif sys.argv[i] == '--no-isolated':
            remove_isolated = False
            i += 1
        else:
            print(f"未知选项: {sys.argv[i]}")
            i += 1
    
    # 创建优化器
    optimizer = MapOptimizer(map_file)
    
    # 优化地图
    optimizer.optimize_pathways(
        erosion_size=erosion_size,
        remove_thin_walls=remove_thin_walls,
        remove_isolated=remove_isolated,
        min_obstacle_size=min_obstacle_size
    )
    
    # 可选：拓宽通道
    if widen_width > 0:
        optimizer.widen_pathways(width=widen_width)
    
    # 保存优化后的地图
    output_file = optimizer.save_optimized_map()
    
    # 可视化对比
    optimizer.visualize_comparison()
    
    print("\n" + "=" * 60)
    print("✅ 所有优化完成！")
    print("=" * 60)
    print(f"\n优化后的地图可用于路径规划:")
    print(f"  {output_file}")
    print(f"\n使用 map_viewer.py 查看优化后的地图:")
    print(f"  python map_viewer.py {output_file}")

if __name__ == "__main__":
    main()
