"""
Interactive SLAM Map Editor
交互式SLAM地图编辑器 - 手动编辑地图障碍物
"""

import numpy as np
import json
import sys
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button, Slider, RadioButtons
from datetime import datetime
import platform

# 配置中文字体支持
def setup_chinese_font():
    """设置matplotlib中文字体"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    print("✓ 中文字体已配置")

# 初始化中文字体
setup_chinese_font()

class InteractiveMapEditor:
    def __init__(self, map_file):
        """
        初始化交互式地图编辑器
        
        Args:
            map_file: 地图文件路径 (.npy)
        """
        print("=" * 70)
        print("交互式SLAM地图编辑器 - Interactive Map Editor")
        print("=" * 70)
        
        # 加载地图
        self.map_file = map_file
        self.grid = np.load(map_file)
        self.original_grid = self.grid.copy()
        
        print(f"\n✓ 地图加载成功: {map_file}")
        print(f"  网格尺寸: {self.grid.shape}")
        
        # 加载元数据
        metadata_file = map_file.replace('.npy', '_metadata.json')
        try:
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            print(f"✓ 元数据加载成功")
        except:
            print("⚠ 未找到元数据文件，使用默认参数")
            self.metadata = {
                'resolution': 0.02,
                'width': 4.0,
                'height': 2.0
            }
        
        # 编辑模式
        self.mode = 'erase'  # 'erase' or 'draw' or 'free'
        self.brush_size = 3
        self.drawing = False
        
        # 历史记录（用于撤销）
        self.history = [self.grid.copy()]
        self.history_index = 0
        self.max_history = 50
        
        print("\n" + "=" * 70)
        print("使用说明:")
        print("=" * 70)
        print("🖱️  鼠标左键 + 拖拽：擦除/绘制障碍物")
        print("🖱️  鼠标右键 + 拖拽：设置为自由空间")
        print("📏 笔刷大小滑块：调整笔刷大小")
        print("🔘 模式选择：擦除障碍/绘制障碍/设置自由")
        print("↩️  撤销按钮：撤销上一步操作")
        print("↪️  重做按钮：重做操作")
        print("🔄 重置按钮：恢复到原始地图")
        print("💾 保存按钮：保存编辑后的地图")
        print("=" * 70)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """设置用户界面"""
        # 创建图形窗口
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title('Interactive Map Editor - 交互式地图编辑器')
        
        # 主地图显示区域
        self.ax_main = plt.subplot2grid((10, 12), (0, 0), colspan=10, rowspan=10)
        self.im = self.ax_main.imshow(self.grid, cmap='gray_r', origin='lower', 
                                      interpolation='nearest', vmin=0, vmax=1)
        self.ax_main.set_title('SLAM Map - 点击拖拽编辑', fontsize=14, fontweight='bold')
        self.ax_main.set_xlabel('X (栅格)')
        self.ax_main.set_ylabel('Y (栅格)')
        self.ax_main.grid(True, alpha=0.3, linestyle='--')
        
        # 添加颜色条
        cbar = plt.colorbar(self.im, ax=self.ax_main)
        cbar.set_label('Occupancy (0=自由, 1=障碍)', rotation=270, labelpad=20)
        
        # 笔刷大小滑块
        ax_brush = plt.subplot2grid((10, 12), (0, 10), colspan=2, rowspan=1)
        self.slider_brush = Slider(ax_brush, '笔刷', 1, 20, valinit=3, valstep=1)
        self.slider_brush.on_changed(self._update_brush_size)
        
        # 模式选择按钮
        ax_mode = plt.subplot2grid((10, 12), (1, 10), colspan=2, rowspan=3)
        self.radio_mode = RadioButtons(ax_mode, ('擦除障碍', '绘制障碍', '设置自由'), active=0)
        self.radio_mode.on_clicked(self._change_mode)
        
        # 撤销按钮
        ax_undo = plt.subplot2grid((10, 12), (5, 10), colspan=2, rowspan=1)
        self.btn_undo = Button(ax_undo, '↩️ 撤销 (Undo)')
        self.btn_undo.on_clicked(self._undo)
        
        # 重做按钮
        ax_redo = plt.subplot2grid((10, 12), (6, 10), colspan=2, rowspan=1)
        self.btn_redo = Button(ax_redo, '↪️ 重做 (Redo)')
        self.btn_redo.on_clicked(self._redo)
        
        # 重置按钮
        ax_reset = plt.subplot2grid((10, 12), (7, 10), colspan=2, rowspan=1)
        self.btn_reset = Button(ax_reset, '🔄 重置 (Reset)')
        self.btn_reset.on_clicked(self._reset)
        
        # 保存按钮
        ax_save = plt.subplot2grid((10, 12), (8, 10), colspan=2, rowspan=1)
        self.btn_save = Button(ax_save, '💾 保存 (Save)')
        self.btn_save.on_clicked(self._save)
        
        # 统计信息显示
        ax_stats = plt.subplot2grid((10, 12), (9, 10), colspan=2, rowspan=1)
        ax_stats.axis('off')
        self.text_stats = ax_stats.text(0.1, 0.5, '', fontsize=9, 
                                        verticalalignment='center')
        self._update_stats()
        
        # 连接鼠标事件
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        
        plt.tight_layout()
    
    def _update_brush_size(self, val):
        """更新笔刷大小"""
        self.brush_size = int(val)
    
    def _change_mode(self, label):
        """改变编辑模式"""
        if label == '擦除障碍':
            self.mode = 'erase'
        elif label == '绘制障碍':
            self.mode = 'draw'
        elif label == '设置自由':
            self.mode = 'free'
        print(f"✓ 模式切换为: {label}")
    
    def _on_mouse_press(self, event):
        """鼠标按下事件"""
        if event.inaxes == self.ax_main:
            self.drawing = True
            self._edit_at_position(event.xdata, event.ydata, event.button)
    
    def _on_mouse_release(self, event):
        """鼠标释放事件"""
        if self.drawing:
            self.drawing = False
            # 保存到历史记录
            self._save_to_history()
    
    def _on_mouse_move(self, event):
        """鼠标移动事件"""
        if self.drawing and event.inaxes == self.ax_main:
            self._edit_at_position(event.xdata, event.ydata, event.button)
    
    def _edit_at_position(self, x, y, button):
        """在指定位置编辑地图"""
        if x is None or y is None:
            return
        
        # 转换为栅格坐标
        grid_x = int(round(x))
        grid_y = int(round(y))
        
        # 检查边界
        if not (0 <= grid_x < self.grid.shape[1] and 0 <= grid_y < self.grid.shape[0]):
            return
        
        # 确定编辑值
        if button == MouseButton.LEFT:
            if self.mode == 'erase':
                value = 0.0  # 设置为自由空间
            elif self.mode == 'draw':
                value = 0.9  # 设置为障碍物
            elif self.mode == 'free':
                value = 0.0  # 设置为自由空间
        elif button == MouseButton.RIGHT:
            value = 0.0  # 右键总是擦除
        else:
            return
        
        # 应用笔刷
        half_size = self.brush_size // 2
        for dy in range(-half_size, half_size + 1):
            for dx in range(-half_size, half_size + 1):
                # 圆形笔刷
                if dx*dx + dy*dy <= half_size*half_size:
                    ny, nx = grid_y + dy, grid_x + dx
                    if 0 <= ny < self.grid.shape[0] and 0 <= nx < self.grid.shape[1]:
                        self.grid[ny, nx] = value
        
        # 更新显示
        self.im.set_data(self.grid)
        self._update_stats()
        self.fig.canvas.draw_idle()
    
    def _save_to_history(self):
        """保存当前状态到历史记录"""
        # 删除当前位置之后的历史
        self.history = self.history[:self.history_index + 1]
        
        # 添加新状态
        self.history.append(self.grid.copy())
        self.history_index += 1
        
        # 限制历史记录数量
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.history_index -= 1
    
    def _undo(self, event):
        """撤销操作"""
        if self.history_index > 0:
            self.history_index -= 1
            self.grid = self.history[self.history_index].copy()
            self.im.set_data(self.grid)
            self._update_stats()
            self.fig.canvas.draw()
            print("✓ 撤销成功")
        else:
            print("⚠ 已经是最早的状态")
    
    def _redo(self, event):
        """重做操作"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.grid = self.history[self.history_index].copy()
            self.im.set_data(self.grid)
            self._update_stats()
            self.fig.canvas.draw()
            print("✓ 重做成功")
        else:
            print("⚠ 已经是最新的状态")
    
    def _reset(self, event):
        """重置到原始地图"""
        self.grid = self.original_grid.copy()
        self.history = [self.grid.copy()]
        self.history_index = 0
        self.im.set_data(self.grid)
        self._update_stats()
        self.fig.canvas.draw()
        print("✓ 已重置到原始地图")
    
    def _save(self, event):
        """保存编辑后的地图"""
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.map_file.replace('.npy', f'_edited_{timestamp}.npy')
        output_metadata = output_file.replace('.npy', '_metadata.json')
        
        # 保存地图
        np.save(output_file, self.grid)
        print(f"\n✓ 编辑后地图已保存: {output_file}")
        
        # 更新并保存元数据
        self.metadata['edited'] = True
        self.metadata['edit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(output_metadata, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"✓ 元数据已保存: {output_metadata}")
        
        # 保存可视化
        vis_file = output_file.replace('.npy', '_visualization.png')
        fig_save, ax_save = plt.subplots(figsize=(10, 5))
        ax_save.imshow(self.grid, cmap='gray_r', origin='lower')
        ax_save.set_title('Edited SLAM Map', fontsize=14, fontweight='bold')
        ax_save.set_xlabel('X (Grid)')
        ax_save.set_ylabel('Y (Grid)')
        ax_save.grid(True, alpha=0.3)
        plt.colorbar(ax_save.imshow(self.grid, cmap='gray_r', origin='lower'), 
                    ax=ax_save, label='Occupancy')
        plt.savefig(vis_file, dpi=150, bbox_inches='tight')
        plt.close(fig_save)
        print(f"✓ 可视化已保存: {vis_file}")
        
        print(f"\n使用 map_viewer.py 查看编辑后的地图:")
        print(f"  python map_viewer.py {output_file}")
    
    def _update_stats(self):
        """更新统计信息显示"""
        total = self.grid.size
        unknown = np.sum((self.grid >= 0.4) & (self.grid <= 0.6))
        free = np.sum(self.grid < 0.4)
        occupied = np.sum(self.grid > 0.6)
        
        stats_text = (
            f"统计信息:\n"
            f"自由: {free}\n"
            f"障碍: {occupied}\n"
            f"未知: {unknown}\n"
            f"---\n"
            f"笔刷: {self.brush_size}px\n"
            f"模式: {self.mode}"
        )
        
        self.text_stats.set_text(stats_text)
    
    def run(self):
        """运行编辑器"""
        print("\n编辑器已启动！开始编辑地图...")
        print("关闭窗口以退出编辑器\n")
        plt.show()

def main():
    if len(sys.argv) < 2:
        print("=" * 70)
        print("交互式SLAM地图编辑器")
        print("=" * 70)
        print("\n使用方法:")
        print("  python map_editor.py <地图文件.npy>")
        print("\n示例:")
        print("  python map_editor.py slam_map_20251128_140658.npy")
        print("\n功能:")
        print("  - 鼠标拖拽擦除/绘制障碍物")
        print("  - 调整笔刷大小")
        print("  - 撤销/重做操作")
        print("  - 保存编辑后的地图")
        print("=" * 70)
        sys.exit(1)
    
    map_file = sys.argv[1]
    
    # 创建并运行编辑器
    editor = InteractiveMapEditor(map_file)
    editor.run()
    
    print("\n" + "=" * 70)
    print("✅ 编辑器已关闭")
    print("=" * 70)

if __name__ == "__main__":
    main()
