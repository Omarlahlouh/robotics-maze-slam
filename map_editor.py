"""
Interactive SLAM Map Editor
Interactive SLAM Map Editor - Manually edit map obstacles
"""

import numpy as np
import json
import sys
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button, Slider, RadioButtons
from datetime import datetime
import platform

def setup_chinese_font():
    system = platform.system()
    
    if system == 'Windows':

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    elif system == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']

    plt.rcParams['axes.unicode_minus'] = False



setup_chinese_font()

class InteractiveMapEditor:
    def __init__(self, map_file):
        """
        Initialize the interactive map editor
        
        Args:
            map_file: Map file path (.npy)
        """
        print("=" * 70)
        print("Interactive SLAM map editor - Interactive Map Editor")
        print("=" * 70)
        
        # Load the map
        self.map_file = map_file
        self.grid = np.load(map_file)
        self.original_grid = self.grid.copy()
        
        print(f"\nThe map has loaded successfully: {map_file}")
        print(f"Grid size: {self.grid.shape}")
        
        # Load metadata
        metadata_file = map_file.replace('.npy', '_metadata.json')
        try:
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            print(f"The metadata has been loaded successfully")
        except:
            print("The metadata file was not found. Use the default parameters")
            self.metadata = {
                'resolution': 0.02,
                'width': 4.0,
                'height': 2.0
            }
        
        # Editing mode
        self.mode = 'erase'  # 'erase' or 'draw' or 'free'
        self.brush_size = 3
        self.drawing = False
        
        # Historical record (for revocation)
        self.history = [self.grid.copy()]
        self.history_index = 0
        self.max_history = 50
        
        print("\n" + "=" * 70)
        print("Instructions for Use:")
        print("=" * 70)
        print("Left mouse button + drag: Erase/draw obstacles")
        print("Right-click + drag: Set to free space")
        print("Brush size slider: Adjust the size of the brush")
        print("Mode selection: Erase Obstacles/Draw Obstacles/Set Freely")
        print("Undo button: Undo the previous operation")
        print("Redo button: Redo operation")
        print("Reset button: Restore to the original map")
        print("Save button: Save the edited map")
        print("=" * 70)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set the user interface"""
        # Create a graphic window
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title('Interactive Map Editor')
        
        # The main map display area
        self.ax_main = plt.subplot2grid((10, 12), (0, 0), colspan=10, rowspan=10)
        self.im = self.ax_main.imshow(self.grid, cmap='gray_r', origin='lower', 
                                      interpolation='nearest', vmin=0, vmax=1)
        self.ax_main.set_title('SLAM Map - Click drag to edit', fontsize=14, fontweight='bold')
        self.ax_main.set_xlabel('X (grid)')
        self.ax_main.set_ylabel('Y (grid)')
        self.ax_main.grid(True, alpha=0.3, linestyle='--')
        
        # Add color bars
        cbar = plt.colorbar(self.im, ax=self.ax_main)
        cbar.set_label('Occupancy (0=free, 1=Obstacle)', rotation=270, labelpad=20)
        
        # Brush size slider
        ax_brush = plt.subplot2grid((10, 12), (0, 10), colspan=2, rowspan=1)
        self.slider_brush = Slider(ax_brush, 'Brush', 1, 20, valinit=3, valstep=1)
        self.slider_brush.on_changed(self._update_brush_size)
        
        # Mode selection button
        ax_mode = plt.subplot2grid((10, 12), (1, 10), colspan=2, rowspan=3)
        self.radio_mode = RadioButtons(ax_mode, ('Erase obstacles', 'Draw obstacles', 'Set free'), active=0)
        self.radio_mode.on_clicked(self._change_mode)
        
        # Undo button
        ax_undo = plt.subplot2grid((10, 12), (5, 10), colspan=2, rowspan=1)
        self.btn_undo = Button(ax_undo, '(Undo)')
        self.btn_undo.on_clicked(self._undo)
        
        # Redo button
        ax_redo = plt.subplot2grid((10, 12), (6, 10), colspan=2, rowspan=1)
        self.btn_redo = Button(ax_redo, '(Redo)')
        self.btn_redo.on_clicked(self._redo)
        
        # Reset button
        ax_reset = plt.subplot2grid((10, 12), (7, 10), colspan=2, rowspan=1)
        self.btn_reset = Button(ax_reset, '(Reset)')
        self.btn_reset.on_clicked(self._reset)
        
        # Save button
        ax_save = plt.subplot2grid((10, 12), (8, 10), colspan=2, rowspan=1)
        self.btn_save = Button(ax_save, '(Save)')
        self.btn_save.on_clicked(self._save)
        
        # Statistical information display
        ax_stats = plt.subplot2grid((10, 12), (9, 10), colspan=2, rowspan=1)
        ax_stats.axis('off')
        self.text_stats = ax_stats.text(0.1, 0.5, '', fontsize=9, 
                                        verticalalignment='center')
        self._update_stats()
        
        # Connect mouse event
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        
        plt.tight_layout()
    
    def _update_brush_size(self, val):
        """Update the brush size"""
        self.brush_size = int(val)
    
    def _change_mode(self, label):
        """Change the editing mode"""
        if label == 'Erase obstacles':
            self.mode = 'erase'
        elif label == 'Draw obstacles':
            self.mode = 'draw'
        elif label == 'set free':
            self.mode = 'free'
        print(f"Mode switch to: {label}")
    
    def _on_mouse_press(self, event):
        """Mouse press event"""
        if event.inaxes == self.ax_main:
            self.drawing = True
            self._edit_at_position(event.xdata, event.ydata, event.button)
    
    def _on_mouse_release(self, event):
        """Mouse release event"""
        if self.drawing:
            self.drawing = False
            # save to history
            self._save_to_history()
    
    def _on_mouse_move(self, event):
        """Mouse move event"""
        if self.drawing and event.inaxes == self.ax_main:
            self._edit_at_position(event.xdata, event.ydata, event.button)
    
    def _edit_at_position(self, x, y, button):
        """Edit the map at the designated location"""
        if x is None or y is None:
            return
        
        # Convert to raster coordinates
        grid_x = int(round(x))
        grid_y = int(round(y))
        
        # Check the boundary
        if not (0 <= grid_x < self.grid.shape[1] and 0 <= grid_y < self.grid.shape[0]):
            return
        
        # Determine the edit value
        if button == MouseButton.LEFT:
            if self.mode == 'erase':
                value = 0.0  # Set it as free space
            elif self.mode == 'draw':
                value = 0.9  # Set as an obstacle
            elif self.mode == 'free':
                value = 0.0  # Set it as free space
        elif button == MouseButton.RIGHT:
            value = 0.0  # The right button is always erased
        else:
            return
        
        # Apply the brush
        half_size = self.brush_size // 2
        for dy in range(-half_size, half_size + 1):
            for dx in range(-half_size, half_size + 1):
                # Round brush
                if dx*dx + dy*dy <= half_size*half_size:
                    ny, nx = grid_y + dy, grid_x + dx
                    if 0 <= ny < self.grid.shape[0] and 0 <= nx < self.grid.shape[1]:
                        self.grid[ny, nx] = value
        
        # Update display
        self.im.set_data(self.grid)
        self._update_stats()
        self.fig.canvas.draw_idle()
    
    def _save_to_history(self):
        """Save the current state to the history record"""
        # Delete the history after the current position
        self.history = self.history[:self.history_index + 1]
        
        # Add a new status
        self.history.append(self.grid.copy())
        self.history_index += 1
        
        # 限制历史记录数量
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.history_index -= 1
    
    def _undo(self, event):
        """Undo operation"""
        if self.history_index > 0:
            self.history_index -= 1
            self.grid = self.history[self.history_index].copy()
            self.im.set_data(self.grid)
            self._update_stats()
            self.fig.canvas.draw()
            print("Undo successful")
        else:
            print("It is already the earliest state")
    
    def _redo(self, event):
        """Redo operation"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.grid = self.history[self.history_index].copy()
            self.im.set_data(self.grid)
            self._update_stats()
            self.fig.canvas.draw()
            print("Redo successful")
        else:
            print("It is already the latest status")
    
    def _reset(self, event):
        """Reset to the original map"""
        self.grid = self.original_grid.copy()
        self.history = [self.grid.copy()]
        self.history_index = 0
        self.im.set_data(self.grid)
        self._update_stats()
        self.fig.canvas.draw()
        print("Reset to the original map")
    
    def _save(self, event):
        """Save the edited map"""
        # Generate the file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.map_file.replace('.npy', f'_edited_{timestamp}.npy')
        output_metadata = output_file.replace('.npy', '_metadata.json')
        
        # Save the map
        np.save(output_file, self.grid)
        print(f"\nThe edited map has been saved: {output_file}")
        
        # Update and save the metadata
        self.metadata['edited'] = True
        self.metadata['edit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(output_metadata, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"The metadata has been saved: {output_metadata}")
        
        # Save visualization
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
        print(f"The visualization has been saved: {vis_file}")
        
        print(f"\nUse map_viewer.py View the edited map:")
        print(f"  python map_viewer.py {output_file}")
    
    def _update_stats(self):
        """Update the statistical information display"""
        total = self.grid.size
        unknown = np.sum((self.grid >= 0.4) & (self.grid <= 0.6))
        free = np.sum(self.grid < 0.4)
        occupied = np.sum(self.grid > 0.6)
        
        stats_text = (
            f"Statistical information:\n"
            f"free: {free}\n"
            f"Obstacle: {occupied}\n"
            f"unknown: {unknown}\n"
            f"---\n"
            f"brush: {self.brush_size}px\n"
            f"mode: {self.mode}"
        )
        
        self.text_stats.set_text(stats_text)
    
    def run(self):
        """Run the editor"""
        print("\nThe editor has been launched! Start editing the map...")
        print("Close the window to exit the editor\n")
        plt.show()

def main():
    if len(sys.argv) < 2:
        print("=" * 70)
        print("Interactive SLAM map editor")
        print("=" * 70)
        print("\nUsage method:")
        print("  python map_editor.py <Map file.npy>")
        print("\nExample:")
        print("  python map_editor.py slam_map_20251128_140658.npy")
        print("\nFunction:")
        print("  - Drag the mouse to erase/draw obstacles")
        print("  - Adjust the size of the brush")
        print("  - Undo/Redo operation")
        print("  - Save the edited map")
        print("=" * 70)
        sys.exit(1)
    
    map_file = sys.argv[1]
    
    # Create and run the editor
    editor = InteractiveMapEditor(map_file)
    editor.run()
    
    print("\n" + "=" * 70)
    print("The editor has been closed")
    print("=" * 70)

if __name__ == "__main__":
    main()
