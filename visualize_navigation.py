"""
Visual navigation planning - Display map, starting point, ending point and planned route (if available)
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

if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(os.path.join(os.path.dirname(__file__), 'controllers', 'simple_robot_controller'))

from occupancy_grid_map import OccupancyGridMap
from path_planner import PathPlanner

def visualize_with_navigation(map_file, start_pos, goal_pos):
    """
    Visual maps and navigation points
    
    Args:
        map_file: Map file
        start_pos: Beginning (x, y)
        goal_pos: Ending (x, y)
    """
    print("=" * 70)
    print("Visualization of navigation paths")
    print("=" * 70)
    
    # Load the map
    grid = np.load(map_file)
    print(f"\nMap loading: {os.path.basename(map_file)}")
    print(f"Size: {grid.shape}")
    
    # Load metadata
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
    
    # Create a map object
    slam_map = OccupancyGridMap(width=width, height=height, resolution=resolution)
    slam_map.grid = grid
    
    # Transform coordinates
    start_grid = slam_map.world_to_grid(start_pos[0], start_pos[1])
    goal_grid = slam_map.world_to_grid(goal_pos[0], goal_pos[1])
    
    print(f"\nStart: world({start_pos[0]:.3f}, {start_pos[1]:.3f}) → Grid({start_grid[0]}, {start_grid[1]})")
    print(f"Exit: world({goal_pos[0]:.3f}, {goal_pos[1]:.3f}) → Grid({goal_grid[0]}, {goal_grid[1]})")
    
    # Check the status of the beginning and the ending
    start_occ = grid[start_grid[1], start_grid[0]] if slam_map.is_valid_cell(*start_grid) else -1
    goal_occ = grid[goal_grid[1], goal_grid[0]] if slam_map.is_valid_cell(*goal_grid) else -1
    
    print(f"\nStart state: ", end='')
    if start_occ < 0:
        print("Beyond the map")
    elif start_occ < 0.4:
        print(f"Free space (Occupancy rate: {start_occ:.3f})")
    elif start_occ <= 0.6:
        print(f"Unknown area (Occupancy rate: {start_occ:.3f})")
    else:
        print(f"Obstacle (Occupancy rate: {start_occ:.3f})")
    
    print(f"Exit state: ", end='')
    if goal_occ < 0:
        print("Beyond the map")
    elif goal_occ < 0.4:
        print(f"Free space (Occupancy rate: {goal_occ:.3f})")
    elif goal_occ <= 0.6:
        print(f"Unknown area (Occupancy rate: {goal_occ:.3f})")
    else:
        print(f"Obstacle (Occupancy rate: {goal_occ:.3f})")
    
    # Try to plan the path
    print(f"\n" + "=" * 70)
    print("Try path planning...")
    print("=" * 70)
    
    planner = PathPlanner(slam_map)
    path = planner.plan_path(start_pos[0], start_pos[1], goal_pos[0], goal_pos[1])
    
    if path:
        path_length = sum([
            np.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)
            for i in range(len(path) - 1)
        ])
        print(f"The path planning was successful!")
        print(f"Path points: {len(path)}")
        print(f"Path length: {path_length:.2f} m")
        print(f"Straight-line distance: {np.sqrt((goal_pos[0]-start_pos[0])**2 + (goal_pos[1]-start_pos[1])**2):.2f} m")
    else:
        print(f"Path planning failed")
        print(f"Reason: the area from the start to the exit is completely blocked by an obstacle")
    
    # Create visualizations
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Display map
    extent = [slam_map.origin_x, slam_map.origin_x + slam_map.width,
              slam_map.origin_y, slam_map.origin_y + slam_map.height]
    im = ax.imshow(grid, cmap='gray_r', origin='lower', extent=extent, alpha=0.8)
    
    # Draw grid lines
    for i in range(0, grid.shape[1], 10):
        x = slam_map.origin_x + i * resolution
        ax.axvline(x, color='gray', linewidth=0.3, alpha=0.3)
    for j in range(0, grid.shape[0], 10):
        y = slam_map.origin_y + j * resolution
        ax.axhline(y, color='gray', linewidth=0.3, alpha=0.3)
    
    # Draw the start (big green circle)
    start_circle = Circle(start_pos, 0.08, color='green', alpha=0.8, zorder=10, linewidth=3, fill=False)
    ax.add_patch(start_circle)
    ax.plot(start_pos[0], start_pos[1], 'g*', markersize=25, zorder=11)
    ax.text(start_pos[0], start_pos[1] + 0.15, 'Start\nSTART',
            ha='center', va='bottom', fontsize=13, fontweight='bold',
            color='green', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='green', linewidth=2))
    
    # Draw the exit (the big red circle)
    goal_circle = Circle(goal_pos, 0.08, color='red', alpha=0.8, zorder=10, linewidth=3, fill=False)
    ax.add_patch(goal_circle)
    ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=25, zorder=11)
    ax.text(goal_pos[0], goal_pos[1] + 0.15, 'Exit\nEXIT',
            ha='center', va='bottom', fontsize=13, fontweight='bold',
            color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='red', linewidth=2))
    
    # If there is a path, draw it
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, 'b-', linewidth=4, label='Plan the path', zorder=5, alpha=0.7)
        ax.plot(path_x, path_y, 'co', markersize=6, zorder=6, alpha=0.8)
        
        # Add arrows
        for i in range(0, len(path) - 1, max(1, len(path) // 15)):
            dx = path_x[i+1] - path_x[i]
            dy = path_y[i+1] - path_y[i]
            ax.arrow(path_x[i], path_y[i], dx*0.8, dy*0.8,
                    head_width=0.05, head_length=0.03, fc='blue', ec='blue',
                    alpha=0.6, zorder=7, linewidth=1.5)
    else:
        # Draw a straight line (indicating that it cannot reach directly)
        ax.plot([start_pos[0], goal_pos[0]], [start_pos[1], goal_pos[1]], 
               'r--', linewidth=2, label='Straight path (blocked)', alpha=0.5, zorder=3)
    
    # 设置
    ax.set_xlabel('X Coordinates (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coordinates (meters)', fontsize=12, fontweight='bold')
    
    if path:
        title = f'Navigation path planning successful\npath length: {path_length:.2f}m, Path points: {len(path)}'
        title_color = 'green'
    else:
        title = 'The path is blocked - obstacles need to be cleared'
        title_color = 'red'
    
    ax.set_title(title, fontsize=15, fontweight='bold', color=title_color, pad=20)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Add color bars
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Occupation probability\n(0=free, 1=Obstacle)', rotation=270, labelpad=20, fontsize=10)
    
    # Add explanatory text
    info_text = (
        f"Start: ({start_pos[0]:.3f}, {start_pos[1]:.3f}) m\n"
        f"Exit: ({goal_pos[0]:.3f}, {goal_pos[1]:.3f}) m\n"
        f"Straight-line distance: {np.sqrt((goal_pos[0]-start_pos[0])**2 + (goal_pos[1]-start_pos[1])**2):.2f} m"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    plt.tight_layout()
    
    # Save
    output_file = map_file.replace('.npy', '_navigation_visualization.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nThe visualization has been saved: {output_file}")
    
    plt.show()
    
    return path

def main():
    if len(sys.argv) < 2:
        print("Usage method: python visualize_navigation.py <Map file.npy> [Start x] [Start y] [Exit x] [Exit y]")
        print("\nExample:")
        print("python visualize_navigation.py slam_map.npy")
        print("python visualize_navigation.py slam_map.npy 0 0 1.68 0.22")
        sys.exit(1)
    
    map_file = sys.argv[1]
    
    # Analytical coordinates
    if len(sys.argv) >= 6:
        start_x, start_y = float(sys.argv[2]), float(sys.argv[3])
        goal_x, goal_y = float(sys.argv[4]), float(sys.argv[5])
    else:
        start_x, start_y = 0.0, 0.0
        goal_x, goal_y = 1.68, 0.22
        print(f"Use the default coordinates: Start({start_x}, {start_y}), Exit({goal_x}, {goal_y})")
    
    path = visualize_with_navigation(map_file, (start_x, start_y), (goal_x, goal_y))
    
    print("\n" + "=" * 70)
    if path:
        print("The path planning was successful! It can perform automatic navigation")
    else:
        print("The path is blocked. Needed:")
        print("1. Use map_editor.py to clear the blocking obstacles")
        print("2. Or choose other accessible destination positions")
    print("=" * 70)

if __name__ == "__main__":
    main()
