"""
Navigate to Exit - Automatically navigate to the exit
Plan the path and navigate to the designated exit using the saved SLAM map
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.patches import Circle, FancyArrowPatch
import sys
import os

# Add controller path
sys.path.append(os.path.join(os.path.dirname(__file__), 'controllers', 'simple_robot_controller'))

from occupancy_grid_map import OccupancyGridMap
from path_planner import PathPlanner

import platform
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class NavigationVisualizer:
    def __init__(self, map_file, start_pos, goal_pos):
        """
        Initialize the navigation visualizer
        
        Args:
            map_file: Map file path
            start_pos: Start coordinate (x, y)
            goal_pos: Goal coordinate (x, y)
        """
        print("=" * 70)
        print("Automatically navigate to the exit - Navigate to Exit")
        print("=" * 70)
        
        # Load the map
        self.grid = np.load(map_file)
        print(f"\nThe map has loaded successfully.: {map_file}")
        print(f"Grid size: {self.grid.shape}")
        
        # Load metadata
        metadata_file = map_file.replace('.npy', '_metadata.json')
        try:
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            print(f"The metadata has been loaded successfully")
            resolution = self.metadata.get('resolution', 0.02)
            width = self.metadata.get('width', 4.0)
            height = self.metadata.get('height', 2.0)
        except:
            print("The metadata file was not found. Use the default parameters")
            resolution = 0.02
            width = 4.0
            height = 2.0
        
        # Create a map object
        self.slam_map = OccupancyGridMap(width=width, height=height, resolution=resolution)
        self.slam_map.grid = self.grid
        
        # Set the start and the goal
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        
        print(f"\nStart position: ({start_pos[0]:.3f}, {start_pos[1]:.3f}) m")
        print(f"Goal position: ({goal_pos[0]:.3f}, {goal_pos[1]:.3f}) m")
        
        # Path planning
        self.planner = PathPlanner(self.slam_map)
        self.path = None
        
    def plan_path(self):
        """Plan the path"""
        print("\n" + "=" * 70)
        print("Start path planning...")
        print("=" * 70)
        
        # Plan the path using the A* algorithm
        self.path = self.planner.plan_path(
            self.start_pos[0], self.start_pos[1],
            self.goal_pos[0], self.goal_pos[1]
        )
        
        if self.path is None:
            print("\nPath planning failed! Possible reasons：")
            print("  1. The start or the goal is on an obstacle")
            print("  2. The start and the goal are completely blocked by an obstacle")
            print("  3. Map quality issue")
            print("\nSuggestions：")
            print("  - Use map_editor.py manually clear the obstacles blocking the path")
            print("  - Check whether the starting point and the ending point coordinates are correct")
            return False
        
        # Calculate the path length
        path_length = 0.0
        for i in range(len(self.path) - 1):
            dx = self.path[i+1][0] - self.path[i][0]
            dy = self.path[i+1][1] - self.path[i][1]
            path_length += np.sqrt(dx**2 + dy**2)
        
        print(f"\nThe path planning was successful!")
        print(f"  Path points: {len(self.path)}")
        print(f"  Path length: {path_length:.2f} m")
        print(f"  Straight-line distance: {np.sqrt((self.goal_pos[0]-self.start_pos[0])**2 + (self.goal_pos[1]-self.start_pos[1])**2):.2f} m")
        
        # Print path
        print(f"\nPath details (the first 5 points):")
        for i, (x, y) in enumerate(self.path[:5]):
            print(f"  Point {i+1}: ({x:.3f}, {y:.3f}) m")
        if len(self.path) > 5:
            print(f"  ... （totally{len(self.path)}points）")
            x, y = self.path[-1]
            print(f"  Point {len(self.path)}: ({x:.3f}, {y:.3f}) m [goal]")
        
        return True
    
    def visualize(self, save_file=None):
        """Visual maps and paths"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Display map
        im = ax.imshow(self.grid, cmap='gray_r', origin='lower', extent=[
            self.slam_map.origin_x,
            self.slam_map.origin_x + self.slam_map.width,
            self.slam_map.origin_y,
            self.slam_map.origin_y + self.slam_map.height
        ])
        
        # Draw the start (green circle)
        start_circle = Circle(self.start_pos, 0.05, color='green', alpha=0.7, zorder=10)
        ax.add_patch(start_circle)
        ax.text(self.start_pos[0], self.start_pos[1] + 0.1, 'start',
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                color='green', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Draw the goal (red star shape)
        ax.plot(self.goal_pos[0], self.goal_pos[1], 'r*', markersize=20, zorder=10)
        ax.text(self.goal_pos[0], self.goal_pos[1] + 0.1, 'exit',
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Draw a path
        if self.path:
            path_x = [p[0] for p in self.path]
            path_y = [p[1] for p in self.path]
            ax.plot(path_x, path_y, 'b-', linewidth=3, label='Plan the path', zorder=5, alpha=0.7)
            
            # Draw path points
            ax.plot(path_x, path_y, 'co', markersize=4, zorder=6, alpha=0.6)
            
            # Draw arrows to indicate directions
            for i in range(0, len(self.path) - 1, max(1, len(self.path) // 10)):
                arrow = FancyArrowPatch(
                    (path_x[i], path_y[i]), 
                    (path_x[i+1], path_y[i+1]),
                    arrowstyle='->', mutation_scale=20, linewidth=2,
                    color='blue', alpha=0.6, zorder=7
                )
                ax.add_patch(arrow)
        
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title('Automatic navigation path planning - from the start to the exit', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Add color bars
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Occupation probability (0=free, 1=Obstacle)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"\nThe path planning diagram has been saved: {save_file}")
        
        plt.show()
        
    def export_path_for_robot(self, output_file='navigation_path.json'):
        """Export the path for the robot to use"""
        if not self.path:
            print("There is no path that can be exported")
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
        
        print(f"\nThe path data has been exported: {output_file}")
        print(f"This path can be loaded in simple_robot_controller.py")

def main():
    print("=" * 70)
    print("Automatically navigate to the exit - Navigate to Exit")
    print("=" * 70)
    
    if len(sys.argv) < 2:
        print("\nUsage method:")
        print("  python navigate_to_exit.py <地图文件.npy> [start x] [start y] [exit x] [exit y]")
        print("\nExample:")
        print("  python navigate_to_exit.py slam_map_edited.npy 0 0 1.75 0.20")
        print("  python navigate_to_exit.py slam_map_edited.npy")
        print("\nParameter:")
        print("  Map file: The saved SLAM map(.npy)")
        print("  Start coordinates: The world coordinates of the start(x, y)，Default(0, 0)")
        print("  Exit coordinates: The world exit of the start(x, y)，Default(1.75, 0.20)")
        print("=" * 70)
        sys.exit(1)
    
    map_file = sys.argv[1]
    
    # Parse coordinate parameters
    if len(sys.argv) >= 6:
        start_x = float(sys.argv[2])
        start_y = float(sys.argv[3])
        goal_x = float(sys.argv[4])
        goal_y = float(sys.argv[5])
    else:
        # Default value: The coordinate specified by the user
        start_x, start_y = 0.0, 0.0  # start
        goal_x, goal_y = 1.75, 0.20  # exit
        print(f"\nUse the default coordinates:")
        print(f"start: ({start_x}, {start_y})")
        print(f"exit: ({goal_x}, {goal_y})")
    
    # Create a navigation visualizer
    nav_viz = NavigationVisualizer(
        map_file=map_file,
        start_pos=(start_x, start_y),
        goal_pos=(goal_x, goal_y)
    )
    
    # Plan the path
    success = nav_viz.plan_path()
    
    if not success:
        print("\n" + "=" * 70)
        print("Navigation planning failed")
        print("=" * 70)
        sys.exit(1)
    
    # Export path
    output_dir = os.path.dirname(map_file)
    path_file = os.path.join(output_dir, 'navigation_path.json')
    nav_viz.export_path_for_robot(path_file)
    
    # Visualization
    viz_file = map_file.replace('.npy', '_navigation_plan.png')
    nav_viz.visualize(save_file=viz_file)
    
    print("\n" + "=" * 70)
    print("Navigation planning completed!")
    print("=" * 70)
    print("\nNext step:")
    print("  1. Start the simulation in Webots")
    print("  2. Press '3' to switch to the automatic navigation mode")
    print("  3. The robot will automatically navigate along the planned path to the exit")
    print("\nPath file:")
    print(f" {path_file}")
    print(f" {viz_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
