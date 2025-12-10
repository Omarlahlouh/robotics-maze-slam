"""
SLAM Map Viewer
Visualize saved occupancy grid maps
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json

def load_and_visualize_map(map_file):
    """
    Load and visualize a saved SLAM map
    
    Args:
        map_file: Path to the .npy map file
    """
    # Load the map
    try:
        grid = np.load(map_file)
        print(f"Map loaded: {map_file}")
        print(f"Grid shape: {grid.shape}")
    except Exception as e:
        print(f"Error loading map: {e}")
        return
    
    # Load metadata if available
    metadata_file = map_file.replace('.npy', '_metadata.json')
    robot_pose = None
    map_info = None
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                robot_pose = metadata.get('robot_pose')
                map_info = {
                    'width': metadata.get('width'),
                    'height': metadata.get('height'),
                    'resolution': metadata.get('resolution'),
                    'origin_x': metadata.get('origin_x'),
                    'origin_y': metadata.get('origin_y')
                }
                print(f"Metadata loaded: {metadata_file}")
                if robot_pose:
                    print(f"Robot pose: ({robot_pose['world_x']:.3f}, {robot_pose['world_y']:.3f}, {robot_pose['theta']:.1f}°)")
                    print(f"Robot grid position: ({robot_pose['grid_x']}, {robot_pose['grid_y']})")
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
            robot_pose = None
            map_info = None
    
    # Calculate statistics
    total_cells = grid.size
    unknown = np.sum((grid > 0.4) & (grid < 0.6))
    free = np.sum(grid < 0.4)
    occupied = np.sum(grid > 0.6)
    explored_percent = 100.0 * (total_cells - unknown) / total_cells
    
    print(f"\nMap Statistics:")
    print(f"  Total cells: {total_cells}")
    print(f"  Unknown: {unknown} ({100*unknown/total_cells:.1f}%)")
    print(f"  Free: {free} ({100*free/total_cells:.1f}%)")
    print(f"  Occupied: {occupied} ({100*occupied/total_cells:.1f}%)")
    print(f"  Explored: {explored_percent:.1f}%")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Grayscale map (0=free, 0.5=unknown, 1=occupied)
    im1 = ax1.imshow(grid, cmap='gray_r', origin='lower', vmin=0, vmax=1)
    ax1.set_title('Occupancy Grid Map\n(Black=Free, Gray=Unknown, White=Occupied)')
    ax1.set_xlabel('Grid X')
    ax1.set_ylabel('Grid Y')
    plt.colorbar(im1, ax=ax1, label='Occupancy Probability')
    
    # Plot 2: Colored map for better visualization
    # Create custom colormap: blue=free, gray=unknown, red=occupied
    colored_map = np.zeros((*grid.shape, 3))
    
    # Free space (blue)
    free_mask = grid < 0.4
    colored_map[free_mask] = [0.7, 0.9, 1.0]  # Light blue
    
    # Unknown (gray)
    unknown_mask = (grid >= 0.4) & (grid <= 0.6)
    colored_map[unknown_mask] = [0.5, 0.5, 0.5]  # Gray
    
    # Occupied (red to dark red based on probability)
    occupied_mask = grid > 0.6
    occupied_intensity = (grid[occupied_mask] - 0.6) / 0.4  # Normalize to 0-1
    colored_map[occupied_mask, 0] = 0.3 + 0.7 * occupied_intensity  # Red channel
    colored_map[occupied_mask, 1] = 0.1  # Green channel
    colored_map[occupied_mask, 2] = 0.1  # Blue channel
    
    ax2.imshow(colored_map, origin='lower')
    ax2.set_title('Colored Map\n(Blue=Free, Gray=Unknown, Red=Occupied)')
    ax2.set_xlabel('Grid X')
    ax2.set_ylabel('Grid Y')
    
    # Draw robot pose if available
    if robot_pose:
        grid_x = robot_pose['grid_x']
        grid_y = robot_pose['grid_y']
        theta_rad = np.radians(robot_pose['theta'])
        
        # Draw robot position as a marker
        marker_size = 100
        ax1.scatter(grid_x, grid_y, c='lime', s=marker_size, marker='o', 
                   edgecolors='black', linewidths=2, label='Robot', zorder=10)
        ax2.scatter(grid_x, grid_y, c='lime', s=marker_size, marker='o', 
                   edgecolors='black', linewidths=2, label='Robot', zorder=10)
        
        # Draw orientation arrow
        arrow_length = 15  # pixels
        dx = arrow_length * np.cos(theta_rad)
        dy = arrow_length * np.sin(theta_rad)
        ax1.arrow(grid_x, grid_y, dx, dy, head_width=5, head_length=5, 
                 fc='yellow', ec='black', linewidth=2, zorder=11)
        ax2.arrow(grid_x, grid_y, dx, dy, head_width=5, head_length=5, 
                 fc='yellow', ec='black', linewidth=2, zorder=11)
        
        # Add legend
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
    
    # Add grid lines
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    
    # Save visualization
    output_file = map_file.replace('.npy', '_visualization.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    # Show the plot
    plt.show()

def find_latest_map():
    """Find the most recent SLAM map file"""
    map_files = [f for f in os.listdir('.') if f.startswith('slam_map_') and f.endswith('.npy')]
    if not map_files:
        return None
    # Sort by modification time
    map_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return map_files[0]

def main():
    """Main function"""
    print("=" * 60)
    print("SLAM Map Viewer")
    print("=" * 60)
    
    # Check if map file is provided as argument
    if len(sys.argv) > 1:
        map_file = sys.argv[1]
    else:
        # Try to find the latest map
        map_file = find_latest_map()
        if map_file:
            print(f"\nNo map file specified. Using latest map: {map_file}")
        else:
            print("\nNo map file found in current directory.")
            print("\nUsage:")
            print("  python map_viewer.py [map_file.npy]")
            print("\nOr run from the controller directory after saving a map with 'M' key.")
            return
    
    # Check if file exists
    if not os.path.exists(map_file):
        print(f"\nError: Map file not found: {map_file}")
        return
    
    # Load and visualize
    print()
    load_and_visualize_map(map_file)

if __name__ == "__main__":
    main()
