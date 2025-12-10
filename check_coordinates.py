"""
Coordinate Checking tool - Check the status of the specified world coordinates on the map
"""

import numpy as np
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'controllers', 'simple_robot_controller'))
from occupancy_grid_map import OccupancyGridMap

def check_coordinate(map_file, world_x, world_y):
    """Check the status of the world coordinates on the map"""
    
    # Load the map
    grid = np.load(map_file)
    print(f"The map has loaded successfully: {map_file}")
    print(f"Grid size: {grid.shape}")
    
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
    
    # Convert to grid coordinates
    grid_x, grid_y = slam_map.world_to_grid(world_x, world_y)
    
    print(f"\nCoordinate check:")
    print(f"World coordinate: ({world_x:.3f}, {world_y:.3f}) m")
    print(f"grid coordinate: ({grid_x}, {grid_y})")
    
    # Check if it is within the map range
    if not slam_map.is_valid_cell(grid_x, grid_y):
        print(f"The coordinates are out of the map range!")
        print(f"Map range: X[{slam_map.origin_x:.2f}, {slam_map.origin_x+slam_map.width:.2f}], "
              f"Y[{slam_map.origin_y:.2f}, {slam_map.origin_y+slam_map.height:.2f}]")
        return
    
    # Obtain the occupied value
    occupancy = grid[grid_y, grid_x]
    
    print(f"Occupation probability: {occupancy:.3f}")
    
    if occupancy < 0.4:
        print(f"Free space (passable)")
    elif occupancy <= 0.6:
        print(f"Unknown area")
    else:
        print(f"Obstacle (Impassable)")
    
    # Inspect the surrounding area
    print(f"\nInspection of the surrounding area（3x3）:")
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
                    print(" ← objective", end='')
                print()

def main():
    if len(sys.argv) < 4:
        print("Usage method: python check_coordinates.py <地图文件.npy> <x> <y>")
        print("\nExample:")
        print("  python check_coordinates.py slam_map.npy 1.75 0.20")
        sys.exit(1)
    
    map_file = sys.argv[1]
    world_x = float(sys.argv[2])
    world_y = float(sys.argv[3])
    
    check_coordinate(map_file, world_x, world_y)

if __name__ == "__main__":
    main()
