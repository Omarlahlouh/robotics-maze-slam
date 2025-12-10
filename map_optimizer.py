"""
SLAM Map Optimizer
Optimize the existing map, remove error obstacles, and restore feasible routes
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
        Initialize the map optimizer
        
        Args:
            map_file: Map file path (.npy)
        """
        print("=" * 60)
        print("SLAM Map Optimizer - Map optimization tool")
        print("=" * 60)
        
        # Load the map
        self.map_file = map_file
        self.grid = np.load(map_file)
        print(f"\nThe map has loaded successfully: {map_file}")
        print(f"Grid size: {self.grid.shape}")
        
        # Load metadata
        metadata_file = map_file.replace('.npy', '_metadata.json')
        try:
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            print(f"The metadata has been loaded successfully")
            print(f"Resolution: {self.metadata.get('resolution', 0.02)}m")
            print(f"Map size: {self.metadata.get('width', 4.0)}m × {self.metadata.get('height', 2.0)}m")
        except:
            print("The metadata file was not found. Use the default parameters")
            self.metadata = {
                'resolution': 0.02,
                'width': 4.0,
                'height': 2.0
            }
        
        # Save the original map
        self.original_grid = self.grid.copy()
        
        # Print statistical information
        self._print_statistics("Original map")
    
    def _print_statistics(self, label="Map"):
        """Print the map statistics information"""
        total = self.grid.size
        unknown = np.sum((self.grid >= 0.4) & (self.grid <= 0.6))
        free = np.sum(self.grid < 0.4)
        occupied = np.sum(self.grid > 0.6)
        
        print(f"\n{label}Statistic:")
        print(f"Total number of grid: {total}")
        print(f"Unknown area: {unknown} ({unknown/total*100:.1f}%)")
        print(f"Free space: {free} ({free/total*100:.1f}%)")
        print(f"Obstacle: {occupied} ({occupied/total*100:.1f}%)")
        print(f"Explored rate: {(total-unknown)/total*100:.1f}%")
    
    def optimize_pathways(self, erosion_size=1, remove_thin_walls=True, 
                         remove_isolated=True, min_obstacle_size=3):
        """
        Optimize feasible routes and remove error obstacles
        
        Args:
            erosion_size: The size of structural elements in corrosion operations (removing thin obstacles)
            remove_thin_walls: Whether to remove the thin wall (single-pixel wall)
            remove_isolated: Whether to remove the isolated obstacle points
            min_obstacle_size: Minimum obstacle size (those less than this value are regarded as noise points)
        """
        print("\n" + "=" * 60)
        print("Start optimizing feasible routes...")
        print("=" * 60)
        
        optimized = self.grid.copy()
        
        # Convert to a binary image
        binary_map = optimized > 0.6
        
        # 1. Remove isolated obstacle points
        if remove_isolated:
            print(f"\n[Step 1] Remove isolated obstacle points (connected domain analysis)...")
            labeled, num_features = ndimage.label(binary_map)
            
            removed_count = 0
            for i in range(1, num_features + 1):
                component = (labeled == i)
                size = np.sum(component)
                
                # Remove the connected domains that are less than the threshold
                if size < min_obstacle_size:
                    optimized[component] = 0.0  # Set it as free space
                    removed_count += 1
            
            print(f"Removed{removed_count}the isolated obstacle points")
            binary_map = optimized > 0.6
        
        # 2. Remove thin walls (single-pixel walls)
        if remove_thin_walls:
            print(f"\n[Step 2] Remove the thin wall...")
            # Detect the 8-neighbors of each obstacle pixel
            thin_wall_mask = np.zeros_like(binary_map)
            
            for i in range(1, binary_map.shape[0] - 1):
                for j in range(1, binary_map.shape[1] - 1):
                    if binary_map[i, j]:
                        # 8 Neighbors
                        neighbors = binary_map[i-1:i+2, j-1:j+2]
                        obstacle_neighbors = np.sum(neighbors) - 1  # Subtract oneself
                        
                        # If there are few obstacle neighbors（<=2），it might be a thin wall
                        if obstacle_neighbors <= 2:
                            thin_wall_mask[i, j] = True
            
            removed = np.sum(thin_wall_mask)
            optimized[thin_wall_mask] = 0.0
            print(f"removed{removed}the thin wall pixels")
            binary_map = optimized > 0.6
        
        # 3. Corrosion operation (removing thin obstacles)
        if erosion_size > 0:
            print(f"\n[Step 3] Corrosion operation (Remove thin obstacles and dimensions={erosion_size}）...")
            structure = np.ones((erosion_size*2+1, erosion_size*2+1))
            eroded = ndimage.binary_erosion(binary_map, structure=structure)
            
            removed = np.sum(binary_map) - np.sum(eroded)
            optimized[binary_map & ~eroded] = 0.0
            print(f"corroded{removed}the edge barrier pixels")
            binary_map = optimized > 0.6
        
        # 4. Open operation (removing small protruding parts)
        print(f"\n[Step 4] Open operation (smooth the edge of the obstacle)...")
        structure = np.ones((3, 3))
        opened = ndimage.binary_opening(binary_map, structure=structure)
        
        removed = np.sum(binary_map) - np.sum(opened)
        optimized[binary_map & ~opened] = 0.0
        print(f"Smoothed out{removed}the edge protruding pixels")
        
        # 5. Expand the main walls (maintain the structure)
        print(f"\n[Step 5] Slightly expand the main walls (maintain the structure)...")
        binary_map = optimized > 0.6
        structure = np.ones((2, 2))
        dilated = ndimage.binary_dilation(binary_map, structure=structure)
        
        # It only expands where it is certain to be free space
        free_space = self.original_grid < 0.3
        safe_dilation = dilated & ~free_space
        
        added = np.sum(safe_dilation) - np.sum(binary_map)
        optimized[safe_dilation & ~binary_map] = 0.8
        print(f"expanded{added}the wall pixels (only in safe areas)")
        
        self.grid = optimized
        
        print("\n" + "=" * 60)
        print("Optimization completed!")
        print("=" * 60)
        
        self._print_statistics("Optimized map")
        
        # Statistical improvement
        original_occupied = np.sum(self.original_grid > 0.6)
        optimized_occupied = np.sum(self.grid > 0.6)
        cleared = original_occupied - optimized_occupied
        
        print(f"\nImprove statistics:")
        print(f"Original obstacle:{original_occupied}Pixel")
        print(f"Optimized obstacles:{optimized_occupied}Pixel")
        print(f"Clear the obstacles:{cleared}Pixel ({cleared/original_occupied*100:.1f}%)")
        
        if cleared > 0:
            print(f"Successfully cleared{cleared}error obstacles and restored the feasible route！")
        else:
            print(f"The obstacles have not been cleared, so the map quality might already be very good")
    
    def widen_pathways(self, width=2):
        """
        Widen the passage to ensure that the robot can pass through
        
        Args:
            width: Corrosion width (pixels)
        """
        print(f"\n[Additional Optimization] Widen the channel (corrode{width}pixels)...")
        
        binary_map = self.grid > 0.6
        structure = np.ones((width*2+1, width*2+1))
        eroded = ndimage.binary_erosion(binary_map, structure=structure)
        
        removed = np.sum(binary_map) - np.sum(eroded)
        self.grid[binary_map & ~eroded] = 0.0
        
        print(f"The passage was widened，removed{removed}the edge obstacles")
    
    def save_optimized_map(self, suffix="_optimized"):
        """Save the optimized map"""
        # Generate the output file name
        output_file = self.map_file.replace('.npy', f'{suffix}.npy')
        output_metadata = output_file.replace('.npy', '_metadata.json')
        
        # Save the map
        np.save(output_file, self.grid)
        print(f"\nThe optimized map has been saved: {output_file}")
        
        # Update metadata
        self.metadata['optimized'] = True
        self.metadata['optimization_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(output_metadata, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"The metadata has been saved: {output_metadata}")
        
        return output_file
    
    def visualize_comparison(self, save_file=None):
        """Visualize and compare the original map with the optimized map"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original map
        axes[0].imshow(self.original_grid, cmap='gray_r', origin='lower')
        axes[0].set_title('Original map', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X (Grid)')
        axes[0].set_ylabel('Y (Grid)')
        axes[0].grid(True, alpha=0.3)
        
        # Optimized map
        axes[1].imshow(self.grid, cmap='gray_r', origin='lower')
        axes[1].set_title('Optimized map', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X (Grid)')
        axes[1].set_ylabel('Y (Grid)')
        axes[1].grid(True, alpha=0.3)
        
        # Difference graph (red = cleared obstacles, green = newly added obstacles)
        diff = np.zeros((*self.grid.shape, 3))
        diff[:, :, 1] = (self.original_grid > 0.6) & (self.grid <= 0.6)
        diff[:, :, 0] = (self.original_grid <= 0.6) & (self.grid > 0.6)
        
        axes[2].imshow(diff, origin='lower')
        axes[2].set_title('Difference graph (green = Cleared obstacles, red = newly added obstacles）', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('X (Grid)')
        axes[2].set_ylabel('Y (Grid)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_file is None:
            save_file = self.map_file.replace('.npy', '_optimization_comparison.png')
        
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"\nThe comparison chart has been saved: {save_file}")
        
        plt.show()
        
        return save_file

def main():
    if len(sys.argv) < 2:
        print("Usage method: python map_optimizer.py < map file.npy > [options]")
        print("\nOptions:")
        print("--erosion <Size>        Corrosion size (Default: 1)")
        print("--min-size <Size>       Minimum obstacle size (Default: 3)")
        print("--widen <Width>         Widen the width of the passage (Default: 0)")
        print("--no-thin-walls         Not remove the thin wall")
        print("--no-isolated           Not remove the outliers")
        print("\nExample:")
        print("python map_optimizer.py slam_map.npy")
        print("python map_optimizer.py slam_map.npy --erosion 2 --widen 1")
        sys.exit(1)
    
    map_file = sys.argv[1]
    
    # Parsing parameters
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
    
    # Create an optimizer
    optimizer = MapOptimizer(map_file)
    
    # Optimize the map
    optimizer.optimize_pathways(
        erosion_size=erosion_size,
        remove_thin_walls=remove_thin_walls,
        remove_isolated=remove_isolated,
        min_obstacle_size=min_obstacle_size
    )
    
    # Optional: Widen the passage
    if widen_width > 0:
        optimizer.widen_pathways(width=widen_width)
    
    # Save the optimized map
    output_file = optimizer.save_optimized_map()
    
    # Visual comparison
    optimizer.visualize_comparison()
    
    print("\n" + "=" * 60)
    print("All optimizations completed!")
    print("=" * 60)
    print(f"\nThe optimized map can be used for path planning:")
    print(f"{output_file}")
    print(f"\nUse map_view.py to view the optimized map:")
    print(f"python map_viewer.py {output_file}")

if __name__ == "__main__":
    main()
