"""
Occupancy Grid Map for SLAM
Implements probabilistic occupancy grid mapping using LiDAR data
"""

import numpy as np
import math
from scipy import ndimage

class OccupancyGridMap:
    def __init__(self, width=4.0, height=2.0, resolution=0.02):
        """
        Initialize occupancy grid map
        
        Args:
            width: Map width in meters
            height: Map height in meters
            resolution: Grid cell size in meters
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        
        # Calculate grid dimensions
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # Initialize grid with unknown (0.5 = unknown state)
        self.grid = np.full((self.grid_height, self.grid_width), 0.5, dtype=np.float32)
        
        # Initialize log-odds grid for probabilistic updates (0 = unknown)
        self.log_odds = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        # Obstacle verification: track observation counts
        self.obstacle_observations = np.zeros((self.grid_height, self.grid_width), dtype=np.int16)
        self.free_observations = np.zeros((self.grid_height, self.grid_width), dtype=np.int16)
        self.min_observations_for_obstacle = 5  # Require 5+ observations (increased for faster turning)
        self.verification_enabled = True  # Enable two-pass verification
        
        # Map origin (bottom-left corner in world coordinates)
        # Center the map around robot starting position
        self.origin_x = -width / 2.0  # -2.0m
        self.origin_y = -height / 2.0  # -1.0m
        
        # Sensor model parameters (inverse sensor model)
        # Very conservative for no downsampling
        self.prob_occ = 0.75     # Probability of occupied when hit (stronger)
        self.prob_free = 0.485   # Probability of free in ray path (very conservative)
        self.log_odds_occ = self._prob_to_log_odds(self.prob_occ)
        self.log_odds_free = self._prob_to_log_odds(self.prob_free)
        
        # Measurement filtering
        self.min_range = 0.08    # Minimum valid range (m)
        self.max_range = 4.5     # Maximum valid range (m)
        self.obstacle_thickness = 1  # Cells to mark as obstacle (1 cell = 0.02m)
        
        # LiDAR downsampling (disabled per user request)
        # Use all 360 rays with aggressive filtering
        self.lidar_downsample_factor = 1  # No downsampling
        
        # Prior probability (unknown state)
        self.prob_prior = 0.5
        self.log_odds_prior = 0.0
        
        # Thresholds
        self.max_log_odds = 10.0
        self.min_log_odds = -10.0
        
        print(f"Occupancy Grid Map initialized: {self.grid_width}x{self.grid_height} cells")
        print(f"Map size: {width}m x {height}m, Resolution: {resolution}m")
        print(f"Obstacle verification: {'Enabled' if self.verification_enabled else 'Disabled'} (min observations: {self.min_observations_for_obstacle})")
    
    def _prob_to_log_odds(self, prob):
        """Convert probability to log-odds"""
        return math.log(prob / (1.0 - prob))
    
    def _log_odds_to_prob(self, log_odds):
        """Convert log-odds to probability"""
        return 1.0 - 1.0 / (1.0 + math.exp(log_odds))
    
    def world_to_grid(self, x, y):
        """
        Convert world coordinates to grid coordinates
        
        Args:
            x, y: World coordinates in meters
            
        Returns:
            grid_x, grid_y: Grid cell indices
        """
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """
        Convert grid coordinates to world coordinates
        
        Args:
            grid_x, grid_y: Grid cell indices
            
        Returns:
            x, y: World coordinates in meters
        """
        x = grid_x * self.resolution + self.origin_x
        y = grid_y * self.resolution + self.origin_y
        return x, y
    
    def is_valid_cell(self, grid_x, grid_y):
        """Check if grid cell is within map bounds"""
        return (0 <= grid_x < self.grid_width and 
                0 <= grid_y < self.grid_height)
    
    def bresenham_line(self, x0, y0, x1, y1):
        """
        Bresenham's line algorithm to get cells along a ray
        
        Args:
            x0, y0: Start grid coordinates
            x1, y1: End grid coordinates
            
        Returns:
            List of (x, y) grid coordinates along the line
        """
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            if self.is_valid_cell(x, y):
                cells.append((x, y))
            
            if x == x1 and y == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return cells
    
    def update_map(self, robot_x, robot_y, robot_theta, ranges):
        """
        Update occupancy grid map with LiDAR scan
        
        Args:
            robot_x, robot_y: Robot position in world coordinates
            robot_theta: Robot orientation in radians
            ranges: Array of LiDAR range measurements
        """
        # Convert robot position to grid coordinates
        robot_grid_x, robot_grid_y = self.world_to_grid(robot_x, robot_y)
        
        if not self.is_valid_cell(robot_grid_x, robot_grid_y):
            return
        
        # Number of LiDAR beams
        num_beams = len(ranges)
        angle_increment = 2 * math.pi / num_beams
        
        # Process each LiDAR beam with downsampling
        for i, distance in enumerate(ranges):
            # Downsample: only process every Nth beam
            if i % self.lidar_downsample_factor != 0:
                continue
            # Skip invalid readings with stricter filtering
            if distance >= self.max_range or distance < self.min_range:
                continue
            
            # Skip very close readings (likely noise or self-detection)
            if distance < 0.12:
                continue
            
            # Calculate beam angle in world frame
            beam_angle = (i * angle_increment - math.pi)
            lidar_x = distance * math.cos(-beam_angle)
            lidar_y = distance * math.sin(-beam_angle)
            
            # Calculate end point of the beam
            end_x = robot_x + math.cos(robot_theta) * lidar_x - math.sin(robot_theta) * lidar_y
            end_y = robot_y + math.sin(robot_theta) * lidar_x + math.cos(robot_theta) * lidar_y
            
            # Convert to grid coordinates
            end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)
            
            if not self.is_valid_cell(end_grid_x, end_grid_y):
                continue
            
            # Get all cells along the ray
            ray_cells = self.bresenham_line(robot_grid_x, robot_grid_y, 
                                           end_grid_x, end_grid_y)
            
            # Update cells along the ray (free space)
            for x, y in ray_cells[:-1]:  # Exclude the last cell (obstacle)
                if self.verification_enabled:
                    # Count free space observations
                    self.free_observations[y, x] += 1
                
                # Update: add inverse sensor model, subtract prior
                self.log_odds[y, x] += self.log_odds_free - self.log_odds_prior
                # Clamp to prevent overflow
                self.log_odds[y, x] = max(self.min_log_odds, 
                                         min(self.max_log_odds, self.log_odds[y, x]))
            
            # Update the end cell and nearby cells (occupied with thickness)
            if len(ray_cells) > 0:
                # Mark the obstacle with some thickness
                obstacle_cells = ray_cells[-min(self.obstacle_thickness, len(ray_cells)):]
                for x, y in obstacle_cells:
                    if self.verification_enabled:
                        # Count obstacle observations
                        self.obstacle_observations[y, x] += 1
                        
                        # Two-pass verification: only update if confirmed
                        if self.obstacle_observations[y, x] < self.min_observations_for_obstacle:
                            # Not enough observations yet, skip update
                            continue
                    
                    # Update: add inverse sensor model, subtract prior
                    self.log_odds[y, x] += self.log_odds_occ - self.log_odds_prior
                    self.log_odds[y, x] = max(self.min_log_odds, 
                                             min(self.max_log_odds, self.log_odds[y, x]))
        
        # Convert log-odds to probabilities
        self.grid = np.vectorize(self._log_odds_to_prob)(self.log_odds)
    
    def get_map_data(self):
        """
        Get map data for visualization
        
        Returns:
            grid: Occupancy grid (0=free, 1=occupied, 0.5=unknown)
            info: Map metadata dictionary
        """
        info = {
            'width': self.grid_width,
            'height': self.grid_height,
            'resolution': self.resolution,
            'origin_x': -self.origin_x,
            'origin_y': -self.origin_y
        }
        return self.grid.copy(), info
    
    def save_map(self, filename, robot_pose=None):
        """
        Save map to file with robot pose
        
        Args:
            filename: Base filename (without extension)
            robot_pose: Tuple of (x, y, theta) robot pose in world coordinates
        """
        # Save map grid
        np.save(filename, self.grid)
        
        # Save metadata including robot pose
        metadata = {
            'width': self.width,
            'height': self.height,
            'resolution': self.resolution,
            'origin_x': self.origin_x,
            'origin_y': self.origin_y,
            'grid_width': self.grid_width,
            'grid_height': self.grid_height
        }
        
        # Add robot pose if provided
        if robot_pose is not None:
            x, y, theta = robot_pose
            grid_x, grid_y = self.world_to_grid(x, y)
            metadata['robot_pose'] = {
                'world_x': float(x),
                'world_y': float(y),
                'theta': float(theta),
                'grid_x': int(grid_x),
                'grid_y': int(grid_y)
            }
        
        # Save metadata as JSON
        import json
        metadata_file = filename.replace('.npy', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Map saved to {filename}")
        if robot_pose is not None:
            print(f"Robot pose: ({x:.3f}, {y:.3f}, {theta:.1f}°) saved to {metadata_file}")
    
    def load_map(self, filename):
        """Load map from file"""
        self.grid = np.load(filename)
        self.log_odds = np.vectorize(lambda p: self._prob_to_log_odds(p if p != 0.5 else 0.5))(self.grid)
        print(f"Map loaded from {filename}")
    
    def get_occupied_cells(self, threshold=0.7):
        """
        Get list of occupied cells
        
        Args:
            threshold: Probability threshold for occupied
            
        Returns:
            List of (x, y) world coordinates of occupied cells
        """
        occupied = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.grid[y, x] > threshold:
                    world_x, world_y = self.grid_to_world(x, y)
                    occupied.append((world_x, world_y))
        return occupied
    
    def get_map_statistics(self):
        """Get statistics about the map"""
        total_cells = self.grid_width * self.grid_height
        unknown = np.sum((self.grid > 0.4) & (self.grid < 0.6))
        free = np.sum(self.grid < 0.4)
        occupied = np.sum(self.grid > 0.6)
        
        return {
            'total_cells': total_cells,
            'unknown': int(unknown),
            'free': int(free),
            'occupied': int(occupied),
            'explored_percent': 100.0 * (total_cells - unknown) / total_cells
        }
    
    def apply_morphological_filter(self):
        """
        Apply aggressive morphological filtering to remove noise
        Suitable for environments with horizontal and vertical walls only
        """
        # Convert to binary map (occupied vs not occupied)
        binary_map = (self.grid > 0.6).astype(np.uint8)
        
        # Aggressive opening operation: remove small isolated noise
        kernel_size = 3  # Larger kernel for more aggressive filtering
        eroded = ndimage.binary_erosion(binary_map, structure=np.ones((kernel_size, kernel_size)))
        opened = ndimage.binary_dilation(eroded, structure=np.ones((kernel_size, kernel_size)))
        
        # Closing operation: fill small holes in walls
        dilated = ndimage.binary_dilation(opened, structure=np.ones((kernel_size, kernel_size)))
        closed = ndimage.binary_erosion(dilated, structure=np.ones((kernel_size, kernel_size)))
        
        # Remove small connected components (isolated noise)
        labeled, num_features = ndimage.label(closed)
        min_size = 10  # Minimum size of valid obstacles (10 cells = 0.2m x 0.2m)
        for i in range(1, num_features + 1):
            if np.sum(labeled == i) < min_size:
                closed[labeled == i] = 0
        
        # Update grid: aggressively remove noise
        filtered_grid = self.grid.copy()
        
        # Remove noise: cells that were occupied but filtered out
        noise_mask = binary_map & ~closed
        filtered_grid[noise_mask] = 0.3  # Mark as free (not just unknown)
        
        # Fill holes: cells that should be occupied
        hole_mask = ~binary_map & closed
        filtered_grid[hole_mask] = 0.7
        
        self.grid = filtered_grid
        
        # Update log-odds accordingly
        self.log_odds = np.vectorize(lambda p: self._prob_to_log_odds(p if p != 0.5 else 0.5))(self.grid)
        
        # Update observation counts for removed noise
        if self.verification_enabled:
            # Reset observation counts for cells that were identified as noise
            self.obstacle_observations[noise_mask] = 0
        
        print(f"Morphological filter: removed {np.sum(noise_mask)} noise cells, filled {np.sum(hole_mask)} holes")
    
    def set_verification(self, enabled, min_observations=3):
        """
        Enable or disable obstacle verification
        
        Args:
            enabled: True to enable verification
            min_observations: Minimum observations required to confirm obstacle
        """
        self.verification_enabled = enabled
        self.min_observations_for_obstacle = min_observations
        print(f"[Obstacle Verification] {'Enabled' if enabled else 'Disabled'} (min observations: {min_observations})")
    
    def get_verification_statistics(self):
        """
        Get obstacle verification statistics
        
        Returns:
            dict: Statistics about observations and confirmed obstacles
        """
        confirmed_obstacles = np.sum(self.obstacle_observations >= self.min_observations_for_obstacle)
        pending_obstacles = np.sum((self.obstacle_observations > 0) & 
                                  (self.obstacle_observations < self.min_observations_for_obstacle))
        
        return {
            'confirmed_obstacles': int(confirmed_obstacles),
            'pending_obstacles': int(pending_obstacles),
            'total_obstacle_observations': int(np.sum(self.obstacle_observations)),
            'total_free_observations': int(np.sum(self.free_observations)),
            'max_observations': int(np.max(self.obstacle_observations)),
            'verification_enabled': self.verification_enabled,
            'min_observations_required': self.min_observations_for_obstacle
        }
    
    def reset_verification_counts(self):
        """
        Reset all observation counts
        """
        self.obstacle_observations.fill(0)
        self.free_observations.fill(0)
        print("[Obstacle Verification] Observation counts reset")
