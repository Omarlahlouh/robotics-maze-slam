"""
Automatic Maze Explorer
Implements wall-following and frontier-based exploration for SLAM
"""

import numpy as np
import math
from enum import Enum

class ExplorerState(Enum):
    """Explorer states"""
    EXPLORING = 1
    TURNING = 2
    MOVING_FORWARD = 3
    AVOIDING_OBSTACLE = 4
    COMPLETED = 5

class AutoExplorer:
    def __init__(self, slam_map):
        """
        Initialize automatic explorer
        
        Args:
            slam_map: OccupancyGridMap instance
        """
        self.slam_map = slam_map
        self.state = ExplorerState.EXPLORING
        
        # Exploration parameters
        self.safe_distance = 0.30  # Safe distance from obstacles (m)
        self.wall_follow_distance = 0.35  # Preferred distance from wall (m)
        self.forward_speed = 0.1   # Increased to match manual speed for faster exploration
        self.turn_speed = 0.5      # Balanced turn speed
        
        # State tracking
        self.turn_direction = 1  # 1 for left, -1 for right
        self.stuck_counter = 0
        self.last_position = (0, 0)
        self.position_history = []
        self.exploration_time = 0
        self.max_exploration_time = 30000  # ~32 minutes max (30000 * 64ms)
        self.start_position = None  # Record start position for loop closure
        self.min_exploration_time = 500  # Minimum time before checking loop closure
        
        # Wall following (right-hand rule)
        self.following_right_wall = True
        
        print("Auto Explorer initialized - Right-hand wall following")
    
    def get_control_command(self, ranges, robot_x, robot_y, robot_theta):
        """
        Get control command for autonomous exploration
        
        Args:
            ranges: LiDAR range measurements
            robot_x, robot_y: Robot position
            robot_theta: Robot orientation
            
        Returns:
            (linear_vel, angular_vel): Control velocities
        """
        if ranges is None or len(ranges) == 0:
            return 0.0, 0.0
        
        # Update exploration time
        self.exploration_time += 1
        
        # Record start position
        if self.start_position is None:
            self.start_position = (robot_x, robot_y)
        
        # Check if exploration is complete
        if self.exploration_time > self.max_exploration_time:
            print(f"Exploration timeout after {self.exploration_time} steps")
            self.state = ExplorerState.COMPLETED
            return 0.0, 0.0
        
        # Track position for stuck detection
        current_pos = (robot_x, robot_y)
        if len(self.position_history) > 50:
            self.position_history.pop(0)
        self.position_history.append(current_pos)
        
        # Analyze LiDAR data
        front_distance = self._get_sector_min_distance(ranges, -15, 15)
        right_distance = self._get_sector_min_distance(ranges, -90, -60)
        left_distance = self._get_sector_min_distance(ranges, 60, 90)
        right_front = self._get_sector_min_distance(ranges, -45, -15)
        left_front = self._get_sector_min_distance(ranges, 15, 45)
        right_back = self._get_sector_min_distance(ranges, -135, -90)  # Check right back
        
        # Wall following with right-hand rule
        linear_vel, angular_vel = self._wall_following_control(
            front_distance, right_distance, left_distance, 
            right_front, left_front, right_back
        )
        
        return linear_vel, angular_vel
    
    def _get_sector_min_distance(self, ranges, start_angle, end_angle):
        """
        Get minimum distance in a sector
        
        Args:
            ranges: LiDAR range measurements
            start_angle, end_angle: Sector angles in degrees
            
        Returns:
            Minimum distance in sector
        """
        num_points = len(ranges)
        angle_per_point = 360.0 / num_points
        
        # Convert angles to indices
        start_idx = int((start_angle + 180) / angle_per_point) % num_points
        end_idx = int((end_angle + 180) / angle_per_point) % num_points
        
        # Handle wrap-around
        if start_idx <= end_idx:
            sector_ranges = ranges[start_idx:end_idx+1]
        else:
            sector_ranges = np.concatenate([ranges[start_idx:], ranges[:end_idx+1]])
        
        # Filter valid ranges
        valid_ranges = sector_ranges[sector_ranges < 5.0]
        
        if len(valid_ranges) > 0:
            return np.min(valid_ranges)
        return 5.0
    
    def _wall_following_control(self, front, right, left, right_front, left_front, right_back):
        """
        Wall following control using right-hand rule (improved for corners)
        
        Args:
            front, right, left, right_front, left_front, right_back: Sector distances
            
        Returns:
            (linear_vel, angular_vel): Control velocities
        """
        linear_vel = 0.0
        angular_vel = 0.0
        
        # Priority 1: Emergency stop - too close to front
        if front < self.safe_distance * 0.8:
            # Very close to front wall, stop and turn left sharply
            linear_vel = 0.0
            angular_vel = self.turn_speed
            self.state = ExplorerState.TURNING
            
        # Priority 2: Corner detection - front obstacle approaching
        elif front < self.safe_distance * 1.2:
            # Approaching corner, check if should turn right (open space on right)
            if right > self.wall_follow_distance * 2.0 and right_back > self.wall_follow_distance * 1.5:
                # Open space on right, turn right to follow wall
                linear_vel = self.forward_speed * 0.4
                angular_vel = -self.turn_speed * 0.7
                self.state = ExplorerState.TURNING
            else:
                # Turn left (normal corner)
                linear_vel = self.forward_speed * 0.3
                angular_vel = self.turn_speed * 0.8
                self.state = ExplorerState.TURNING
            
        # Priority 3: Right front corner - early detection
        elif right_front < self.safe_distance * 1.2:
            # Obstacle on right front, turn left early
            linear_vel = self.forward_speed * 0.5
            angular_vel = self.turn_speed * 0.7
            self.state = ExplorerState.AVOIDING_OBSTACLE
            
        # Priority 4: Left front corner
        elif left_front < self.safe_distance * 1.2:
            # Obstacle on left front, turn right
            linear_vel = self.forward_speed * 0.5
            angular_vel = -self.turn_speed * 0.7
            self.state = ExplorerState.AVOIDING_OBSTACLE
            
        # Priority 5: Too close to right wall
        elif right < self.wall_follow_distance * 0.6:
            # Too close to right wall, turn left
            linear_vel = self.forward_speed * 0.6
            angular_vel = self.turn_speed * 0.6
            self.state = ExplorerState.EXPLORING
            
        # Priority 6: Too far from right wall
        elif right > self.wall_follow_distance * 1.8:
            # Too far from right wall, turn right to follow it
            linear_vel = self.forward_speed * 0.7
            angular_vel = -self.turn_speed * 0.5
            self.state = ExplorerState.EXPLORING
            
        # Priority 7: Move forward along wall with fine adjustment
        else:
            # Good distance from walls, move forward
            linear_vel = self.forward_speed
            
            # Fine-tune to maintain wall distance
            if right < self.wall_follow_distance * 0.9:
                angular_vel = self.turn_speed * 0.3  # Turn slightly left
            elif right > self.wall_follow_distance * 1.1:
                angular_vel = -self.turn_speed * 0.3  # Turn slightly right
            else:
                angular_vel = 0.0
            
            self.state = ExplorerState.MOVING_FORWARD
        
        return linear_vel, angular_vel
    
    def is_exploration_complete(self):
        """Check if exploration is complete"""
        if self.state == ExplorerState.COMPLETED:
            return True
        
        # Check map exploration percentage
        stats = self.slam_map.get_map_statistics()
        if stats['explored_percent'] > 85.0:
            print(f"Exploration complete: {stats['explored_percent']:.1f}% explored")
            self.state = ExplorerState.COMPLETED
            return True
        
        # Check if robot returned to start (loop closure) after minimum exploration time
        if self.exploration_time > self.min_exploration_time and self.start_position is not None:
            import math
            dx = self.position_history[-1][0] - self.start_position[0]
            dy = self.position_history[-1][1] - self.start_position[1]
            distance_to_start = math.sqrt(dx**2 + dy**2)
            
            if distance_to_start < 0.2 and stats['explored_percent'] > 60.0:
                print(f"Loop closure detected! Returned to start position.")
                print(f"Exploration: {stats['explored_percent']:.1f}% explored")
                self.state = ExplorerState.COMPLETED
                return True
        
        return False
    
    def get_state_string(self):
        """Get current state as string"""
        state_names = {
            ExplorerState.EXPLORING: "Exploring",
            ExplorerState.TURNING: "Turning",
            ExplorerState.MOVING_FORWARD: "Moving Forward",
            ExplorerState.AVOIDING_OBSTACLE: "Avoiding Obstacle",
            ExplorerState.COMPLETED: "Completed"
        }
        return state_names.get(self.state, "Unknown")
    
    def reset(self):
        """Reset explorer state"""
        self.state = ExplorerState.EXPLORING
        self.stuck_counter = 0
        self.position_history = []
        self.exploration_time = 0
        print("Auto Explorer reset")
