"""
Path Planner for Navigation
Implements A* path planning and path following
"""

import numpy as np
import math
from heapq import heappush, heappop

class PathPlanner:
    def __init__(self, slam_map):
        """
        Initialize path planner
        
        Args:
            slam_map: OccupancyGridMap instance
        """
        self.slam_map = slam_map
        self.current_path = []
        self.path_index = 0
        self.goal_reached = False
        
        # Navigation parameters
        self.waypoint_tolerance = 0.10  # Distance to consider waypoint reached (m) - 10cm for precise cornering
        self.goal_tolerance = 0.20  # Distance to consider goal reached (m) - increased from 0.15
        self.max_speed = 0.06  # Further reduced from 0.1 to 0.06 for safer navigation
        self.turn_speed = 0.3  # Further reduced from 0.5 to 0.3 for gentler turns
        self.lookahead_distance = 0.20  # Look ahead distance - increased for smoother cornering
        
        # Safety parameters
        self.safety_margin = 2  # Safety margin in grid cells (avoid walls) - 2 cells = 4cm for 8cm robot
        
        print("Path Planner initialized")
        print(f"Safety margin: {self.safety_margin} cells ({self.safety_margin * slam_map.resolution:.3f}m)")
    
    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """
        Plan path from start to goal using A*
        
        Args:
            start_x, start_y: Start position in world coordinates
            goal_x, goal_y: Goal position in world coordinates
            
        Returns:
            List of (x, y) waypoints in world coordinates, or None if no path found
        """
        # Convert to grid coordinates
        start_grid = self.slam_map.world_to_grid(start_x, start_y)
        goal_grid = self.slam_map.world_to_grid(goal_x, goal_y)
        
        # Check if start and goal are valid
        if not self.slam_map.is_valid_cell(*start_grid):
            print(f"Start position {start_grid} is outside map")
            return None
        
        if not self.slam_map.is_valid_cell(*goal_grid):
            print(f"Goal position {goal_grid} is outside map")
            return None
        
        # Get map data
        grid, _ = self.slam_map.get_map_data()
        
        # Check if goal is occupied
        if grid[goal_grid[1], goal_grid[0]] > 0.6:
            print(f"Goal position is occupied")
            return None
        
        # A* search
        path_grid = self._astar_search(grid, start_grid, goal_grid)
        
        if path_grid is None:
            print("No path found")
            return None
        
        # Convert grid path to world coordinates
        path_world = []
        for gx, gy in path_grid:
            wx, wy = self.slam_map.grid_to_world(gx, gy)
            path_world.append((wx, wy))
        
        # Simplify path (remove redundant waypoints)
        path_world = self._simplify_path(path_world)
        
        # Smooth corners for better turning
        path_world = self._smooth_corners(path_world)
        
        self.current_path = path_world
        self.path_index = 0
        self.goal_reached = False
        
        print(f"Path planned: {len(path_world)} waypoints (after smoothing)")
        return path_world
    
    def _astar_search(self, grid, start, goal):
        """
        A* path planning algorithm
        
        Args:
            grid: Occupancy grid
            start: (x, y) start grid coordinates
            goal: (x, y) goal grid coordinates
            
        Returns:
            List of (x, y) grid coordinates, or None if no path found
        """
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        def is_free(x, y):
            if not self.slam_map.is_valid_cell(x, y):
                return False
            # Consider cell free if probability < 0.5
            if grid[y, x] >= 0.5:
                return False
            
            # Check safety margin - avoid cells near obstacles
            for dy in range(-self.safety_margin, self.safety_margin + 1):
                for dx in range(-self.safety_margin, self.safety_margin + 1):
                    nx, ny = x + dx, y + dy
                    if self.slam_map.is_valid_cell(nx, ny):
                        if grid[ny, nx] > 0.6:  # Obstacle nearby
                            return False
            
            return True
        
        # 8-connected neighbors
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not is_free(*neighbor):
                    continue
                
                # Cost is higher for diagonal moves
                move_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def _simplify_path(self, path):
        """Simplify path by removing redundant waypoints"""
        if len(path) < 3:
            return path
        
        simplified = [path[0]]
        
        for i in range(1, len(path) - 1):
            prev = simplified[-1]
            curr = path[i]
            next_pt = path[i + 1]
            
            # Check if current point is on the line between prev and next
            dx1 = curr[0] - prev[0]
            dy1 = curr[1] - prev[1]
            dx2 = next_pt[0] - curr[0]
            dy2 = next_pt[1] - curr[1]
            
            # Calculate angle difference
            angle1 = math.atan2(dy1, dx1)
            angle2 = math.atan2(dy2, dx2)
            angle_diff = abs(angle1 - angle2)
            
            # Keep waypoint if direction changes significantly
            if angle_diff > 0.3:  # ~17 degrees
                simplified.append(curr)
        
        simplified.append(path[-1])
        return simplified
    
    def _smooth_corners(self, path, corner_radius=0.80):
        """
        Smooth sharp corners to increase turning radius
        
        Args:
            path: List of (x, y) waypoints
            corner_radius: Radius for corner smoothing (meters)
            
        Returns:
            Smoothed path
        """
        if len(path) < 3:
            return path
        
        smoothed = [path[0]]  # Keep start point
        
        for i in range(1, len(path) - 1):
            prev = path[i - 1]
            curr = path[i]
            next_pt = path[i + 1]
            
            # Calculate direction vectors
            dx1 = curr[0] - prev[0]
            dy1 = curr[1] - prev[1]
            dx2 = next_pt[0] - curr[0]
            dy2 = next_pt[1] - curr[1]
            
            # Calculate angles
            angle1 = math.atan2(dy1, dx1)
            angle2 = math.atan2(dy2, dx2)
            angle_diff = abs(angle1 - angle2)
            
            # Normalize angle difference
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            
            # If this is a sharp corner (> 30 degrees), add smoothing points
            if angle_diff > 0.52:  # ~30 degrees
                # Distance from corner to smoothing point
                dist1 = math.sqrt(dx1**2 + dy1**2)
                dist2 = math.sqrt(dx2**2 + dy2**2)
                
                # Ensure we have enough distance to smooth
                smooth_dist = min(corner_radius, dist1 * 0.4, dist2 * 0.4)
                
                if smooth_dist > 0.05:  # Only smooth if we have enough space
                    # Calculate smoothing points before and after the corner
                    t1 = smooth_dist / dist1 if dist1 > 0 else 0
                    t2 = smooth_dist / dist2 if dist2 > 0 else 0
                    
                    # Point before corner
                    p_before = (
                        curr[0] - dx1 * t1,
                        curr[1] - dy1 * t1
                    )
                    
                    # Point after corner
                    p_after = (
                        curr[0] + dx2 * t2,
                        curr[1] + dy2 * t2
                    )
                    
                    # Add smoothing points
                    smoothed.append(p_before)
                    
                    # Add interpolated point at corner (for smoother arc)
                    mid_x = (p_before[0] + p_after[0]) / 2
                    mid_y = (p_before[1] + p_after[1]) / 2
                    smoothed.append((mid_x, mid_y))
                    
                    smoothed.append(p_after)
                else:
                    smoothed.append(curr)
            else:
                # Not a sharp corner, keep original point
                smoothed.append(curr)
        
        smoothed.append(path[-1])  # Keep end point
        
        return smoothed
    
    def get_control_command(self, robot_x, robot_y, robot_theta):
        """
        Get control command to follow current path
        
        Args:
            robot_x, robot_y: Robot position
            robot_theta: Robot orientation
            
        Returns:
            (linear_vel, angular_vel): Control velocities
        """
        if not self.current_path or self.goal_reached:
            return 0.0, 0.0
        
        # Get current target waypoint
        if self.path_index >= len(self.current_path):
            self.goal_reached = True
            print("Goal reached!")
            return 0.0, 0.0
        
        target_x, target_y = self.current_path[self.path_index]
        
        # Calculate distance and angle to target
        dx = target_x - robot_x
        dy = target_y - robot_y
        distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)
        
        # Calculate angle error
        angle_error = target_angle - robot_theta
        # Normalize to [-pi, pi]
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi
        
        # Debug output (every 50 calls, ~3 seconds)
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 50 == 0:
            print(f"[NAV] Waypoint {self.path_index}/{len(self.current_path)}: Target=({target_x:.2f},{target_y:.2f}), Dist={distance:.3f}m, AngleErr={math.degrees(angle_error):.1f}°")
        
        # Lookahead for corner detection
        is_corner = self._is_approaching_corner(self.path_index)
        
        # Check if waypoint is reached
        # Use larger tolerance for corners to avoid getting stuck
        base_tolerance = self.goal_tolerance if self.path_index == len(self.current_path) - 1 else self.waypoint_tolerance
        tolerance = base_tolerance * 1.5 if is_corner else base_tolerance
        
        # Skip waypoint if very close (avoid spinning in place)
        if distance < tolerance:
            print(f"Reached waypoint {self.path_index + 1}/{len(self.current_path)}")
            self.path_index += 1
            if self.path_index >= len(self.current_path):
                self.goal_reached = True
                print("Goal reached!")
                return 0.0, 0.0
            return self.get_control_command(robot_x, robot_y, robot_theta)
        
        # REMOVED: Skip waypoint logic (was causing robot to spin at start)
        # Trust the path planning and let the robot navigate to waypoints even with large angle errors
        
        # Pure pursuit control with improved turning behavior
        # Apply extra caution at corners
        corner_factor = 0.5 if is_corner else 1.0
        
        # If angle error is very large, slow down significantly but keep moving forward
        if abs(angle_error) > 1.0:  # > 57 degrees
            # Very slow forward motion to allow turning to complete
            linear_vel = self.max_speed * 0.03  # Further reduced to 3%
            angular_vel = 1.0 * angle_error * corner_factor  # Stronger turning rate
            angular_vel = max(-self.turn_speed, min(self.turn_speed, angular_vel))
            if is_corner:
                print(f"⚠️ Sharp corner ahead - turning carefully")
        elif abs(angle_error) > 0.5:  # > 28 degrees - very significant misalignment
            # Much slower - focus on turning first
            linear_vel = self.max_speed * 0.1  # Only 10% to allow turning
            angular_vel = 1.0 * angle_error * corner_factor
            angular_vel = max(-self.turn_speed, min(self.turn_speed, angular_vel))
        elif abs(angle_error) > 0.3:  # > 17 degrees - still significant
            # Slow down and turn - still needs significant alignment  
            linear_vel = self.max_speed * (0.15 if is_corner else 0.2)  # Reduced from 0.2/0.3
            angular_vel = 1.0 * angle_error * corner_factor
            angular_vel = max(-self.turn_speed, min(self.turn_speed, angular_vel))
        elif abs(angle_error) > 0.1:  # > 5.7 degrees - needs careful alignment
            # Slow down more for precise alignment in narrow corridors
            linear_vel = self.max_speed * (0.3 if is_corner else 0.4)
            angular_vel = 1.0 * angle_error * corner_factor  # Stronger turning
            angular_vel = max(-self.turn_speed * 0.8, min(self.turn_speed * 0.8, angular_vel))
        elif abs(angle_error) > 0.05:  # > 2.9 degrees - fine alignment needed
            # Near-aligned but still needs adjustment for narrow corridors
            linear_vel = self.max_speed * (0.5 if is_corner else 0.7)
            angular_vel = 0.8 * angle_error * corner_factor
            angular_vel = max(-self.turn_speed * 0.6, min(self.turn_speed * 0.6, angular_vel))
        else:
            # Very well aligned - safe to go faster
            linear_vel = self.max_speed * (0.8 if is_corner else 1.0)
            angular_vel = 0.5 * angle_error
            angular_vel = max(-self.turn_speed * 0.5, min(self.turn_speed * 0.5, angular_vel))
        
        return linear_vel, angular_vel
    
    def _is_approaching_corner(self, current_index, lookahead=3):
        """
        Check if approaching a corner by looking ahead in the path
        
        Args:
            current_index: Current waypoint index
            lookahead: Number of waypoints to look ahead
            
        Returns:
            True if approaching a sharp corner
        """
        if current_index >= len(self.current_path) - lookahead:
            return False
        
        # Get current and lookahead waypoints
        current = self.current_path[current_index]
        lookahead_point = self.current_path[min(current_index + lookahead, len(self.current_path) - 1)]
        
        if current_index > 0:
            prev = self.current_path[current_index - 1]
            
            # Calculate direction vectors
            dx1 = current[0] - prev[0]
            dy1 = current[1] - prev[1]
            dx2 = lookahead_point[0] - current[0]
            dy2 = lookahead_point[1] - current[1]
            
            # Calculate angle between vectors
            angle1 = math.atan2(dy1, dx1)
            angle2 = math.atan2(dy2, dx2)
            angle_diff = abs(angle1 - angle2)
            
            # Normalize angle difference to [0, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            angle_diff = abs(angle_diff)
            
            # Consider it a corner if angle change > 30 degrees
            return angle_diff > 0.52  # ~30 degrees
        
        return False
    
    def is_goal_reached(self):
        """Check if goal is reached"""
        return self.goal_reached
    
    def reset(self):
        """Reset planner state"""
        self.current_path = []
        self.path_index = 0
        self.goal_reached = False
        print("Path Planner reset")
