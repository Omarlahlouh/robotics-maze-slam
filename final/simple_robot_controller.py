"""
Simple Robot Controller for SLAM Navigation
Stage 1: Basic differential drive robot with keyboard control
"""

from controller import Robot, Motor, PositionSensor, Keyboard, Lidar
import math
import numpy as np
from occupancy_grid_map import OccupancyGridMap
from auto_explorer import AutoExplorer
from path_planner import PathPlanner
from map_visualizer import MapVisualizer
from enum import Enum

class SimpleRobotController:
    def __init__(self):
        # Initialize robot
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Robot physical parameters (similar to e-puck)
        self.wheel_radius = 0.0205  # 2.05cm wheel radius (like e-puck)
        self.wheel_base = 0.074     # 7.4cm distance between wheels (0.037 * 2)
        self.max_speed = 6.28       # Maximum wheel speed (rad/s)
        
        # Initialize motors
        self.left_motor = self.robot.getDevice('left_motor')
        self.right_motor = self.robot.getDevice('right_motor')
        
        # Set motors to velocity mode
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        # Initialize encoders
        self.left_encoder = self.robot.getDevice('left_encoder')
        self.right_encoder = self.robot.getDevice('right_encoder')
        self.left_encoder.enable(self.timestep)
        self.right_encoder.enable(self.timestep)
        
        # Initialize keyboard
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.timestep)
        
        # Initialize LiDAR
        self.lidar = self.robot.getDevice('lidar')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        
        # Initialize GPS for accurate positioning
        self.gps = self.robot.getDevice('gps')
        if self.gps:
            self.gps.enable(self.timestep)
            self.use_gps = True
            print("GPS sensor enabled for accurate positioning")
        else:
            self.use_gps = False
            print("GPS not available, using odometry only")
        
        # Initialize IMU for accurate orientation
        self.imu = self.robot.getDevice('imu')
        if self.imu:
            try:
                self.imu.enable(self.timestep)
                self.use_imu = True
                print("IMU sensor enabled for accurate orientation")
            except:
                self.use_imu = False
                print("IMU initialization failed, using odometry for orientation")
        else:
            self.use_imu = False
            print("IMU not available, using odometry for orientation")
        
        # Temporary fix: use initial GPS position to calculate initial theta
        if self.use_gps and self.gps:
            # Wait one step for GPS to initialize
            self.robot.step(self.timestep)
            gps_init = self.gps.getValues()
            self.x = gps_init[0]
            self.y = gps_init[1]
            print(f"Initial GPS position: ({self.x:.3f}, {self.y:.3f})")
        
        # Robot state (initialize with Webots starting position)
        self.x = -1.5     # Robot position x (matches Webots translation)
        self.y = 0.0      # Robot position y
        self.theta = 0.0  # Robot orientation
        self.theta_offset = 0.0  # Orientation offset for GPS mode
        
        # Previous encoder values for odometry
        self.prev_left_encoder = 0.0
        self.prev_right_encoder = 0.0
        
        # Initialize SLAM map
        self.slam_map = OccupancyGridMap(width=4.0, height=2.0, resolution=0.02)
        self.map_update_counter = 0
        self.map_update_interval = 15  # Update map every 15 timesteps (no downsampling, reduce frequency)
        
        # Motion state tracking for filtering
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        self.angular_vel_threshold = 0.6   # rad/s, increased to allow faster turns (0.5 is ok)
        self.linear_vel_threshold = 0.15   # m/s, slightly increased
        
        # Initialize auto explorer and path planner
        self.auto_explorer = AutoExplorer(self.slam_map)
        self.path_planner = PathPlanner(self.slam_map)
        
        # Initialize real-time map visualizer
        self.visualizer = MapVisualizer(self.slam_map, update_interval=1000)
        self.visualizer.start()
        
        # Control modes
        self.mode = 1  # 1=Manual, 2=Auto Explore, 3=Auto Navigate
        self.navigation_goal = (1.5, -0.6)  # Default goal position for mode 3 (exit)
        self.loaded_map_file = None  # Loaded map for navigation
        self.navigation_path_loaded = False  # Flag for path loaded
        
        # Stuck detection variables
        self.last_position = (self.x, self.y)
        self.position_update_time = 0
        self.stuck_threshold = 8.0  # seconds (increased to allow slow turning)
        self.min_movement_threshold = 0.05  # meters
        self.in_recovery_mode = False
        
        # Side wall avoidance state
        self.side_wall_recovery_active = False
        self.side_wall_recovery_start_time = 0
        self.side_wall_recovery_duration = 2.0  # seconds for recovery maneuver
        
        print("SimpleBot initialized successfully!")
        print("\n=== Control Modes ===")
        print("  1: Manual Control (WASD)")
        print("  2: Auto Exploration (SLAM Mapping)")
        print("  3: Auto Navigation (Load map & path, go to exit)")
        print("  L: Load saved map and navigation path")
        print("\n=== Manual Controls ===")
        print("  W/S: Forward/Backward")
        print("  A/D: Turn Left/Right")
        print("  Space: Stop")
        print("  M: Save Map")
        print("\n=== Current Mode: 1 (Manual) ===")
        print("Wheel configuration: Left wheel = negative Y, Right wheel = positive Y")
        
        # Test motor initialization
        print(f"Left motor max velocity: {self.left_motor.getMaxVelocity()}")
        print(f"Right motor max velocity: {self.right_motor.getMaxVelocity()}")
        print("Ready to receive keyboard input...")
    
    def _replan_from_current_position(self):
        """Replan path from current robot position to goal"""
        goal_x, goal_y = self.navigation_goal
        
        print(f"📍 Replanning from current position ({self.x:.2f}, {self.y:.2f}) to goal ({goal_x:.2f}, {goal_y:.2f})")
        
        # Plan new path
        path = self.path_planner.plan_path(self.x, self.y, goal_x, goal_y)
        
        if path:
            print(f"✅ Path replanned successfully!")
            print(f"   New waypoints: {len(path)}")
            path_length = sum([
                math.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)
                for i in range(len(path) - 1)
            ])
            print(f"   Path length: {path_length:.2f}m")
            
            # Update visualizer
            self.visualizer.set_planned_path(path, 0)
            self.navigation_path_loaded = True
        else:
            print("❌ Failed to replan path!")
            print("   Possible reasons:")
            print("   - Current position is in obstacle")
            print("   - No path exists to goal")
            print("   Switching back to manual mode")
            self.mode = 1
    
    def load_map_and_path(self, map_file='slam_map_20251128_140658_edited_20251128_141828.npy', 
                          path_file='navigation_path.json'):
        """Load saved map and navigation path"""
        import os
        import json
        
        # Construct full paths
        map_path = os.path.join(os.path.dirname(__file__), map_file)
        path_path = os.path.join(os.path.dirname(__file__), path_file)
        
        try:
            # Load map
            import numpy as np
            grid = np.load(map_path)
            self.slam_map.grid = grid
            self.loaded_map_file = map_file
            print(f"\n✓ Map loaded: {map_file}")
            print(f"  Grid shape: {grid.shape}")
            
            # Load path
            with open(path_path, 'r') as f:
                path_data = json.load(f)
            
            # Convert path to waypoints
            path = [(p['x'], p['y']) for p in path_data['path']]
            self.path_planner.current_path = path
            # Skip waypoint 0 (start position) - robot already at/near start
            self.path_planner.path_index = 1  # Start from first actual navigation waypoint
            self.path_planner.goal_reached = False
            self.navigation_path_loaded = True
            
            print(f"✓ Path loaded: {path_file}")
            print(f"  Waypoints: {len(path)}")
            print(f"  ⏭️  Skipping waypoint 0 (start position), beginning from waypoint 1")
            print(f"  Path length: {path_data['path_length']:.2f}m")
            print(f"  Start: ({path_data['start']['x']:.2f}, {path_data['start']['y']:.2f})")
            print(f"  Goal: ({path_data['goal']['x']:.2f}, {path_data['goal']['y']:.2f})")
            print("\n✅ Ready for auto navigation! Press '3' to start.")
            
            # Update visualizer with planned path
            self.visualizer.set_planned_path(path, 0)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Failed to load map or path: {e}")
            print("   Make sure files exist in controllers/simple_robot_controller/")
            return False
        
    def get_keyboard_input(self):
        """Get keyboard input and return desired velocities"""
        linear_vel = 0.0
        angular_vel = 0.0
        
        key = self.keyboard.getKey()
        
        # Mode switching
        if key == ord('1'):
            self.switch_mode(1)
        elif key == ord('2'):
            self.switch_mode(2)
        elif key == ord('3'):
            self.switch_mode(3)
        
        # Manual control (only in mode 1)
        if self.mode == 1:
            if key == ord('W') or key == ord('w'):  # Forward
                linear_vel = 0.1  # Reduced from 0.2
                print("Moving forward")
            elif key == ord('S') or key == ord('s'):  # Backward
                linear_vel = -0.1  # Reduced from -0.2
                print("Moving backward")
            elif key == ord('A') or key == ord('a'):  # Turn left
                angular_vel = 0.5  # Increased for faster mapping
                print("Turning left")
            elif key == ord('D') or key == ord('d'):  # Turn right
                angular_vel = -0.5  # Increased for faster mapping
                print("Turning right")
            elif key == ord(' '):  # Stop
                linear_vel = 0.0
                angular_vel = 0.0
                print("Stop")
        
        # Common commands (all modes)
        if key == ord('M') or key == ord('m'):  # Save map
            self.save_slam_map()
            print("Map saved!")
        elif key == ord('L') or key == ord('l'):  # Load map and path
            self.load_map_and_path()
            
        return linear_vel, angular_vel
    
    def switch_mode(self, new_mode):
        """Switch control mode"""
        if new_mode == self.mode:
            return
        
        self.mode = new_mode
        
        if new_mode == 1:
            print("\n=== Mode 1: Manual Control ===")
            print("Use WASD to control the robot")
            self.auto_explorer.reset()
            self.path_planner.reset()
        
        elif new_mode == 2:
            print("\n=== Mode 2: Auto Exploration ===")
            print("Robot will automatically explore and build map")
            print("Press 1 to return to manual control")
            self.auto_explorer.reset()
        
        elif new_mode == 3:
            print("\n=== Mode 3: Auto Navigation ===")
            if not self.loaded_map_file:
                print("❌ No map loaded!")
                print("   Please press 'L' to load map first.")
                self.mode = 1  # Return to manual mode
                return
            
            # Check if we need to replan from current position
            print(f"Current robot position: ({self.x:.2f}, {self.y:.2f})")
            
            if self.navigation_path_loaded:
                planned_start = self.path_planner.current_path[0] if self.path_planner.current_path else None
                if planned_start:
                    # Check if robot is far from planned start
                    dist_to_start = math.sqrt((self.x - planned_start[0])**2 + (self.y - planned_start[1])**2)
                    print(f"Distance to planned start: {dist_to_start:.2f}m")
                    
                    if dist_to_start > 0.3:  # More than 30cm away
                        print("⚠️ Robot position differs from planned start!")
                        print("   Replanning path from current position...")
                        self._replan_from_current_position()
                    else:
                        print("✅ Using loaded navigation path")
                        print(f"   Waypoints: {len(self.path_planner.current_path)}")
            else:
                print("⚠️ No path loaded, planning from current position...")
                self._replan_from_current_position()
            
            print("   Robot will navigate to exit automatically")
            print("   Press 1 to stop and return to manual control")
    
    def differential_drive_kinematics(self, linear_vel, angular_vel):
        """Convert linear and angular velocities to wheel speeds"""
        # Differential drive kinematics (corrected for proper turning direction)
        left_wheel_vel = (linear_vel + angular_vel * self.wheel_base / 2.0) / self.wheel_radius
        right_wheel_vel = (linear_vel - angular_vel * self.wheel_base / 2.0) / self.wheel_radius
        
        # Limit wheel speeds
        left_wheel_vel = max(-self.max_speed, min(self.max_speed, left_wheel_vel))
        right_wheel_vel = max(-self.max_speed, min(self.max_speed, right_wheel_vel))
        
        return left_wheel_vel, right_wheel_vel
    
    def update_position_gps(self):
        """Update robot position using GPS and IMU (accurate positioning)"""
        if self.gps:
            gps_values = self.gps.getValues()
            # GPS provides X, Y, Z coordinates
            self.x = gps_values[0]
            self.y = gps_values[1]
            # Z coordinate ignored for 2D SLAM
        
        # Use IMU for orientation if available, otherwise use odometry
        if self.use_imu and self.imu:
            # IMU provides roll, pitch, yaw
            rpy = self.imu.getRollPitchYaw()
            # Yaw angle (rotation around Z axis)
            # Webots IMU: yaw=0 means robot facing +X (East)
            # We need: theta=0 means robot facing +Y (North)
            # So we need to rotate by -90° (or -pi/2)
            self.theta = rpy[2] - math.pi / 2
            
            # Normalize angle
            while self.theta > math.pi:
                self.theta -= 2 * math.pi
            while self.theta < -math.pi:
                self.theta += 2 * math.pi
            
            # Debug: print IMU values occasionally
            import random
            if random.random() < 0.01:  # 1% chance to print
                print(f"IMU: yaw={math.degrees(rpy[2]):.1f}° → theta={math.degrees(self.theta):.1f}°")
        else:
            # Fallback to odometry for orientation
            left_encoder = self.left_encoder.getValue()
            right_encoder = self.right_encoder.getValue()
            
            delta_left = left_encoder - self.prev_left_encoder
            delta_right = right_encoder - self.prev_right_encoder
            
            left_distance = delta_left * self.wheel_radius
            right_distance = delta_right * self.wheel_radius
            
            delta_theta = (left_distance - right_distance) / self.wheel_base
            self.theta += delta_theta
            
            # Normalize angle
            while self.theta > math.pi:
                self.theta -= 2 * math.pi
            while self.theta < -math.pi:
                self.theta += 2 * math.pi
            
            self.prev_left_encoder = left_encoder
            self.prev_right_encoder = right_encoder
    
    def update_odometry_orientation(self):
        """Update only robot orientation using odometry (for use with GPS position)"""
        left_encoder = self.left_encoder.getValue()
        right_encoder = self.right_encoder.getValue()
        
        delta_left = left_encoder - self.prev_left_encoder
        delta_right = right_encoder - self.prev_right_encoder
        
        left_distance = delta_left * self.wheel_radius
        right_distance = delta_right * self.wheel_radius
        
        delta_theta = (left_distance - right_distance) / self.wheel_base
        self.theta += delta_theta
        
        # Normalize angle
        while self.theta > math.pi:
            self.theta -= 2 * math.pi
        while self.theta < -math.pi:
            self.theta += 2 * math.pi
        
        self.prev_left_encoder = left_encoder
        self.prev_right_encoder = right_encoder
    
    def update_odometry(self):
        """Update robot position using wheel encoders (fallback if no GPS)"""
        # Get current encoder values
        left_encoder = self.left_encoder.getValue()
        right_encoder = self.right_encoder.getValue()
        
        # Calculate wheel displacements
        delta_left = left_encoder - self.prev_left_encoder
        delta_right = right_encoder - self.prev_right_encoder
        
        # Convert to linear displacements
        left_distance = delta_left * self.wheel_radius
        right_distance = delta_right * self.wheel_radius
        
        # Calculate robot displacement and rotation (corrected for proper direction)
        delta_distance = (left_distance + right_distance) / 2.0
        delta_theta = (left_distance - right_distance) / self.wheel_base
        
        # Update robot pose
        self.theta += delta_theta
        self.x += delta_distance * math.cos(self.theta)
        self.y += delta_distance * math.sin(self.theta)
        
        # Normalize angle
        while self.theta > math.pi:
            self.theta -= 2 * math.pi
        while self.theta < -math.pi:
            self.theta += 2 * math.pi
        
        # Update previous encoder values
        self.prev_left_encoder = left_encoder
        self.prev_right_encoder = right_encoder
    
    def process_lidar_data(self):
        """Process LiDAR data and return range measurements"""
        # Get range image (distances)
        range_image = self.lidar.getRangeImage()
        if range_image:
            ranges = np.array(range_image)
            # Filter out invalid readings (inf means no detection)
            ranges = np.where(ranges == float('inf'), 5.0, ranges)
            
            # Debug: Check for suspicious readings
            very_close = np.sum(ranges < 0.1)
            if very_close > 50:
                print(f"WARNING: {very_close} LiDAR points < 0.1m (possible self-detection)")
            
            return ranges
        return None
    
    def get_lidar_obstacles(self, ranges, min_distance=0.3):
        """Detect obstacles from LiDAR data"""
        if ranges is None:
            return []
        
        obstacles = []
        for i, distance in enumerate(ranges):
            if distance < min_distance:
                angle = (i / len(ranges)) * 2 * math.pi - math.pi  # Convert to radians
                obstacles.append((angle, distance))
        
        return obstacles
    
    def _apply_obstacle_avoidance(self, linear_vel, angular_vel, ranges):
        """
        Apply dynamic obstacle avoidance during navigation
        
        Args:
            linear_vel: Planned linear velocity
            angular_vel: Planned angular velocity
            ranges: LiDAR range data
            
        Returns:
            (linear_vel, angular_vel): Adjusted velocities
        """
        # Define sectors for obstacle detection
        # Front: -30° to +30°
        # Left: 30° to 90°
        # Right: -90° to -30°
        
        num_points = len(ranges)
        angle_increment = 360.0 / num_points
        
        # Safety distances
        critical_distance = 0.12  # Stop if obstacle within 12cm
        warning_distance = 0.10   # Slow down if obstacle within 10cm (reduced from 0.20 to avoid over-sensitivity during turns)
        safe_distance = 0.15      # Alert if obstacle within 30cm
        
        # Detect obstacles in different sectors
        front_min = float('inf')
        left_min = float('inf')
        right_min = float('inf')
        
        for i, r in enumerate(ranges):
            # Filter out invalid readings and self-detection
            if r > 5.0 or r < 0.12:  # Skip invalid and self-detection (<12cm)
                continue
                
            angle = (i * angle_increment) % 360
            if angle > 180:
                angle -= 360
            
            # Front sector (-30° to +30°)
            if -30 <= angle <= 30:
                front_min = min(front_min, r)
            # Left sector (30° to 90°)
            elif 30 < angle <= 90:
                left_min = min(left_min, r)
            # Right sector (-90° to -30°)
            elif -90 <= angle < -30:
                right_min = min(right_min, r)
        
        # Apply obstacle avoidance logic
        original_linear = linear_vel
        original_angular = angular_vel
        
        # Check if robot is actively turning (large angular velocity indicates turning)
        # During turns, the "front obstacle" may just be a wall we're turning away from
        is_turning = abs(angular_vel) > 0.15  # Turning if angular vel > 0.15 rad/s (~8.6 degrees/s)
        
        # Critical: Stop if too close to front obstacle
        if front_min < critical_distance:
            print(f"⚠️ CRITICAL: Front obstacle at {front_min:.2f}m - STOPPING")
            linear_vel = 0.0
            # Try to turn away from obstacle
            if left_min > right_min:
                angular_vel = 0.3  # Turn left
            else:
                angular_vel = -0.3  # Turn right
        
        # Warning: Slow down if obstacle ahead
        elif front_min < warning_distance:
            print(f"⚠️ WARNING: Front obstacle at {front_min:.2f}m - SLOWING")
            linear_vel *= 0.3  # Reduce to 30% speed
            # Gentle turn away
            if left_min > right_min:
                angular_vel += 0.2
            else:
                angular_vel -= 0.2
        
        # Safe: Reduce speed slightly
        elif front_min < safe_distance:
            # During turns, reduce slowdown effect (obstacle may be wall we're turning away from)
            if is_turning:
                linear_vel *= 0.99  # Only reduce to 95% speed when turning (minimal impact)
                print(f"ℹ️ CAUTION: Front obstacle at {front_min:.2f}m (turning)")
            else:
                linear_vel *= 0.9  # Reduce to 60% speed
                print(f"ℹ️ CAUTION: Front obstacle at {front_min:.2f}m")
                
                # If angular velocity is very small, actively assist with turning
                # This helps when path planning gives too small turn angle
                if abs(angular_vel) < 0.20:  # Not actively turning (< 5.7 degrees/s)
                    # Add turning assistance based on side clearance
                    turn_assist = 0.35  # Base assist angle
                    if left_min > right_min + 0.01:  # More space on left
                        angular_vel += turn_assist  # Turn left
                        print(f"🔄 Turn assist: +{turn_assist} rad/s (left space: {left_min:.2f}m)")
                    elif right_min > left_min + 0.01:  # More space on right
                        angular_vel -= turn_assist  # Turn right
                        print(f"🔄 Turn assist: -{turn_assist} rad/s (right space: {right_min:.2f}m)")
        
        # SIMPLIFIED SIDE WALL AVOIDANCE - Only for extreme cases (trust path planning)
        # Only act if side wall is critically close (robot body width is ~8cm)
        side_wall_critical = 0.07  # 7cm - only trigger for imminent collision
        
        if left_min < side_wall_critical and left_min < right_min:
            # Left wall critically close - gentle correction
            print(f"🚨 LEFT WALL TOO CLOSE ({left_min:.2f}m) - CORRECTING")
            linear_vel *= 0.7  # Reduce speed slightly
            angular_vel -= 0.2  # Gentle right correction
        elif right_min < side_wall_critical and right_min < left_min:
            # Right wall critically close - gentle correction
            print(f"🚨 RIGHT WALL TOO CLOSE ({right_min:.2f}m) - CORRECTING")
            linear_vel *= 0.7  # Reduce speed slightly
            angular_vel += 0.2  # Gentle left correction
        
        # Log if adjustment was made (throttled output)
        if not hasattr(self, '_avoidance_log_counter'):
            self._avoidance_log_counter = 0
        self._avoidance_log_counter += 1
        
        if abs(linear_vel - original_linear) > 0.01 or abs(angular_vel - original_angular) > 0.01:
            if self._avoidance_log_counter % 10 == 0:  # Log every 10th adjustment
                print(f"🛡️ Obstacle avoidance: v={linear_vel:.2f} (was {original_linear:.2f}), " +
                      f"ω={angular_vel:.2f} (was {original_angular:.2f})")
        
        return linear_vel, angular_vel
    
    def save_slam_map(self):
        """Save SLAM map to file with robot pose"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"slam_map_{timestamp}.npy"
        
        # Save map with current robot pose
        robot_pose = (self.x, self.y, math.degrees(self.theta))
        self.slam_map.save_map(filename, robot_pose=robot_pose)
        
        # Also save map statistics
        stats = self.slam_map.get_map_statistics()
        stats_filename = f"slam_map_{timestamp}_stats.txt"
        with open(stats_filename, 'w') as f:
            f.write(f"SLAM Map Statistics\n")
            f.write(f"===================\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Robot Position: x={self.x:.3f}m, y={self.y:.3f}m, theta={math.degrees(self.theta):.1f}°\n")
            f.write(f"\nMap Info:\n")
            f.write(f"Size: {self.slam_map.width}m x {self.slam_map.height}m\n")
            f.write(f"Resolution: {self.slam_map.resolution}m\n")
            f.write(f"Grid: {self.slam_map.grid_width} x {self.slam_map.grid_height} cells\n")
            f.write(f"\nExploration:\n")
            f.write(f"Explored: {stats['explored_percent']:.1f}%\n")
            f.write(f"Free cells: {stats['free']}\n")
            f.write(f"Occupied cells: {stats['occupied']}\n")
            f.write(f"Unknown cells: {stats['unknown']}\n")
        print(f"Map statistics saved to {stats_filename}")
    
    def run(self):
        """Main control loop"""
        step_count = 0
        
        while self.robot.step(self.timestep) != -1:
            # Process keyboard input (for mode switching and manual control)
            manual_linear, manual_angular = self.get_keyboard_input()
            
            # Get control commands based on current mode
            if self.mode == 1:
                # Mode 1: Manual control
                linear_vel = manual_linear
                angular_vel = manual_angular
            
            elif self.mode == 2:
                # Mode 2: Auto exploration
                ranges = self.process_lidar_data()
                linear_vel, angular_vel = self.auto_explorer.get_control_command(
                    ranges, self.x, self.y, self.theta
                )
                
                # Check if exploration is complete
                if self.auto_explorer.is_exploration_complete():
                    print("\n=== Exploration Complete! ===")
                    self.save_slam_map()
                    print("Switching to manual mode...")
                    self.switch_mode(1)
            
            elif self.mode == 3:
                # Mode 3: Auto navigation with obstacle avoidance
                # Get LiDAR data for obstacle detection
                ranges = self.process_lidar_data()
                
                # Get planned navigation command
                linear_vel, angular_vel = self.path_planner.get_control_command(
                    self.x, self.y, self.theta
                )
                
                # Apply dynamic obstacle avoidance
                if ranges is not None:
                    linear_vel, angular_vel = self._apply_obstacle_avoidance(
                        linear_vel, angular_vel, ranges
                    )
                
                # STUCK DETECTION AND RECOVERY
                current_time = self.robot.getTime()
                current_pos = (self.x, self.y)
                distance_moved = math.sqrt(
                    (current_pos[0] - self.last_position[0])**2 + 
                    (current_pos[1] - self.last_position[1])**2
                )
                
                # Check if robot has moved significantly
                if distance_moved > self.min_movement_threshold:
                    # Robot is moving, update last position and reset timer
                    self.last_position = current_pos
                    self.position_update_time = current_time
                    self.in_recovery_mode = False
                else:
                    # Robot hasn't moved much
                    time_stuck = current_time - self.position_update_time
                    
                    if time_stuck > self.stuck_threshold and not self.in_recovery_mode:
                        # Robot is stuck for more than threshold time
                        print(f"\n🆘 ROBOT STUCK DETECTED! Stationary for {time_stuck:.1f}s")
                        print(f"   Position: ({self.x:.2f}, {self.y:.2f})")
                        print(f"   Attempting recovery: Replanning from current position...")
                        
                        # Enter recovery mode
                        self.in_recovery_mode = True
                        
                        # Try to replan from current position
                        self._replan_from_current_position()
                        
                        # Reset timer
                        self.position_update_time = current_time
                        self.last_position = current_pos
                
                # Update visualizer with current waypoint index
                self.visualizer.update_waypoint_index(self.path_planner.path_index)
                
                # Check if goal is reached
                if self.path_planner.is_goal_reached():
                    print("\n=== Navigation Complete! ===")
                    print("Switching to manual mode...")
                    self.switch_mode(1)
            
            else:
                linear_vel = 0.0
                angular_vel = 0.0
            
            # Convert to wheel speeds
            left_speed, right_speed = self.differential_drive_kinematics(linear_vel, angular_vel)
            
            # Set motor speeds
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)
            
            # Debug: Print motor speeds when they change
            if abs(left_speed) > 0.01 or abs(right_speed) > 0.01:
                print(f"Motor speeds: Left={left_speed:.2f}, Right={right_speed:.2f}")
            
            # Update position
            # Use GPS for position, odometry for orientation (simpler and more reliable)
            if self.use_gps:
                gps_values = self.gps.getValues()
                self.x = gps_values[0]
                self.y = gps_values[1]
            
            # Always use odometry for orientation (theta)
            # IMU coordinate system is too complex, odometry is more reliable for SLAM
            self.update_odometry_orientation()
            
            # Process LiDAR data
            ranges = self.process_lidar_data()
            obstacles = self.get_lidar_obstacles(ranges)
            
            # Track current velocities for motion filtering
            self.current_linear_vel = linear_vel
            self.current_angular_vel = angular_vel
            
            # Update SLAM map periodically (skip during fast motion)
            # IMPORTANT: In navigation mode (3), do NOT update the SLAM map
            # Only use the pre-built static map that was loaded from file
            if self.mode != 3:  # Only update map in manual (1) and exploration (2) modes
                self.map_update_counter += 1
                if self.map_update_counter >= self.map_update_interval and ranges is not None:
                    # Skip map update if robot is turning or moving too fast
                    # This prevents motion distortion in the point cloud
                    if abs(self.current_angular_vel) < self.angular_vel_threshold and \
                       abs(self.current_linear_vel) < self.linear_vel_threshold:
                        self.slam_map.update_map(self.x, self.y, self.theta, ranges)
                        self.map_update_counter = 0
                        
                        # Apply morphological filter every 5 map updates to remove noise
                        # More frequent filtering for no downsampling mode
                        if step_count % (self.map_update_interval * 5) == 0:
                            self.slam_map.apply_morphological_filter()
                    else:
                        # Reset counter even if we skip, so we don't accumulate skips
                        self.map_update_counter = 0
                        if step_count % 100 == 0:  # Log occasionally
                            print(f"Skipping map update: angular_vel={abs(self.current_angular_vel):.3f} rad/s (threshold={self.angular_vel_threshold})")
            
            # Update visualizer with robot pose
            self.visualizer.update_robot_pose(self.x, self.y, self.theta)
            
            # Print status every 100 steps (approximately 6.4 seconds)
            step_count += 1
            if step_count % 100 == 0:
                mode_names = {1: "Manual", 2: "Auto Explore", 3: "Auto Navigate"}
                print(f"\n=== Robot Status (Mode: {mode_names[self.mode]}) ===")
                print(f"Pose: x={self.x:.3f}m, y={self.y:.3f}m, theta={math.degrees(self.theta):.1f}°")
                print(f"Velocities: linear={linear_vel:.2f}m/s, angular={angular_vel:.2f}rad/s")
                print(f"Encoders: Left={self.left_encoder.getValue():.3f}, Right={self.right_encoder.getValue():.3f}")
                
                # Mode-specific status
                if self.mode == 2:
                    print(f"Explorer State: {self.auto_explorer.get_state_string()}")
                elif self.mode == 3:
                    if self.path_planner.current_path:
                        progress = (self.path_planner.path_index / len(self.path_planner.current_path)) * 100
                        print(f"Navigation Progress: {progress:.1f}%")
                        print(f"Current waypoint: {self.path_planner.path_index}/{len(self.path_planner.current_path)}")
                        if self.path_planner.path_index < len(self.path_planner.current_path):
                            target_x, target_y = self.path_planner.current_path[self.path_planner.path_index]
                            dx = target_x - self.x
                            dy = target_y - self.y
                            dist = math.sqrt(dx**2 + dy**2)
                            print(f"Target: ({target_x:.3f}, {target_y:.3f}), Distance: {dist:.3f}m")
                
                if ranges is not None:
                    # LiDAR statistics
                    valid_ranges = ranges[ranges < 5.0]
                    if len(valid_ranges) > 0:
                        print(f"\n=== LiDAR Data ===")
                        print(f"Total points: {len(ranges)}")
                        print(f"Valid detections: {len(valid_ranges)}")
                        print(f"Min distance: {np.min(valid_ranges):.3f}m")
                        print(f"Max distance: {np.max(valid_ranges):.3f}m")
                        print(f"Mean distance: {np.mean(valid_ranges):.3f}m")
                        print(f"Obstacles (<0.3m): {len(obstacles)}")
                        
                        if obstacles:
                            closest = min(obstacles, key=lambda x: x[1])
                            print(f"Closest obstacle: {closest[1]:.3f}m at {math.degrees(closest[0]):.1f}°")
                    else:
                        print(f"LiDAR: No valid detections")
                
                # SLAM map statistics
                map_stats = self.slam_map.get_map_statistics()
                print(f"\n=== SLAM Map ===")
                print(f"Explored: {map_stats['explored_percent']:.1f}%")
                print(f"Free cells: {map_stats['free']}")
                print(f"Occupied cells: {map_stats['occupied']}")
                print(f"Unknown cells: {map_stats['unknown']}")
                
                # Obstacle verification statistics
                verif_stats = self.slam_map.get_verification_statistics()
                print(f"\n=== Obstacle Verification ===")
                print(f"Status: {'Enabled' if verif_stats['verification_enabled'] else 'Disabled'}")
                if verif_stats['verification_enabled']:
                    print(f"Confirmed obstacles: {verif_stats['confirmed_obstacles']}")
                    print(f"Pending verification: {verif_stats['pending_obstacles']}")
                    print(f"Min observations required: {verif_stats['min_observations_required']}")
                    print(f"Max observations: {verif_stats['max_observations']}")
                print("=" * 40)

if __name__ == "__main__":
    controller = SimpleRobotController()
    controller.run()
