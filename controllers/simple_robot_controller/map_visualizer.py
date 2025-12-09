"""
Real-time Map Visualizer
Display SLAM map in a separate window during robot operation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

class MapVisualizer:
    def __init__(self, slam_map, update_interval=1000):
        """
        Initialize real-time map visualizer
        
        Args:
            slam_map: OccupancyGridMap instance
            update_interval: Update interval in milliseconds
        """
        self.slam_map = slam_map
        self.update_interval = update_interval
        self.fig = None
        self.ax = None
        self.im = None
        self.robot_marker = None
        self.robot_arrow = None
        self.robot_pose = None  # (x, y, theta)
        self.planned_path = None  # List of (x, y) waypoints
        self.current_waypoint_index = 0
        self.path_line = None
        self.target_marker = None
        self.running = False
        
    def start(self):
        """Start the visualizer in a separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run_visualizer, daemon=True)
        self.thread.start()
        print("Map visualizer started (separate window)")
    
    def _run_visualizer(self):
        """Run the matplotlib visualization"""
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        
        # Initial plot
        grid, info = self.slam_map.get_map_data()
        self.im = self.ax.imshow(grid, cmap='gray_r', origin='lower', 
                                 vmin=0, vmax=1, interpolation='nearest')
        
        self.ax.set_title('SLAM Map (Real-time)\nBlack=Free, Gray=Unknown, White=Occupied')
        self.ax.set_xlabel(f'Grid X (Resolution: {info["resolution"]}m)')
        self.ax.set_ylabel(f'Grid Y (Resolution: {info["resolution"]}m)')
        
        cbar = plt.colorbar(self.im, ax=self.ax, label='Occupancy Probability')
        self.ax.grid(True, alpha=0.2, linewidth=0.5)
        
        plt.tight_layout()
        
        # Update loop
        while self.running:
            try:
                self.update()
                plt.pause(self.update_interval / 1000.0)
            except Exception as e:
                print(f"Visualizer error: {e}")
                break
        
        plt.close(self.fig)
    
    def update_robot_pose(self, x, y, theta):
        """Update robot pose for visualization
        
        Args:
            x, y: Robot position in world coordinates
            theta: Robot orientation in radians
        """
        self.robot_pose = (x, y, theta)
    
    def set_planned_path(self, path, current_index=0):
        """Set the planned navigation path
        
        Args:
            path: List of (x, y) waypoints in world coordinates
            current_index: Current waypoint index
        """
        self.planned_path = path
        self.current_waypoint_index = current_index
    
    def update_waypoint_index(self, index):
        """Update current waypoint index"""
        self.current_waypoint_index = index
    
    def update(self):
        """Update the visualization"""
        if self.im is not None:
            grid, info = self.slam_map.get_map_data()
            self.im.set_data(grid)
            
            # Update title with statistics
            stats = self.slam_map.get_map_statistics()
            title = f'SLAM Map (Real-time) - Explored: {stats["explored_percent"]:.1f}%\n'
            title += f'Free: {stats["free"]} | Occupied: {stats["occupied"]} | Unknown: {stats["unknown"]}'
            
            # Add robot pose to title if available
            if self.robot_pose:
                x, y, theta = self.robot_pose
                title += f'\nRobot: ({x:.2f}, {y:.2f}, {np.degrees(theta):.0f}°)'
            
            self.ax.set_title(title)
            
            # Draw planned path if available
            if self.planned_path:
                # Remove old path
                if self.path_line:
                    self.path_line.remove()
                if self.target_marker:
                    self.target_marker.remove()
                
                # Convert path to grid coordinates
                path_grid_x = []
                path_grid_y = []
                for wx, wy in self.planned_path:
                    gx, gy = self.slam_map.world_to_grid(wx, wy)
                    path_grid_x.append(gx)
                    path_grid_y.append(gy)
                
                # Draw full path (gray for completed, blue for remaining)
                if self.current_waypoint_index > 0:
                    # Completed path (gray)
                    self.ax.plot(path_grid_x[:self.current_waypoint_index+1], 
                               path_grid_y[:self.current_waypoint_index+1],
                               'gray', linewidth=2, alpha=0.5, zorder=5)
                
                if self.current_waypoint_index < len(self.planned_path):
                    # Remaining path (blue)
                    self.path_line, = self.ax.plot(path_grid_x[self.current_waypoint_index:], 
                                                    path_grid_y[self.current_waypoint_index:],
                                                    'b-', linewidth=3, alpha=0.7, zorder=6)
                    
                    # Draw current target waypoint (red)
                    target_gx = path_grid_x[self.current_waypoint_index]
                    target_gy = path_grid_y[self.current_waypoint_index]
                    self.target_marker = self.ax.scatter(target_gx, target_gy, c='red', s=150,
                                                        marker='X', edgecolors='black',
                                                        linewidths=2, zorder=9,
                                                        label='Target')
            
            # Draw robot position if available
            if self.robot_pose:
                x, y, theta = self.robot_pose
                grid_x, grid_y = self.slam_map.world_to_grid(x, y)
                
                # Remove old markers
                if self.robot_marker:
                    self.robot_marker.remove()
                if self.robot_arrow:
                    self.robot_arrow.remove()
                
                # Draw new markers
                self.robot_marker = self.ax.scatter(grid_x, grid_y, c='lime', s=150, 
                                                   marker='o', edgecolors='black', 
                                                   linewidths=2, zorder=10, label='Robot')
                
                # Draw orientation arrow
                arrow_length = 15
                dx = arrow_length * np.cos(theta)
                dy = arrow_length * np.sin(theta)
                self.robot_arrow = self.ax.arrow(grid_x, grid_y, dx, dy, 
                                                head_width=5, head_length=5,
                                                fc='yellow', ec='black', 
                                                linewidth=2, zorder=11)
            
            # Compatible with different matplotlib versions
            try:
                self.fig.canvas.draw_flush_events()
            except AttributeError:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
    
    def stop(self):
        """Stop the visualizer"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        print("Map visualizer stopped")
