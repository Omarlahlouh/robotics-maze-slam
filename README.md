Project Overview

This project implements a robot that can explore a maze, build a SLAM map, and navigate to the exit inside Webots.

The system is made of three main parts:
	1.	SLAM Mapping
Generates an occupancy grid map from LiDAR + odometry.
	2.	Path Planning & Navigation
	•	A* search
	•	Waypoint following
	•	Movement toward the maze exit
	3.	Machine Learning Model (ML) – Omar’s Component
	•	Trains a small neural network using real robot logs
	•	Predicts the next robot position from the current state
	•	Used only for analysis/testing (not for controlling the robot)

⸻

Goal of the Project

To understand how SLAM, planning, and learning fit together in a full robotic pipeline, and to evaluate autonomous robot behaviour inside a structured maze.




Project Structure

controllers/ → SLAM, exploration, and navigation code  
ml/          → ML scripts (dataset building, training, evaluation)  
logs/        → Real robot run logs used for ML  
protos/      → Robot model files  
worlds/      → Webots world and maze setup


⸻

Implementation Details

Manually Implemented Components

Core Robot Logic
	•	simple_robot_controller.py
Handles:
	•	Keyboard control
	•	Exploration mode
	•	SLAM updates
	•	Navigation control

Autonomous Exploration
	•	auto_explorer.py
Implements:
	•	Right-hand wall-following
	•	Autonomous maze exploration

Path Planning
	•	path_planner.py
Provides:
	•	A* search
	•	Path smoothing
	•	Waypoint-following logic

Machine Learning (ML) – Omar Khaled
	•	ML scripts: prepare_dataset.py, train_model.py, evaluate_model.py
	•	Builds dataset from real robot logs
	•	Trains a small neural network
	•	Evaluates prediction accuracy (position error, MSE)

⸻

Pre-Programmed / External Code Used

Webots Built-In Tools

Used for:
	•	Sensors
	•	Motors
	•	LiDAR handling
	•	Simulation loop

Utility Scripts (Adapted from Webots Samples)
	•	map_visualizer.py
	•	map_editor.py
	•	map_optimizer.py
	•	verify_path.py
	•	visualize_navigation.py

Used for debugging, visualisation, and map checking.

Third-Party Libraries
	•	PyTorch — neural network model + optimisation
	•	NumPy — dataset handling
	•	Matplotlib — optional plotting
	•	Standard Python modules — csv, json, math, etc.

⸻

Authors
	•	Omar Khaled – Machine Learning + Testing
	•	Xiangyao Guo – SLAM
	•	Shang Wang – Navigation
	•	Ziad Tarek – Path Planning

