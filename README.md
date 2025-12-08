Project Overview

This project implements a robot in Webots that can explore a maze, build a SLAM map, and navigate to the exit.
The system is built from three main parts:
	1.	SLAM Mapping 
	2.	Path Planning + Navigation 
	3.	Machine Learning Model (ML) 
	•	Trains a small model using real robot logs
	•	Predicts the next robot position from the current state
	•	Used only for analysis and testing (not for controlling the robot)

The goal of the project is to show how SLAM, planning, and learning can work together in a full robotic pipeline.

Project Structure

controllers/ → SLAM, exploration, and navigation code  
ml/          → ML scripts (dataset building, training, evaluation)  
logs/        → Real robot run logs used for ML  
protos/      → Robot model files  
worlds/      → Webots world and maze setup

Implementation Details

Manually Implemented Components

The following parts of the project were written by us specifically for this assignment:
	•	Simple Robot Controller (simple_robot_controller.py)
Handles keyboard control, exploration mode, SLAM updates, and navigation control.
	•	Auto Explorer (auto_explorer.py)
Right-hand wall-following behaviour for autonomous maze exploration.
	•	Path Planner (path_planner.py)
A* search implementation, path smoothing, and waypoint-following logic.
	•	ML Scripts (prepare_dataset.py, train_model.py, evaluate_model.py)
Dataset creation from logs, neural-network training, and evaluation.
	•	Logger (simple_logger_controller.py)
Used to record real robot movement logs for training the ML model.

Pre-Programmed / External Code Used

The project also uses several existing tools and helpers:
	•	Webots built-in API
Sensors, motors, LiDAR handling, controller structure, robot simulation loop.
	•	Utility scripts adapted from Webots sample projects:
	•	map_visualizer.py
	•	map_editor.py
	•	map_optimizer.py
	•	verify_path.py
	•	visualize_navigation.py
These were used for debugging, visualisation, and map checking.
	•	Third-party Python libraries:
	•	PyTorch — neural network model + optimisation
	•	NumPy — dataset handling
	•	Matplotlib — optional plotting
	•	Standard Python modules (csv, json, math, etc.)


Authors

•	Omar Khaled – Machine Learning + Testing
•	Xiangyao Guo – SLAM
•	Shang Wang – Navigation
•	Ziad Tarek – Path Planning
