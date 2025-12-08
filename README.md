1. Project Overview

This project implements a robot that can explore a maze, build a SLAM map, and navigate to the exit inside Webots.
Our system uses three main parts:
	1.	SLAM Mapping 
	2.	Path Planning + Navigation 
	3.	Machine Learning Model (ML) 
	•	Trains a small model using real robot logs
	•	Predicts next robot motion based on previous state
	•	Used only for analysis and testing, not for controlling the robot

The goal of the project is to understand how SLAM, planning, and learning fit together in a full robotic pipeline.


2. Project Structure:

Project Structure
-----------------
controllers/        → SLAM, exploration, and navigation code 
ml/                 → My machine learning scripts (dataset, training, evaluation)
logs/               → Real robot run logs (for ML)
protos/             → Robot model files
worlds/             → Webots world and maze setup



Authors
-------
Omar Khaled – Machine Learning + Testing
Xiangyao Guo – SLAM
Shang Wang – Navigation
Ziad Tarek – Path Planning
