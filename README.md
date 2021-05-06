# ReferenceModification
The accompanying repo for the paper ["Autonomous Obstacle Avoidance by Learning Policies for Reference Modification''](https://arxiv.org/abs/2102.11042). 

## Overview
This paper presents a new method for obstacle avoidance by using a pure pursuit path follower and a RL agent in parallel.
The path follower, follows a reference and the RL agent learns to modify the reference to avoid obstacles.

## Repo Overview
- The repo contains a copy of our code that was used in the evaluation of our architecture.
- We compare our method to:
  - Follow the Gap Method
  - Oracle optimisation based path planner
  - A mapless navigation planner common in the literature
- We built a custom simulator to simulate f1tenth cars, avoiding static obstacles in a corridor/forrest
- The corridor, is 20m long and have 4 obstacles which are randomly spawned for each episode

## Folders
- maps: contains the map file for the forest
- config: contains a configuration file for the vehicle with the parameters which are used
- Evals: Contains results from the evaluations, after each evaluation, a .txt and .csv file is saved
- Tests: contains training and testing scripts
- Vehicles: saves the neural network parameters for the RL agents and the training data

## Code
- **Simulator:** contains all the code for the simulator. It is built similarly to the gym env's with classic, step and reset methods.
- **Planners:** contains all the code for the different planners that we test. The planners receive the vehicle state and a lidar scan and return an action in the form of velocity and steering commands.
- Utils constains various utilities

## Installation
- Requirements:
  - PyTorch
  - Numpy
  - Matplotlib
  - casadi 
  - numba
  - scipy
- Installation
  - clone the repo onto your computer
  - navigate into the folder, ```cd ReferenceModification```
  - install it using pip ```python3 -m pip install -e .```
- Built on Linux Ubuntu system (20.04.2 LTS) using Python v3.8.5

## Citing
If you have found our work helpful, please cite as:
```latex
@article{evans2021autonomous,
  title={Autonomous Obstacle Avoidance by Learning Policies for Reference Modification},
  author={Evans, Benjamin and Jordaan, Hendrik W and Engelbrecht, Herman A},
  journal={arXiv preprint arXiv:2102.11042},
  year={2021}
}
```
