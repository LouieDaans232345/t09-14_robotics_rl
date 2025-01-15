import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sim_class import Simulation
import random
import math
import rl_functions as do


class OT2Env(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()

        # Initialize the simulation
        self.sim_sim = Simulation(num_agents=1, render=False)

        self.x_min = do.find_limit(self.sim_sim, axis=0, direction=-1, silence=True)
        self.x_max = do.find_limit(self.sim_sim, axis=0, direction=1, silence=True)
        self.y_min = do.find_limit(self.sim_sim, axis=1, direction=-1, silence=True)
        self.y_max = do.find_limit(self.sim_sim, axis=1, direction=1, silence=True)
        self.z_min = do.find_limit(self.sim_sim, axis=2, direction=-1, silence=True)
        self.z_max = do.find_limit(self.sim_sim, axis=2, direction=1, silence=True)

        self.sim_sim.close()

        self.sim = Simulation(num_agents=1, render=render)

        # Define variables
        self.render = render
        self.max_steps = max_steps
        self.steps = 0

        # Action space: controlling pipette in x, y, z directions
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]),
                                            shape=(3,), dtype=np.float32)

        # Observation space: pipette position (x, y, z) and goal position (x, y, z)
        self.observation_space = spaces.Box(low=-math.inf, high=math.inf,
                                            shape=(6,), dtype=np.float32) # for one robot

    def step(self, action):
        # Append 0 for the drop action (since we're controlling only 3 actions: x, y, z)
        action = np.append(action, 0)  # appending 0 for drop action

        # r5 - Info (+ run action)
        info = self.sim.run([action]) # expects list of actions [[x,y,z,drop], [x,y,z,drop], ...]

        # r1 - Observation
        first_robot_key = list(info.keys())[0]
        pipette_position = np.array(info[first_robot_key]["pipette_position"], dtype=np.float32)
        observation = np.concatenate((pipette_position, self.goal_position)).astype(np.float32)

        # r2 - Reward
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        if hasattr(self, 'previous_distance'):
            distance_reward = self.previous_distance - distance_to_goal
        else:
            distance_reward = 0
        self.previous_distance = distance_to_goal

        # main reward -> negate the distance to goal
        reward = -distance_to_goal
        # -> add penalty if not closer to goal than before
        if distance_reward < 0:
            reward += distance_reward
        # -> add penalty for taking a step
        reward -= 0.05

        # r3 - Terminated
        threshold = 0.001
        if distance_to_goal < threshold: # if task has been completed
            terminated = True
            reward += 1  # Positive reward for success
            print(f'Treshold: {threshold}, Distance to goal: {distance_to_goal}')
        else:
            terminated = False
        
        # r4 - Truncated
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.goal_position = [random.uniform(self.x_min, self.x_max), random.uniform(self.y_min, self.y_max), random.uniform(self.z_min, self.z_max)]
        
        # r2 - Info (+ reset)
        info = self.sim.reset(num_agents=1)

        # r1 - Observation
        first_robot_key = list(info.keys())[0]
        pipette_position = np.array(info[first_robot_key]["pipette_position"], dtype=np.float32)
        observation = np.concatenate((pipette_position, self.goal_position)).astype(np.float32)

        # Reset variables
        self.steps = 0

        return observation, info

    def render(self, mode="human"):
        pass

    def close(self):
        self.sim.close()