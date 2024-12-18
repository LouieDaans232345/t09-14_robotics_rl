from math import pi
from os import pipe
import gymnasium as gym
import numpy as np
from t10envw import OT2Env
import time

env = OT2Env()

num_episodes = 5
for episode in range(num_episodes):
    obs = env.reset()
    print()
    print(env.goal_position)
    terminated = False
    truncated = False
    step = 0

    while not terminated and not truncated:
        # Take a random action from the environment's action space
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        if terminated or truncated:
            print(f"Episode: {episode + 1}, Step: {step + 1}, Action: {action}, Reward: {reward}")
            first_robot_key = list(info.keys())[0]
            pip_pos = np.array(info[first_robot_key]["pipette_position"], dtype=np.float32)
            print(f"Episode finished after {step} steps. Pipette position: {pip_pos}")
            break