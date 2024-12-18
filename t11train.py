import time
import numpy as np
from stable_baselines3 import PPO
from t10envw import OT2Env
import os
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse
from clearml import Task
import sys
import gymnasium as gym
import tensorflow

# Env
env = OT2Env()

# ClearML
task = Task.init(project_name="Mentor Group E/Group 3", task_name="OT2_RL_Training",
                 docker_virtualenv='/root/.clearml/venvs-builds/3.10/bin/python')
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

# WandB
os.environ['WANDB_API_KEY'] = '11c5fccd0b07e41fc8bef045f744781d2f777121'
run = wandb.init(project="RL_train",sync_tensorboard=True)

# Arg parsing
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0005)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
args, _ = parser.parse_known_args()

# Define PPO
model = PPO(
    'MlpPolicy',
    env,
    verbose=1, 
    learning_rate=args.learning_rate, 
    batch_size=args.batch_size, 
    n_steps=args.n_steps, 
    n_epochs=args.n_epochs, 
    tensorboard_log=f"runs/{run.id}",
)


# Classic Callback
wandb_callback = WandbCallback(
    model_save_freq = 100_000,
    model_save_path = f"models/{run.id}",
    verbose = 2
)

# Train
model.learn(total_timesteps=6000000, callback=wandb_callback, progress_bar=True, tb_log_name=f"runs/{run.id}")