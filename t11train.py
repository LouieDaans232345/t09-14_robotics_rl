import time
import numpy as np
from stable_baselines3 import PPO
from t10envw import OT2Env
import os
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
import argparse
from clearml import Task

# ClearML
task = Task.init(project_name="Mentor Group E/Group 3", task_name="OT2_RL_Training")
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

def log_to_clearml(step, metric_name, value):
    task.get_logger().report_scalar(metric_name, "value", value=value, iteration=step)

# WandB
os.environ['WANDB_API_KEY'] = '11c5fccd0b07e41fc8bef045f744781d2f777121'
run = wandb.init(project="RL_train",sync_tensorboard=True)

# Env
env = OT2Env()

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

# Custom Callback (inspired by Barnabas Szalay)
class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomWandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
    
    def _on_step(self) -> bool:
        # Check if 'infos' exists in self.locals
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            episode_info = self.locals['infos'][0].get('episode', {})  # safely extract episode info

            # Log episode rewards
            if 'r' in episode_info:  # Episode reward
                self.episode_rewards.append(episode_info['r'])
                wandb.log({"episode_reward": episode_info['r']}, step=self.num_timesteps)
                log_to_clearml(self.num_timesteps, "episode_reward", episode_info['r'])

            # Log episode lengths
            if 'l' in episode_info:  # Episode length
                self.episode_lengths.append(episode_info['l'])
                wandb.log({"episode_length": episode_info['l']}, step=self.num_timesteps)

        # Success rate logging (if available in env info)
        success = self.locals['infos'][0].get('success', None) if 'infos' in self.locals else None
        if success is not None:
            self.success_rate.append(success)
            wandb.log({"success_rate": np.mean(self.success_rate)}, step=self.num_timesteps)

        # Log entropy (policy exploration)
        entropy = self.model.logger.name_to_value.get('entropy', None)
        if entropy is not None:
            wandb.log({"entropy": entropy}, step=self.num_timesteps)

        # Log learning rate
        wandb.log({"learning_rate": self.model.learning_rate}, step=self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        # Log aggregate statistics at the end of training
        wandb.log({
            "average_episode_reward": np.mean(self.episode_rewards),
            "average_episode_length": np.mean(self.episode_lengths),
            "success_rate": np.mean(self.success_rate)
        })
custom_wandb_callback = CustomWandbCallback()

# Classic Callback
wandb_callback = WandbCallback(
    model_save_freq = 1000,
    model_save_path = f"models/{run.id}",
    verbose = 2
)

# Train
timesteps = 100_000
iterations = 10

for i in range(10):
    model.learn(
        total_timesteps = timesteps,
        callback = [custom_wandb_callback, wandb_callback],
        progress_bar = True,
        reset_num_timesteps = False,
        tb_log_name = f"runs/{run.id}_i{i+1}"
    )
    model.save(f"models/{run.id}/{time_steps*(i+1)}")