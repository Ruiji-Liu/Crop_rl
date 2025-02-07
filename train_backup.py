# train.py
import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from main_backup import CropRowEnv  # Assuming your environment is in main.py
from gymnasium import Env
import torch
from stable_baselines3.common.env_util import make_vec_env
# Configuration
config = {
    "total_timesteps": 100000,
    "log_dir": "./logs",
    "save_freq": 10000,
    "policy": "MlpPolicy",
    "learning_rate": 1e-4,
    "buffer_size": 100000,
    "learning_starts": 10000,
    "batch_size": 128,
    "gamma": 0.99,
    "target_update_interval": 1000
}

def make_env():
    env = CropRowEnv(num_crop_rows=5, corridor_length=5)
    env = Monitor(env)  # Add monitoring
    return env

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Create environment
    # env = DummyVecEnv([make_env])
    env = make_vec_env(CropRowEnv, n_envs=1, env_kwargs={'num_crop_rows': 5, 'corridor_length': 5})
    
    # Create model
    model = DQN(
    config["policy"],
    env,
    verbose=1,
    learning_rate=config["learning_rate"],
    buffer_size=config["buffer_size"],
    learning_starts=config["learning_starts"],
    batch_size=config["batch_size"],
    gamma=config["gamma"],
    target_update_interval=config["target_update_interval"],
    tensorboard_log=config["log_dir"],
    device='cuda'  # Add this line to utilize GPU
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config["save_freq"],
        save_path=config["log_dir"],
        name_prefix="dqn_crop"
    )
    
    # Training
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=checkpoint_callback,
        tb_log_name="dqn"
    )
    
    # Save final model
    model.save(os.path.join(config["log_dir"], "dqn_crop_final"))
    env.close()

if __name__ == "__main__":
    train()