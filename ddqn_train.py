import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from main import CropRowEnv

# Configuration
config = {
    "total_timesteps": 200000,
    "log_dir": "./dqn_logs",
    "save_freq": 20000,
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "buffer_size": 100000,
    "learning_starts": 20000,
    "batch_size": 128,
    "gamma": 0.99,
    "target_update_interval": 2000,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.02,
    "exploration_fraction": 0.4,
    "env_kwargs": {
        "num_crop_rows": 10,
        "corridor_length": 12
    }
}

def train():
    # Create the environment
    env = make_vec_env(
        CropRowEnv,
        n_envs=1,
        env_kwargs=config["env_kwargs"]
    )

    # Use Double DQN with deeper network
    model = DQN(
        config["policy"],
        env,
        policy_kwargs=dict(
            net_arch=[256, 256],               # Deeper network
            activation_fn=torch.nn.ReLU
        ),
        verbose=1,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        target_update_interval=config["target_update_interval"],
        tensorboard_log=config["log_dir"],
        exploration_initial_eps=config["exploration_initial_eps"],
        exploration_final_eps=config["exploration_final_eps"],
        exploration_fraction=config["exploration_fraction"],
        device='cuda' if torch.cuda.is_available() else 'auto'
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config["save_freq"],
        save_path=config["log_dir"],
        name_prefix="dqn_crop"
    )

    # Train
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