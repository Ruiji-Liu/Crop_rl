import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from main import CropRowEnv  # Ensure this imports your custom environment
import matplotlib.pyplot as plt

# Enable interactive plotting if you want to visualize during training.
# plt.ion()

# Configuration for PPO training
config = {
    "total_timesteps": 100000,           # Total timesteps for training
    "log_dir": "./logs",                 # Directory for logs and checkpoints
    "save_freq": 10000,                  # Frequency (in timesteps) to save a checkpoint
    "policy": "MlpPolicy",               # Policy network type; MLP is suitable for low-dimensional inputs
    "learning_rate": 3e-4,               # PPO's default learning rate
    "n_steps": 2048,                     # Number of steps to run for each environment per update
    "batch_size": 64,                    # Mini-batch size for optimization
    "gamma": 0.99,                       # Discount factor
    "clip_range": 0.2,                   # Clipping range for PPO (helps stabilize training)
    "ent_coef": 0.0,                     # Entropy coefficient (set to >0 to encourage exploration)
    "render_freq": 1                     # Render frequency (set to 1 to render every step; adjust for speed)
}

# A custom callback to visualize the environment during training
class VisualizeCallback(BaseCallback):
    def __init__(self, render_freq=1, verbose=0):
        """
        :param render_freq: How often (in steps) to call env.render()
        """
        super(VisualizeCallback, self).__init__(verbose)
        self.render_freq = render_freq
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.render_freq == 0:
            # For vectorized environments, use the first instance for rendering.
            self.training_env.envs[0].render()
        return True

def train():
    # Create a vectorized environment with one copy of CropRowEnv.
    env = make_vec_env(
        CropRowEnv, 
        n_envs=1, 
        env_kwargs={'num_crop_rows': 10, 'corridor_length': 10}
    )

    # Create the PPO model. You can pass additional policy keyword arguments if desired.
    model = PPO(
        config["policy"],
        env,
        verbose=1,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        clip_range=config["clip_range"],
        tensorboard_log=config["log_dir"],
        device='cuda' if torch.cuda.is_available() else 'cpu'
        # Optionally add policy_kwargs=dict(net_arch=[256, 256]) to increase network capacity.
    )

    # Create a checkpoint callback to save the model periodically.
    checkpoint_callback = CheckpointCallback(
        save_freq=config["save_freq"],
        save_path=config["log_dir"],
        name_prefix="ppo_crop"
    )

    # Create the visualization callback.
    visualize_callback = VisualizeCallback(render_freq=config["render_freq"])

    # Train the model with the specified callbacks.
    model.learn(
        total_timesteps=config["total_timesteps"],
        # callback=[checkpoint_callback, visualize_callback],
        callback=checkpoint_callback,
        tb_log_name="ppo"
    )

    # Save the final trained model.
    model.save(os.path.join(config["log_dir"], "ppo_crop_final"))
    env.close()

if __name__ == "__main__":
    train()
