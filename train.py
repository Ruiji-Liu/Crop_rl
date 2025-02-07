import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from main import CropRowEnv
import matplotlib.pyplot as plt

# Enable interactive mode so that matplotlib updates continuously.
# plt.ion()

# Configuration
config = {
    "total_timesteps": 100000,
    "log_dir": "./logs",
    "save_freq": 10000,
    "policy": "MlpPolicy",
    "learning_rate": 1e-4,
    "buffer_size": 10000,
    "learning_starts": 10000,
    "batch_size": 256,
    "gamma": 0.99,
    "target_update_interval": 1000,
    # Render every N steps (set to 1 for every step or higher for less frequent rendering)
    "render_freq": 1
}

# Custom callback to track successful episodes
class SuccessCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SuccessCallback, self).__init__(verbose)
        self.success_count = 0

    def _on_step(self) -> bool:
        # In vectorized environments, dones and rewards are lists.
        dones = self.locals.get("dones")
        rewards = self.locals.get("rewards")
        if dones is not None and rewards is not None:
            if dones[0] and rewards[0] > 0:
                self.success_count += 1
                print(f"Success! Total successful episodes: {self.success_count}")
        return True

# Custom callback to render the environment during training.
class VisualizeCallback(BaseCallback):
    def __init__(self, render_freq=1, verbose=0):
        """
        :param render_freq: Call render() every N steps.
        """
        super(VisualizeCallback, self).__init__(verbose)
        self.render_freq = render_freq
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        # Render every render_freq steps
        if self.step_count % self.render_freq == 0:
            # For vectorized environments, use the first one
            self.training_env.envs[0].render()
        return True

def train():
    # Create the vectorized environment.
    env = make_vec_env(
        CropRowEnv, 
        n_envs=1, 
        env_kwargs={'num_crop_rows': 5, 'corridor_length': 5}
    )

    # Create the model.
    model = DQN(
        config["policy"],
        env,
        # policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        target_update_interval=config["target_update_interval"],
        tensorboard_log=config["log_dir"],
        device='cuda',
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,  # Keep exploration active longer
        exploration_fraction=0.5,
    )

    # Create callbacks.
    success_callback = SuccessCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=config["save_freq"],
        save_path=config["log_dir"],
        name_prefix="dqn_crop"
    )
    visualize_callback = VisualizeCallback(render_freq=config["render_freq"])

    # Train the model with all callbacks.
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=checkpoint_callback, #, checkpoint_callback] #, visualize_callback],
        # callback = [checkpoint_callback, visualize_callback],
        tb_log_name="dqn"
    )

    # Save the final model.
    model.save(os.path.join(config["log_dir"], "dqn_crop_final"))
    env.close()

if __name__ == "__main__":
    train()
