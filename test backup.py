# test.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Add this line to suppress OpenMP error

import numpy as np
from stable_baselines3 import DQN
from main import CropRowEnv  # Import your custom environment

# Configuration (same as in train.py)
config = {
    "num_crop_rows": 5,
    "corridor_length": 5,
    "log_dir": "./logs",
}

def test():
    # Create the environment
    env = CropRowEnv(num_crop_rows=config["num_crop_rows"], corridor_length=config["corridor_length"])

    # Load the trained model
    model_path = os.path.join(config["log_dir"], "dqn_crop_final.zip")
    model = DQN.load(model_path)

    # Reset the environment
    obs, _ = env.reset()
    done = False
    total_reward = 0

    # Run the trained agent in the environment
    while not done:
        # Use the model to predict the action
        action, _ = model.predict(obs, deterministic=True)
        
        # Execute the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Accumulate the reward
        total_reward += reward
        
        # Render the environment (optional)
        env.render()

    # Close the environment
    env.close()
    print(f"Episode finished with total reward: {total_reward}")

if __name__ == "__main__":
    test()