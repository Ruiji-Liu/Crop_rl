# test.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Add this line to suppress OpenMP error

import numpy as np
from stable_baselines3 import DQN
from main import CropRowEnv  # Import your custom environment

# Configuration (same as in train.py)
config = {
    "num_crop_rows": 6,
    "corridor_length": 6,
    'max_episode_steps': 50,
    "log_dir": "./logs",
    "model_dir": "./models",
}

def test():
    # Create the environment
    env = CropRowEnv(num_crop_rows=config["num_crop_rows"], corridor_length=config["corridor_length"], 
                     max_episode_steps=config["max_episode_steps"])

    # Load the trained model
    model_path = os.path.join(config["log_dir"], "dqn_crop_final.zip")
    # model_path = os.path.join(config["log_dir"], "dqn_crop_300000_steps.zip")
    # model_path = os.path.join(config["model_dir"], "kind_of_working.zip")
    
    model = DQN.load(model_path)

    # Reset the environment
    obs, _ = env.reset()
    done = False
    total_reward = 0
    action_num = 0
    # Run the trained agent in the environment
    while not done:
        # Use the model to predict the action
        action, _ = model.predict(obs, deterministic=True)
        action_num += 1
        print("action", action, action_num)
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