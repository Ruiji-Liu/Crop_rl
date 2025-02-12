import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Suppress OpenMP error

import numpy as np
from stable_baselines3 import PPO  # Use PPO since your model is PPO
from main import CropRowEnv  # Import your custom environment

# Configuration (same as in train.py)
config = {
    "num_crop_rows": 10,
    "corridor_length": 10,
    "log_dir": "./logs",
}

def test():
    # Create the environment
    env = CropRowEnv(num_crop_rows=config["num_crop_rows"], corridor_length=config["corridor_length"])
    
    # Load the trained PPO model
    model_path = os.path.join(config["log_dir"], "ppo_crop_50000_steps.zip")
    model = PPO.load(model_path)
    
    # Reset the environment
    obs, _ = env.reset()
    done = False
    total_reward = 0
    action_num = 0
    
    # Run the trained agent in the environment
    while not done:
        # Use the model to predict the action (deterministic for testing)
        action, _ = model.predict(obs, deterministic=True)
        action_num += 1
        print("Action:", action, "Step:", action_num)
        
        # Execute the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Accumulate the reward
        total_reward += reward
        
        # Render the environment
        env.render()
    
    # Close the environment
    env.close()
    print(f"Episode finished with total reward: {total_reward}")

if __name__ == "__main__":
    test()
