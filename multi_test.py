import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from stable_baselines3 import DQN
from main import CropRowEnv

config = {
    "num_crop_rows": 6,
    "corridor_length": 6,
    "max_episode_steps": 50,
    "log_dir": "./logs",
}

def test(num_episodes=20000):
    # Create environment once
    env = CropRowEnv(
        num_crop_rows=config["num_crop_rows"],
        corridor_length=config["corridor_length"],
        max_episode_steps=config["max_episode_steps"]
    )
    
    # Load the trained model
    model = DQN.load(os.path.join(config["log_dir"], "dqn_crop_final.zip"))
    
    success_count = 0
    steps_per_episode = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step_count += 1
            
            if terminated:
                success_count += 1
                
        steps_per_episode.append(step_count)
    
    env.close()
    
    print(f"Total Successful Episodes: {success_count}/{num_episodes}")
    print(f"Average Steps per Episode: {np.mean(steps_per_episode):.2f}")
    print(f"Min Steps: {np.min(steps_per_episode)}, Max Steps: {np.max(steps_per_episode)}")
    
if __name__ == "__main__":
    test()