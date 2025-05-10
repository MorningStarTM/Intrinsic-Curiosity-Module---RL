import numpy as np
import os

def save_episode_rewards(rewards, save_dir, filename="episode_rewards.npy"):
    """
    Saves the episode rewards as a .npy file.

    Args:
        rewards (list or np.ndarray): List of episode rewards to save.
        save_dir (str): Directory path where the file will be saved.
        filename (str): Name of the file to save the rewards in. Default is 'episode_rewards.npy'.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    np.save(save_path, np.array(rewards))
    print(f"Episode rewards saved to {save_path}")
