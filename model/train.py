import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime
import torch

from gymnasium_env.envs import GridWorldEnv

def train_model(
    model_path="policy/ppo_gridworld_model",
    total_timesteps=200000,
    env_config=None,
    save_unique=False
):
    """
    Train (or continue training) a PPO model on the GridWorldEnv.
    By default, constantly replaces previous model/policy with newly trained one
    
    :param model_path: str - File path to save or load the model.
    :param total_timesteps: int - How many timesteps to train for.
    :param env_config: dict - A dictionary of environment parameters (optional).
    :param save_unique: bool - Whether to save the model with a unique timestamp.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # If env_config is not provided, use some defaults
    if env_config is None:
        env_config = {
            "render_mode": None,
            "size": 5,
            "max_steps": 5000,
            "grass_count": 3,
            "ou_count": 5,
        }

    def make_env():
        # Make the environment. No rendering for training.
        return GridWorldEnv(**env_config)

    # Create a vectorized environment
    env = DummyVecEnv([make_env])

    # Check if a previously trained model exists
    if os.path.exists(model_path + ".zip"):
        # Load the existing model and continue training
        model = PPO.load(
            model_path, 
            env=env, 
            device=device
        )
        print(f"Loaded existing model from {model_path}")
    else:
        # Create a new model if none exists
        # MLPPolicy = A multi-layer perceptron policy (default for environments with vector observations).
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device  # Use GPU if exists or has CUDA
        )
        print("Created a new PPO model.")

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    if save_unique:
        # Save the model with a timestamp to differentiate policies created
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_model_path = f"{model_path}_{timestamp}"
        model.save(unique_model_path)
        print(f"Model saved at {unique_model_path}")
    else:
        # Save the updated model without a timestamp
        model.save(model_path)
        print(f"Model saved at {model_path}")


if __name__ == "__main__":
    # If I want to just train the model for an initial policy
    train_model()
