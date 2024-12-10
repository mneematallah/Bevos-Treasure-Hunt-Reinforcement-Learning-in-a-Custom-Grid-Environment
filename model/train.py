import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
import torch
import numpy as np
from gymnasium_env.envs import GridWorldEnv

# Define a custom feature extractor (optional)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.network = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.network(observations)

def train_model(
    model_path="policy/ppo_gridworld_model",
    total_timesteps=1_000_000,
    env_config=None,
    save_unique=False,
):
    """
    Train (or continue training) a PPO model on the GridWorldEnv.
    Incorporates reward normalization, checkpointing, and custom policies.

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
            "penalty_scaling": 1.0
        }

    def make_env():
        # Make the environment with monitoring for debugging
        env = GridWorldEnv(**env_config)
        return Monitor(env)

    # Create a vectorized environment with reward normalization
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    # Set up logging
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # Check if a previously trained model exists
    if os.path.exists(model_path + ".zip"):
        # Load the existing model and continue training
        model = PPO.load(
            model_path, 
            env=env, 
            device=device
        )
        model.set_logger(new_logger)
        print(f"Loaded existing model from {model_path}")
    else:
        # Create a new model if none exists
        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=2.5e-4,  # Adjusted for stability
            n_steps=4096,  # Larger batch size for better gradient estimates
            batch_size=256,  # Match batch size to available hardware
            clip_range=0.1, 
            verbose=1,
            device=device,  # Use GPU if available
        )
        model.set_logger(new_logger)
        print("Created a new PPO model.")

    # Set up a checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,  
        save_path="./checkpoints/",
        name_prefix="ppo_gridworld"
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

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
