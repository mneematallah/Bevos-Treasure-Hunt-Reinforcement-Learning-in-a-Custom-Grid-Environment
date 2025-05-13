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
    total_timesteps=1_000_000,    # Increased training time
    env_config=None,
    save_unique=False,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if env_config is None:
        env_config = {
            "render_mode": None,
            "size": 5,
            "max_steps": 1000,     
            "grass_count": 3,
            "ou_count": 5,
            "penalty_scaling": 0.05
        }

    def make_env():
        env = GridWorldEnv(**env_config)
        return Monitor(env)

    venv = DummyVecEnv([make_env])
    env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_reward=10.0)

    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=env, device=device)
        model.set_logger(new_logger)
        print(f"Loaded existing model from {model_path}")
    else:
        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=5e-5,   
            n_steps=4096,          
            batch_size=256,
            clip_range=0.1,
            ent_coef=0.01,       
            gamma=0.99,
            verbose=1,
            device=device,
        )
        model.set_logger(new_logger)
        print("Created a new PPO model.")

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix="ppo_gridworld"
    )

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    env.save("policy/ppo_gridworld_model_norm")

    if save_unique:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_model_path = f"{model_path}_{timestamp}"
        model.save(unique_model_path)
        print(f"Model saved at {unique_model_path}")
    else:
        model.save(model_path)
        print(f"Model saved at {model_path}")
