import os
import cv2  # For saving the video
import numpy as np
from stable_baselines3 import PPO
from gymnasium_env.envs import GridWorldEnv
import torch
import pygame


def evaluate(model_path, env_config=None, max_episodes=5, video_output_dir="videos"):
    """
    Runs a trained PPO model on the GridWorldEnv and visualizes its behavior.
    Captures the `pygame` rendering to save as a video for each episode.

    :param model_path: str - File path to the saved model.
    :param env_config: dict - A dictionary of environment parameters (optional).
    :param max_episodes: int - Number of episodes to run for evaluation.
    :param video_output_dir: str - Directory to save episode videos.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if env_config is None:
        env_config = {
            "render_mode": "human",
            "size": 5,
            "max_steps": 5000,
            "grass_count": 3,
            "ou_count": 5,
        }

    env = GridWorldEnv(**env_config)

    model = PPO.load(model_path, device=device)
    print(f"Loaded model from {model_path}")

    os.makedirs(video_output_dir, exist_ok=True)

    try:
        for episode in range(max_episodes):
            print(f"Starting Episode {episode + 1}")
            observation, info = env.reset()
            done = False
            total_reward = 0

            video_path = os.path.join(video_output_dir, f"episode_{episode + 1}.mp4")
            frame_size = (env.window_size, env.window_size + 50) 
            fps = 10  
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

            while not done:
                action, _ = model.predict(observation, deterministic=True)
                action = int(action)
                observation, reward, done, truncated, info = env.step(action)
                total_reward += reward

                pygame_frame = pygame.surfarray.array3d(env.window)  
                pygame_frame = np.transpose(pygame_frame, (1, 0, 2))  
                video_writer.write(cv2.cvtColor(pygame_frame, cv2.COLOR_RGB2BGR))  

            print(f"Episode {episode + 1} finished with total reward: {total_reward}")
            video_writer.release()  
    finally:
        env.close()
