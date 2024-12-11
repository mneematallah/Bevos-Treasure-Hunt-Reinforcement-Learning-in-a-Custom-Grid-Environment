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

    # Default environment configuration
    if env_config is None:
        env_config = {
            "render_mode": "human",  # Use `human` for pygame rendering
            "size": 5,
            "max_steps": 5000,
            "grass_count": 3,
            "ou_count": 5,
        }

    # Create the environment
    env = GridWorldEnv(**env_config)

    # Load the trained model
    model = PPO.load(model_path, device=device)
    print(f"Loaded model from {model_path}")

    # Ensure the video output directory exists
    os.makedirs(video_output_dir, exist_ok=True)

    try:
        for episode in range(max_episodes):
            print(f"Starting Episode {episode + 1}")
            observation, info = env.reset()
            done = False
            total_reward = 0

            # Prepare video writer for this episode
            video_path = os.path.join(video_output_dir, f"{model_path}episode_{episode + 1}.mp4")
            frame_size = (env.window_size, env.window_size + 50)  # Match pygame canvas size
            fps = 10  # Frames per second
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

            while not done:
                # Get the best action from the model
                action, _ = model.predict(observation, deterministic=True)
                action = int(action)
                # Take the action in the environment
                observation, reward, done, truncated, info = env.step(action)
                # Accumulate rewards
                total_reward += reward

                # Capture the `pygame` surface as a frame
                pygame_frame = pygame.surfarray.array3d(env.window)  # Get frame from pygame
                pygame_frame = np.transpose(pygame_frame, (1, 0, 2))  # Transpose for OpenCV (H, W, C)
                video_writer.write(cv2.cvtColor(pygame_frame, cv2.COLOR_RGB2BGR))  # Write frame to video

            print(f"Episode {episode + 1} finished with total reward: {total_reward}")
            video_writer.release()  # Save the video for the episode
    finally:
        # Ensure the environment is closed properly
        env.close()
